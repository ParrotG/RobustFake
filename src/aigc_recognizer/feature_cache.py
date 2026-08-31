"""Deterministic, resumable frozen-backbone feature precomputation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.dataset import AIGCManifestDataset, validate_preparation
from aigc_recognizer.model import EncodedViews, FrozenClipDetector, create_detector
from aigc_recognizer.model import (
    RESIDUAL_STATISTICS_VERSION,
    ResidualStatisticsExtractor,
)
from aigc_recognizer.utils import atomic_torch_save, seed_everything, seed_worker

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 1
_TENSOR_FIELDS = (
    "clean_final",
    "transformed_final",
    "clean_intermediate",
    "transformed_intermediate",
    "label",
)
_TEXT_FIELDS = (
    "id",
    "source_dataset",
    "real_source",
    "generator_family",
    "architecture",
    "domain",
)


def _resolve_device(config: AppConfig) -> torch.device:
    requested = config.training.device
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_identity(config: AppConfig) -> dict[str, Any]:
    """Return every immutable input that changes frozen image features."""
    manifest_path = Path(config.data.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "manifest_sha256": _file_sha256(manifest_path),
        "seed": config.project.seed,
        "storage_dtype": config.feature_cache.dtype,
        "backbone": {
            "name": config.model.backbone_name,
            "pretrained": config.model.pretrained,
            "embedding_dim": config.model.embedding_dim,
            "intermediate_layers": config.model.intermediate_layers,
            "intermediate_dim": config.model.intermediate_dim,
        },
        "views": dataclasses.asdict(config.views),
        "standardization": dataclasses.asdict(config.standardization),
        "augmentations": dataclasses.asdict(config.augmentations),
    }


def cache_key(config: AppConfig) -> str:
    """Hash the feature-producing configuration and manifest contents."""
    payload = json.dumps(cache_identity(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_directory(config: AppConfig) -> Path:
    """Resolve a collision-resistant human-readable cache directory."""
    safe_backbone = config.model.backbone_name.lower().replace("/", "-")
    safe_weights = config.model.pretrained.lower().replace("/", "-")
    return Path(config.feature_cache.root_dir) / (
        f"{safe_backbone}-{safe_weights}-{cache_key(config)[:16]}"
    )


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _empty_manifest(config: AppConfig) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key(config),
        "identity": cache_identity(config),
        "dtype": config.feature_cache.dtype,
        "complete": False,
        "splits": {},
    }


def _load_or_create_manifest(config: AppConfig, directory: Path) -> dict[str, Any]:
    path = directory / "cache_manifest.json"
    if not path.is_file():
        manifest = _empty_manifest(config)
        _atomic_json(manifest, path)
        return manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("cache_key") != cache_key(config)
        or manifest.get("identity") != cache_identity(config)
        or manifest.get("dtype") != config.feature_cache.dtype
    ):
        raise RuntimeError("Existing feature cache metadata does not match the active configuration.")
    return manifest


def _autocast(config: AppConfig, device: torch.device) -> Any:
    if not config.training.amp or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loader(config: AppConfig, dataset: Dataset[Any]) -> DataLoader[Any]:
    cache = config.feature_cache
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": cache.batch_size,
        "shuffle": False,
        "num_workers": cache.num_workers,
        "pin_memory": cache.pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if cache.num_workers > 0:
        arguments["prefetch_factor"] = cache.prefetch_factor
        arguments["persistent_workers"] = False
    return DataLoader(**arguments)


def _existing_prefix(directory: Path, entries: list[dict[str, Any]]) -> int:
    expected_start = 0
    for entry in entries:
        if int(entry["start"]) != expected_start or int(entry["end"]) <= expected_start:
            raise RuntimeError("Feature-cache shards are not a contiguous prefix.")
        path = directory / str(entry["path"])
        if not path.is_file():
            raise RuntimeError(f"Feature-cache shard is missing: {path}")
        if "size" in entry and path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"Feature-cache shard size mismatch: {path}")
        expected_start = int(entry["end"])
    return expected_start


def _encoded_payload(
    clean: EncodedViews,
    transformed: EncodedViews,
    batch: dict[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    count = clean.final.shape[0]
    empty = torch.empty(count, clean.final.shape[1], 0, 0, dtype=dtype)
    return {
        "clean_final": clean.final.detach().to(device="cpu", dtype=dtype),
        "transformed_final": transformed.final.detach().to(device="cpu", dtype=dtype),
        "clean_intermediate": (
            clean.intermediate.detach().to(device="cpu", dtype=dtype)
            if clean.intermediate is not None
            else empty
        ),
        "transformed_intermediate": (
            transformed.intermediate.detach().to(device="cpu", dtype=dtype)
            if transformed.intermediate is not None
            else empty.clone()
        ),
        "label": batch["label"].detach().float().cpu(),
        **{field: [str(value) for value in batch[field]] for field in _TEXT_FIELDS},
    }


def _merge_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        **{field: torch.cat([batch[field] for batch in batches], dim=0) for field in _TENSOR_FIELDS},
        **{field: sum((batch[field] for batch in batches), []) for field in _TEXT_FIELDS},
    }


def _save_shard(
    directory: Path,
    manifest: dict[str, Any],
    split: str,
    variant: int,
    start: int,
    features: dict[str, Any],
) -> None:
    count = int(features["label"].shape[0])
    end = start + count
    relative = Path(split) / f"variant-{variant:02d}" / f"shard-{start:06d}-{end:06d}.pt"
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": manifest["cache_key"],
        "split": split,
        "variant": variant,
        "start": start,
        "end": end,
        **features,
    }
    destination = directory / relative
    atomic_torch_save(payload, destination)
    split_state = manifest["splits"].setdefault(
        split, {"count": 0, "variants": {}}
    )
    entries = split_state["variants"].setdefault(str(variant), [])
    entries.append(
        {
            "start": start,
            "end": end,
            "path": str(relative),
            "size": destination.stat().st_size,
        }
    )
    split_state["count"] = max(int(split_state["count"]), end)
    _atomic_json(manifest, directory / "cache_manifest.json")


@torch.inference_mode()
def _cache_split_variant(
    config: AppConfig,
    model: FrozenClipDetector,
    device: torch.device,
    directory: Path,
    manifest: dict[str, Any],
    split: str,
    variant: int,
) -> None:
    dataset = AIGCManifestDataset(config, split, deterministic_variant=variant)
    split_state = manifest["splits"].setdefault(
        split, {"count": len(dataset), "variants": {}}
    )
    if int(split_state["count"]) not in {0, len(dataset)}:
        raise RuntimeError(f"Cached {split} count does not match the active manifest.")
    split_state["count"] = len(dataset)
    entries = split_state["variants"].setdefault(str(variant), [])
    prefix = _existing_prefix(directory, entries)
    if prefix > len(dataset):
        raise RuntimeError("Feature cache contains more records than the active split.")
    if prefix == len(dataset):
        LOGGER.info("Feature cache already complete for %s variant %d.", split, variant)
        return

    subset = Subset(dataset, range(prefix, len(dataset)))
    loader = _loader(config, subset)
    target_dtype = (
        torch.float16 if config.feature_cache.dtype == "float16" else torch.float32
    )
    pending: list[dict[str, Any]] = []
    pending_count = 0
    start = prefix
    description = f"Cache {split} v{variant}"
    for batch in tqdm(loader, desc=description):
        clean_views = batch["clean_views"].to(device, non_blocking=True)
        transformed_views = batch["transformed_views"].to(device, non_blocking=True)
        with _autocast(config, device):
            clean, transformed = model.encode_pair(clean_views, transformed_views)
        encoded = _encoded_payload(clean, transformed, batch, target_dtype)
        pending.append(encoded)
        pending_count += int(encoded["label"].shape[0])
        if pending_count >= config.feature_cache.shard_size:
            merged = _merge_batches(pending)
            _save_shard(directory, manifest, split, variant, start, merged)
            start += pending_count
            pending = []
            pending_count = 0
    if pending:
        merged = _merge_batches(pending)
        _save_shard(directory, manifest, split, variant, start, merged)


def precompute_features(
    config: AppConfig, model: FrozenClipDetector | None = None
) -> Path:
    """Create or resume all configured train and deterministic validation shards."""
    validate_preparation(config)
    seed_everything(config.project.seed)
    device = _resolve_device(config)
    directory = cache_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(config, directory)
    detector = (model if model is not None else create_detector(config.model)).to(device).eval()
    requested = {
        "train": range(config.feature_cache.train_variants),
        "val_id": range(1),
        "val_dg": range(1),
    }
    for split, variants in requested.items():
        for variant in variants:
            _cache_split_variant(
                config, detector, device, directory, manifest, split, variant
            )
    manifest["complete"] = True
    manifest["requested_train_variants"] = config.feature_cache.train_variants
    _atomic_json(manifest, directory / "cache_manifest.json")
    LOGGER.info("Feature cache ready at %s", directory)
    return directory


class CachedFeatureDataset(Dataset[dict[str, Any]]):
    """Load all validated feature shards and sample train variants in memory."""

    def __init__(self, config: AppConfig, split: str) -> None:
        if split not in {"train", "val_id", "val_dg"}:
            raise ValueError("Cached split must be train, val_id, or val_dg.")
        directory = cache_directory(config)
        manifest_path = directory / "cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Feature cache is missing; run aigc-cache-features first: {directory}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete") or manifest.get("cache_key") != cache_key(config):
            raise RuntimeError("Feature cache is incomplete or incompatible.")
        split_state = manifest.get("splits", {}).get(split)
        if not isinstance(split_state, dict):
            raise RuntimeError(f"Feature cache does not contain split {split}.")
        variant_count = config.feature_cache.train_variants if split == "train" else 1
        variants: list[dict[str, torch.Tensor]] = []
        reference_text: dict[str, list[str]] | None = None
        for variant in range(variant_count):
            entries = split_state.get("variants", {}).get(str(variant), [])
            if _existing_prefix(directory, entries) != int(split_state["count"]):
                raise RuntimeError(f"Feature cache is incomplete for {split} variant {variant}.")
            shards = [
                torch.load(directory / entry["path"], map_location="cpu", weights_only=False)
                for entry in entries
            ]
            if any(shard.get("cache_key") != manifest["cache_key"] for shard in shards):
                raise RuntimeError("Feature-cache shard identity mismatch.")
            tensors = {
                field: torch.cat([shard[field] for shard in shards], dim=0)
                for field in _TENSOR_FIELDS
            }
            if config.model.residual_statistics_enabled:
                residual_shards = []
                for entry, shard in zip(entries, shards):
                    sidecar_path = _residual_sidecar_path(directory, entry)
                    if not sidecar_path.is_file():
                        raise FileNotFoundError(
                            "Residual-statistics cache is missing; run "
                            f"aigc-cache-residuals first: {sidecar_path}"
                        )
                    sidecar = torch.load(
                        sidecar_path, map_location="cpu", weights_only=False
                    )
                    if (
                        sidecar.get("cache_key") != manifest["cache_key"]
                        or sidecar.get("extractor_version")
                        != RESIDUAL_STATISTICS_VERSION
                        or sidecar.get("start") != shard.get("start")
                        or sidecar.get("end") != shard.get("end")
                        or sidecar.get("id") != shard.get("id")
                    ):
                        raise RuntimeError(
                            f"Residual-statistics sidecar is incompatible: {sidecar_path}"
                        )
                    residual_shards.append(sidecar)
                tensors.update(
                    {
                        field: torch.cat(
                            [sidecar[field] for sidecar in residual_shards], dim=0
                        )
                        for field in (
                            "clean_residual_statistics",
                            "transformed_residual_statistics",
                        )
                    }
                )
            text = {
                field: sum((shard[field] for shard in shards), [])
                for field in _TEXT_FIELDS
            }
            if reference_text is None:
                reference_text = text
            elif text["id"] != reference_text["id"]:
                raise RuntimeError("Feature-cache variants do not have identical record order.")
            if variants and not torch.equal(tensors["label"], variants[0]["label"]):
                raise RuntimeError("Feature-cache variants do not have identical labels.")
            variants.append(tensors)
        if reference_text is None:
            raise RuntimeError(f"Feature cache contains no records for {split}.")
        self.split = split
        self.text = reference_text
        self.tensors = {
            field: torch.stack([variant[field] for variant in variants], dim=0)
            for field in variants[0]
        }

    def __len__(self) -> int:
        return self.tensors["label"].shape[1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        variant = (
            int(torch.randint(self.tensors["label"].shape[0], ()).item())
            if self.split == "train"
            else 0
        )
        clean_intermediate = self.tensors["clean_intermediate"][variant, index]
        transformed_intermediate = self.tensors["transformed_intermediate"][variant, index]
        return {
            "clean_final": self.tensors["clean_final"][variant, index],
            "transformed_final": self.tensors["transformed_final"][variant, index],
            "clean_intermediate": clean_intermediate,
            "transformed_intermediate": transformed_intermediate,
            **(
                {
                    "clean_residual_statistics": self.tensors[
                        "clean_residual_statistics"
                    ][variant, index],
                    "transformed_residual_statistics": self.tensors[
                        "transformed_residual_statistics"
                    ][variant, index],
                }
                if "clean_residual_statistics" in self.tensors
                else {}
            ),
            "label": self.tensors["label"][variant, index],
            **{field: self.text[field][index] for field in _TEXT_FIELDS},
        }


def _residual_sidecar_path(directory: Path, entry: dict[str, Any]) -> Path:
    relative = Path(str(entry["path"]))
    return directory / relative.with_suffix(".residual.pt")


@torch.inference_mode()
def precompute_residual_statistics(config: AppConfig) -> Path:
    """Create resumable residual-statistics sidecars without recomputing CLIP."""
    validate_preparation(config)
    seed_everything(config.project.seed)
    directory = cache_directory(config)
    manifest_path = directory / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Feature cache is missing; run aigc-cache-features first: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("cache_key") != cache_key(config):
        raise RuntimeError("Feature cache is incomplete or incompatible.")
    device = _resolve_device(config)
    extractor = ResidualStatisticsExtractor().to(device).eval()
    requested = {
        "train": range(config.feature_cache.train_variants),
        "val_id": range(1),
        "val_dg": range(1),
    }
    for split, variants in requested.items():
        split_state = manifest.get("splits", {}).get(split)
        if not isinstance(split_state, dict):
            raise RuntimeError(f"Feature cache does not contain split {split}.")
        for variant in variants:
            dataset = AIGCManifestDataset(
                config, split, deterministic_variant=variant
            )
            entries = split_state.get("variants", {}).get(str(variant), [])
            if _existing_prefix(directory, entries) != len(dataset):
                raise RuntimeError(
                    f"Feature cache is incomplete for {split} variant {variant}."
                )
            progress = tqdm(entries, desc=f"Cache residuals {split} v{variant}")
            for entry in progress:
                destination = _residual_sidecar_path(directory, entry)
                if destination.is_file():
                    existing = torch.load(
                        destination, map_location="cpu", weights_only=False
                    )
                    if (
                        existing.get("cache_key") == manifest["cache_key"]
                        and existing.get("extractor_version")
                        == RESIDUAL_STATISTICS_VERSION
                        and existing.get("start") == entry["start"]
                        and existing.get("end") == entry["end"]
                    ):
                        continue
                    raise RuntimeError(
                        f"Existing residual-statistics sidecar is incompatible: {destination}"
                    )
                subset = Subset(dataset, range(int(entry["start"]), int(entry["end"])))
                clean_parts: list[torch.Tensor] = []
                transformed_parts: list[torch.Tensor] = []
                identifiers: list[str] = []
                for batch in _loader(config, subset):
                    clean = batch["clean_views"].to(device, non_blocking=True)
                    transformed = batch["transformed_views"].to(
                        device, non_blocking=True
                    )
                    clean_parts.append(extractor(clean).cpu())
                    transformed_parts.append(extractor(transformed).cpu())
                    identifiers.extend(str(value) for value in batch["id"])
                payload = {
                    "schema_version": 1,
                    "extractor_version": RESIDUAL_STATISTICS_VERSION,
                    "cache_key": manifest["cache_key"],
                    "split": split,
                    "variant": variant,
                    "start": int(entry["start"]),
                    "end": int(entry["end"]),
                    "id": identifiers,
                    "clean_residual_statistics": torch.cat(clean_parts, dim=0),
                    "transformed_residual_statistics": torch.cat(
                        transformed_parts, dim=0
                    ),
                }
                base_shard = torch.load(
                    directory / str(entry["path"]),
                    map_location="cpu",
                    weights_only=False,
                )
                if identifiers != base_shard["id"]:
                    raise RuntimeError(
                        "Residual-statistics record order does not match the feature shard."
                    )
                atomic_torch_save(payload, destination)
    LOGGER.info("Residual-statistics cache ready at %s", directory)
    return directory


def main() -> None:
    """Precompute deterministic train variants and validation features."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(
        "Precompute resumable frozen-backbone features for cached head training."
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    destination = precompute_features(config)
    print(destination)


def main_residual_statistics() -> None:
    """Precompute residual statistics aligned with existing feature shards."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(
        "Precompute residual-statistics sidecars for cached head training."
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    destination = precompute_residual_statistics(config)
    print(destination)


if __name__ == "__main__":
    main()
