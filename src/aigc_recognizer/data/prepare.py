"""Prepare a reproducible paired subset from a pinned Hugging Face dataset."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import signal
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageOps, UnidentifiedImageError

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config

LOGGER = logging.getLogger(__name__)
GIB = 1024**3
SPLITS = ("train", "val", "test")


class PreparationError(RuntimeError):
    """Raised when safe paired acquisition cannot be completed."""


@dataclass(frozen=True)
class DecodedImage:
    """Safe decoded-image properties shared with official data preparation."""

    width: int
    height: int
    image_format: str
    perceptual_hash: str


def validate_and_describe_image(image_bytes: bytes, config: AppConfig) -> DecodedImage:
    """Validate arbitrary evaluation image bytes without imposing training dimensions."""
    description = _describe_image(image_bytes, config)
    return DecodedImage(
        width=int(description["width"]),
        height=int(description["height"]),
        image_format=str(description["format"]),
        perceptual_hash=str(description["perceptual_hash"]),
    )


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically replace one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _normalize_split(value: object) -> str:
    split = str(value).casefold()
    if split == "validation":
        return "val"
    if split not in SPLITS:
        raise PreparationError(f"Unsupported source split: {value}")
    return split


def _resolve_hf_token(config: AppConfig) -> str | bool | None:
    from huggingface_hub import get_token

    if config.data.hf_auth == "disabled":
        return False
    token = get_token()
    if token:
        LOGGER.info("Using the saved Hugging Face credential for gated dataset requests.")
        return token
    if config.data.hf_auth == "required":
        raise PreparationError(
            "This gated dataset requires a saved Hugging Face token. Accept the dataset "
            "terms in a browser, then run 'hf auth login' in this user environment."
        )
    LOGGER.warning("No saved Hugging Face token was found; requests are unauthenticated.")
    return None


def _is_retryable(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError, OSError)):
            return True
        if type(current).__module__.split(".", 1)[0] in {
            "httpcore",
            "httpx",
            "huggingface_hub",
            "requests",
            "urllib3",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry(config: AppConfig, operation: Any, description: str) -> Any:
    for attempt in range(config.data.network_max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= config.data.network_max_retries:
                raise
            delay = min(60.0, config.data.network_retry_base_seconds * (2**attempt))
            LOGGER.warning(
                "%s failed with %s; retrying in %.1f seconds (%d/%d).",
                description,
                type(exc).__name__,
                delay,
                attempt + 1,
                config.data.network_max_retries,
            )
            time.sleep(delay)
    raise AssertionError("Retry loop terminated unexpectedly.")


def _verify_access(config: AppConfig, token: str | bool | None) -> list[Any]:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = _retry(
            config,
            lambda: HfApi(token=token).dataset_info(
                config.data.repo_id,
                revision=config.data.revision,
                files_metadata=True,
            ),
            "Dataset metadata request",
        )
    except HfHubHTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status in {401, 403}:
            raise PreparationError(
                "Hugging Face denied access to the gated dataset. Accept its terms and run "
                "'hf auth login' before retrying."
            ) from exc
        raise
    if str(info.sha) != config.data.revision:
        raise PreparationError(
            f"Dataset revision resolved to {info.sha}, expected {config.data.revision}."
        )
    return list(info.siblings)


def _download_file(
    config: AppConfig, token: str | bool | None, filename: str, local_dir: str
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        _retry(
            config,
            lambda: hf_hub_download(
                repo_id=config.data.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=config.data.revision,
                token=token,
                local_dir=local_dir,
            ),
            f"Download of {filename}",
        )
    )


def _load_source_metadata(
    config: AppConfig, token: str | bool | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as parquet

    metadata_dir = str(Path(config.data.shard_cache_dir) / "metadata")
    manifest_path = _download_file(config, token, config.data.metadata_file, metadata_dir)
    source_config_path = _download_file(
        config, token, config.data.source_config_file, metadata_dir
    )
    rows = parquet.read_table(manifest_path).to_pylist()
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if str(source_config.get("pipeline_version")) != config.data.expected_pipeline_version:
        raise PreparationError("The source preprocessing pipeline version does not match config.")
    if int(source_config.get("target_resolution", -1)) != config.data.expected_image_size:
        raise PreparationError("The source target resolution does not match config.")
    if set(source_config.get("generators", {})) != set(config.data.generators):
        raise PreparationError("The source generator set does not match data.generators.")
    return rows, source_config


def select_paired_metadata(
    rows: Iterable[Mapping[str, Any]], config: AppConfig
) -> dict[str, dict[str, Any]]:
    """Validate all seven-way parent groups and choose one balanced fake per parent."""
    by_parent: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source_row in rows:
        row = dict(source_row)
        parent = str(row.get("source_real_id") or "")
        generator = str(row.get("generator") or "")
        if not parent or not generator or generator in by_parent[parent]:
            raise PreparationError("Source metadata contains an invalid or duplicate pair row.")
        row["split"] = _normalize_split(row.get("split"))
        by_parent[parent][generator] = row
    if len(by_parent) != config.data.expected_parent_count:
        raise PreparationError(
            f"Source metadata contains {len(by_parent)} parents, expected "
            f"{config.data.expected_parent_count}."
        )
    expected_generators = {"real", *config.data.generators}
    for parent, group in by_parent.items():
        if set(group) != expected_generators:
            raise PreparationError(f"Parent {parent} does not have one real and six fakes.")
        if int(group["real"]["label"]) != 0 or any(
            int(group[name]["label"]) != 1 for name in config.data.generators
        ):
            raise PreparationError(f"Parent {parent} has inconsistent labels.")
        if str(group["real"]["image_id"]) != parent:
            raise PreparationError(f"Parent {parent} has an invalid real image identity.")
        if len({row["split"] for row in group.values()}) != 1:
            raise PreparationError(f"Parent {parent} crosses source splits.")

    selected: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        parents = sorted(
            parent for parent, group in by_parent.items() if group["real"]["split"] == split
        )
        rng = random.Random(f"{config.project.seed}:{split}:generator-assignment")
        rng.shuffle(parents)
        assignments: list[str] = []
        for start in range(0, len(parents), len(config.data.generators)):
            generator_block = list(config.data.generators)
            rng.shuffle(generator_block)
            assignments.extend(generator_block[: len(parents) - start])
        for parent, generator in zip(parents, assignments):
            real = by_parent[parent]["real"]
            fake = by_parent[parent][generator]
            selected[str(real["image_id"])] = real
            selected[str(fake["image_id"])] = fake
    expected_selected = config.data.expected_parent_count * 2
    if len(selected) != expected_selected:
        raise PreparationError(
            f"Paired selection produced {len(selected)} rows, expected {expected_selected}."
        )
    return selected


def _source_shards(siblings: Iterable[Any], config: AppConfig) -> list[str]:
    prefixes = [
        f"{generator}/{split}/"
        for generator in ("real", *config.data.generators)
        for split in SPLITS
    ]
    files: list[tuple[str, int]] = []
    for sibling in siblings:
        name = str(sibling.rfilename)
        if name.endswith(".parquet") and any(name.startswith(prefix) for prefix in prefixes):
            files.append((name, int(sibling.size or 0)))
    if not files:
        raise PreparationError("No image Parquet shards were found at the pinned revision.")
    total = sum(size for _, size in files)
    if total > config.data.max_download_gb * GIB:
        raise PreparationError(
            f"Pinned image shards total {total / GIB:.2f} GiB, exceeding "
            f"data.max_download_gb={config.data.max_download_gb:.2f}."
        )
    peak_prefetch = sum(
        sorted((size for _, size in files), reverse=True)[: config.data.download_workers]
    )
    if peak_prefetch > config.data.max_shard_cache_gb * GIB:
        raise PreparationError(
            f"The largest prefetched shards may require {peak_prefetch / GIB:.2f} GiB, "
            f"exceeding data.max_shard_cache_gb={config.data.max_shard_cache_gb:.2f}."
        )
    LOGGER.info("Preflight found %d image shards totaling %.2f GiB.", len(files), total / GIB)
    return sorted(name for name, _ in files)


def _config_fingerprint(config: AppConfig, selected_ids: Iterable[str]) -> str:
    payload = {
        "repo_id": config.data.repo_id,
        "revision": config.data.revision,
        "generators": config.data.generators,
        "seed": config.project.seed,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(sorted(selected_ids)).encode()
        ).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, Mapping):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes):
        raise PreparationError("A selected source row does not contain image bytes.")
    return value


def _describe_image(image_bytes: bytes, config: AppConfig) -> dict[str, Any]:
    import imagehash

    Image.MAX_IMAGE_PIXELS = config.data.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                image_format = str(source.format or "").casefold()
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                return {
                    "format": image_format,
                    "width": width,
                    "height": height,
                    "perceptual_hash": str(
                        imagehash.phash(image, hash_size=config.data.perceptual_hash_size)
                    ),
                    "difference_hash": str(
                        imagehash.dhash(image, hash_size=config.data.perceptual_hash_size)
                    ),
                }
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PreparationError("A selected image exceeds the safe pixel limit.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreparationError("A selected image cannot be decoded safely.") from exc


def _manifest_records(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        record_id = str(record["id"])
        if record_id in records:
            raise PreparationError(f"Existing manifest contains duplicate id {record_id}.")
        if not (root / record["path"]).is_file():
            raise PreparationError(f"Resume image is missing for {record_id}.")
        records[record_id] = record
    return records


def _write_manifest(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    split_order = {name: index for index, name in enumerate(SPLITS)}
    ordered = sorted(
        records,
        key=lambda row: (split_order[str(row["split"])], int(row["label"]), str(row["id"])),
    )
    content = "".join(json.dumps(dict(record), sort_keys=True) + "\n" for record in ordered)
    atomic_write_text(path, content)


def _state_audit(
    config: AppConfig,
    fingerprint: str,
    completed: set[str],
    shards: list[str],
    records: Mapping[str, Any],
    complete: bool,
    reason: str,
    excluded_parents: Iterable[str] = (),
) -> dict[str, Any]:
    counts = Counter(f"{r['split']}:{r['label']}" for r in records.values())
    generators = Counter(
        f"{r['split']}:{r['generator']}" for r in records.values() if r["label"] == 1
    )
    return {
        "complete": complete,
        "source_mode": "pinned_resumable_parquet_shards",
        "stop_reason": reason,
        "source_revision": config.data.revision,
        "sampling_config_sha256": fingerprint,
        "processed_shards": len(completed),
        "total_shards": len(shards),
        "selected": len(records),
        "class_counts": dict(sorted(counts.items())),
        "fake_generator_counts": dict(sorted(generators.items())),
        "excluded_parent_count": len(set(excluded_parents)),
        "nuisance_report": config.data.nuisance_report_path,
    }


def _checkpoint(
    config: AppConfig,
    fingerprint: str,
    completed: set[str],
    shards: list[str],
    expected_ids: set[str],
    records: Mapping[str, dict[str, Any]],
    complete: bool,
    reason: str,
    excluded_parents: Iterable[str] = (),
) -> dict[str, Any]:
    _write_manifest(Path(config.data.manifest_path), records.values())
    state = {
        "schema_version": 1,
        "complete": complete,
        "sampling_config_sha256": fingerprint,
        "source_revision": config.data.revision,
        "completed_shards": sorted(completed),
        "expected_ids": sorted(expected_ids),
        "extracted_ids": sorted(records),
    }
    atomic_write_text(Path(config.data.state_path), json.dumps(state, indent=2) + "\n")
    audit = _state_audit(
        config, fingerprint, completed, shards, records, complete, reason, excluded_parents
    )
    atomic_write_text(
        Path(config.data.audit_path), json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def _extract_selected_rows(
    shard: Path,
    selected: Mapping[str, Mapping[str, Any]],
    config: AppConfig,
    records: dict[str, dict[str, Any]],
) -> None:
    import pyarrow.parquet as parquet

    root = Path(config.data.output_dir)
    for batch in parquet.ParquetFile(shard).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            image_id = str(row.get("image_id") or "")
            expected = selected.get(image_id)
            if expected is None or image_id in records:
                continue
            parent = str(row.get("source_real_id") or "")
            generator = str(row.get("generator") or "")
            label = int(row.get("label", -1))
            split = _normalize_split(row.get("split"))
            if (
                parent != str(expected["source_real_id"])
                or generator != str(expected["generator"])
                or label != int(expected["label"])
                or split != str(expected["split"])
            ):
                raise PreparationError(f"Source row metadata mismatch for {image_id}.")
            if str(row.get("pipeline_version")) != config.data.expected_pipeline_version:
                raise PreparationError(f"Unexpected pipeline version for {image_id}.")
            image_bytes = _image_bytes(row.get("image"))
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest != str(row.get("sha256")) or digest != str(expected.get("sha256")):
                raise PreparationError(f"SHA-256 mismatch for {image_id}.")
            description = _describe_image(image_bytes, config)
            if (
                description["format"] != "png"
                or description["width"] != config.data.expected_image_size
                or description["height"] != config.data.expected_image_size
                or int(row.get("width", -1)) != config.data.expected_image_size
                or int(row.get("height", -1)) != config.data.expected_image_size
            ):
                raise PreparationError(f"Unexpected image encoding or dimensions for {image_id}.")
            relative = Path("images") / split / ("fake" if label else "real") / f"{image_id}.png"
            destination = root / relative
            if not destination.is_file():
                _atomic_write_bytes(destination, image_bytes)
            records[image_id] = {
                "id": image_id,
                "path": str(relative),
                "label": label,
                "split": split,
                "source_split": str(row.get("split")),
                "source_real_id": parent,
                "generator": generator,
                "model_name": generator,
                "source_dataset": str(row.get("source_dataset") or "unknown"),
                "prompt": str(row.get("prompt") or ""),
                "pipeline_version": str(row.get("pipeline_version")),
                "format": "png",
                "width": description["width"],
                "height": description["height"],
                "bytes": len(image_bytes),
                "content_sha256": digest,
                "perceptual_hash": description["perceptual_hash"],
                "difference_hash": description["difference_hash"],
                "source_revision": config.data.revision,
            }


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _official_hashes(config: AppConfig) -> list[tuple[str, str, Any]]:
    import imagehash

    manifest = Path(config.data.official_leakage_manifest)
    root = Path(config.data.official_leakage_root)
    if not manifest.is_file():
        raise PreparationError(
            "COCO val2017 leakage cannot be audited because the official WildFake manifest is "
            "missing. Run aigc-prepare-official-eval first."
        )
    hashes: list[tuple[str, str, Any]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["label"]) != 0:
            continue
        path = root / record["path"]
        if not path.is_file():
            raise PreparationError(f"Official leakage audit image is missing: {path}")
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            hashes.append(
                (
                    str(imagehash.phash(image, hash_size=config.data.perceptual_hash_size)),
                    str(imagehash.dhash(image, hash_size=config.data.perceptual_hash_size)),
                    imagehash.crop_resistant_hash(image),
                )
            )
    if not hashes:
        raise PreparationError("The official leakage manifest contains no real images.")
    return hashes


def _excluded_parents(records: Mapping[str, Mapping[str, Any]], config: AppConfig) -> set[str]:
    excluded: set[str] = set()
    sha_parents: dict[str, str] = {}
    phash_parents: dict[str, str] = {}
    for record in records.values():
        parent = str(record["source_real_id"])
        digest = str(record["content_sha256"])
        phash = str(record["perceptual_hash"])
        if config.data.exact_deduplication and digest in sha_parents:
            excluded.update((parent, sha_parents[digest]))
        else:
            sha_parents[digest] = parent
        if config.data.perceptual_deduplication and phash in phash_parents:
            excluded.update((parent, phash_parents[phash]))
        else:
            phash_parents[phash] = parent

    official = _official_hashes(config)
    coco_reals = [
        record
        for record in records.values()
        if record["label"] == 0 and str(record["source_dataset"]).casefold() == "coco"
    ]
    for record in coco_reals:
        candidate_phash = str(record["perceptual_hash"])
        candidate_dhash = str(record["difference_hash"])
        candidate_crop_hash: Any | None = None
        for official_phash, official_dhash, official_crop_hash in official:
            phash_distance = _hamming(candidate_phash, official_phash)
            dhash_distance = _hamming(candidate_dhash, official_dhash)
            strict_match = (
                phash_distance <= config.data.leakage_phash_distance
                and dhash_distance <= config.data.leakage_dhash_distance
            )
            crop_match = False
            if not strict_match and phash_distance <= max(
                32, config.data.leakage_phash_distance * 4
            ):
                if candidate_crop_hash is None:
                    import imagehash

                    candidate_path = Path(config.data.output_dir) / str(record["path"])
                    with Image.open(candidate_path) as source:
                        candidate_crop_hash = imagehash.crop_resistant_hash(
                            ImageOps.exif_transpose(source).convert("RGB")
                        )
                crop_match = candidate_crop_hash.matches(
                    official_crop_hash,
                    hamming_cutoff=config.data.leakage_dhash_distance,
                )
            if strict_match or crop_match:
                excluded.add(str(record["source_real_id"]))
                break
    return excluded


def _safe_remove_cached_shard(path: Path, cache_root: Path) -> None:
    try:
        path.absolute().relative_to(cache_root.absolute())
    except ValueError as exc:
        raise PreparationError(f"Refusing to remove a shard outside project cache: {path}") from exc
    path.unlink(missing_ok=True)


def prepare_dataset(config: AppConfig) -> dict[str, Any]:
    """Acquire the pinned paired subset, audit leakage, and run nuisance probing."""
    token = _resolve_hf_token(config)
    siblings = _verify_access(config, token)
    metadata_rows, _source_config = _load_source_metadata(config, token)
    selected = select_paired_metadata(metadata_rows, config)
    expected_ids = set(selected)
    shards = _source_shards(siblings, config)
    fingerprint = _config_fingerprint(config, expected_ids)
    output_root = Path(config.data.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.data.manifest_path)
    state_path = Path(config.data.state_path)
    records = _manifest_records(manifest_path, output_root)
    completed: set[str] = set()
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("sampling_config_sha256") != fingerprint:
            raise PreparationError(
                "Existing preparation state uses a different source selection. Move the old "
                "output directory or restore the matching configuration before retrying."
            )
        completed = set(state.get("completed_shards", []))
        if set(state.get("expected_ids", [])) != expected_ids:
            raise PreparationError("Existing preparation state has a different expected ID set.")
        if bool(state.get("complete")):
            audit = json.loads(Path(config.data.audit_path).read_text(encoding="utf-8"))
            if config.nuisance_audit.enabled and not Path(config.data.nuisance_report_path).is_file():
                try:
                    from aigc_recognizer.data.nuisance import run_nuisance_audit

                    run_nuisance_audit(config)
                except Exception:
                    LOGGER.exception("Nuisance audit failed; prepared data remains usable.")
            return audit
    unknown_existing = set(records) - expected_ids
    if unknown_existing:
        raise PreparationError("Existing manifest contains records outside the selected ID set.")

    pending = [name for name in shards if name not in completed]
    cache_root = Path(config.data.shard_cache_dir) / "payload"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        with ThreadPoolExecutor(max_workers=config.data.download_workers) as executor:
            futures: dict[str, Future[Path]] = {}
            next_index = 0

            def fill_queue() -> None:
                nonlocal next_index
                while next_index < len(pending) and len(futures) < config.data.download_workers:
                    filename = pending[next_index]
                    futures[filename] = executor.submit(
                        _download_file, config, token, filename, str(cache_root)
                    )
                    next_index += 1

            fill_queue()
            processed_since_checkpoint = 0
            for position, filename in enumerate(pending, start=1):
                path = futures.pop(filename).result()
                LOGGER.info("Processing source shard %d/%d: %s", position, len(pending), filename)
                _extract_selected_rows(path, selected, config, records)
                completed.add(filename)
                _safe_remove_cached_shard(path, cache_root)
                processed_since_checkpoint += 1
                fill_queue()
                if processed_since_checkpoint >= config.data.checkpoint_every_shards:
                    _checkpoint(
                        config,
                        fingerprint,
                        completed,
                        shards,
                        expected_ids,
                        records,
                        False,
                        "periodic shard checkpoint",
                    )
                    processed_since_checkpoint = 0
    except BaseException as exc:
        _checkpoint(
            config,
            fingerprint,
            completed,
            shards,
            expected_ids,
            records,
            False,
            f"interrupted by {type(exc).__name__}",
        )
        raise

    missing = expected_ids - set(records)
    if missing:
        _checkpoint(
            config,
            fingerprint,
            completed,
            shards,
            expected_ids,
            records,
            False,
            f"source exhausted with {len(missing)} selected IDs missing",
        )
        raise PreparationError(f"Source shards did not contain {len(missing)} selected image IDs.")

    excluded = _excluded_parents(records, config)
    safe_records = {
        record_id: record
        for record_id, record in records.items()
        if str(record["source_real_id"]) not in excluded
    }
    pair_counts = Counter(str(record["source_real_id"]) for record in safe_records.values())
    pair_labels: dict[str, set[int]] = defaultdict(set)
    for record in safe_records.values():
        pair_labels[str(record["source_real_id"])].add(int(record["label"]))
    for parent, count in pair_counts.items():
        if count != 2 or pair_labels[parent] != {0, 1}:
            raise PreparationError(f"Final parent {parent} is not a complete real/fake pair.")
    audit = _checkpoint(
        config,
        fingerprint,
        completed,
        shards,
        expected_ids,
        safe_records,
        True,
        "all selected shards processed and leakage audit completed",
        excluded,
    )
    if config.nuisance_audit.enabled:
        try:
            from aigc_recognizer.data.nuisance import run_nuisance_audit

            run_nuisance_audit(config)
        except Exception:
            LOGGER.exception("Nuisance audit failed; prepared data remains usable.")
    return audit


def main() -> None:
    """Run the public paired dataset preparation command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Prepare the pinned paired AI image dataset subset.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    previous_handlers: dict[signal.Signals, Any] = {}

    def interrupt(signum: int, _frame: Any) -> None:
        raise PreparationError(f"Dataset preparation received {signal.Signals(signum).name}.")

    for signal_name in ("SIGHUP", "SIGTERM"):
        selected_signal = getattr(signal, signal_name, None)
        if selected_signal is not None:
            previous_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, interrupt)
    try:
        audit = prepare_dataset(config)
    except PreparationError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(2) from exc
    finally:
        for selected_signal, handler in previous_handlers.items():
            signal.signal(selected_signal, handler)
    LOGGER.info("Dataset preparation completed with %d safe images.", audit["selected"])


if __name__ == "__main__":
    main()
