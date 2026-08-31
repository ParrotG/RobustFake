"""Shared manifest-backed evaluation for every isolated external dataset."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import random
import tempfile
from contextlib import nullcontext
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc_recognizer.checkpoint import load_inference_checkpoint
from aigc_recognizer.calibration import GlobalCalibrator, load_global_calibrator
from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.transforms import RobustPairTransform, canonical_rgb
from aigc_recognizer.metrics import binary_metrics
from aigc_recognizer.hub import resolve_inference_checkpoint
from aigc_recognizer.model import (
    RESIDUAL_STATISTICS_VERSION,
    EncodedViews,
    FrozenClipDetector,
    ResidualStatisticsExtractor,
    create_detector,
)
from aigc_recognizer.train import resolve_device
from aigc_recognizer.utils import atomic_torch_save, seed_everything, seed_worker

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationDatasetSpec:
    """Paths and expected identity for one prepared external dataset."""

    name: str
    repo_id: str
    revision: str
    output_dir: str
    manifest_path: str
    audit_path: str
    results_path: str
    predictions_path: str
    expected_real: int
    expected_fake: int


def dataset_spec(config: AppConfig, name: str) -> EvaluationDatasetSpec:
    """Resolve one configured dataset without changing the shared evaluator."""
    if name == "wildfake_official":
        source = config.official_evaluation
        return EvaluationDatasetSpec(
            name=name,
            repo_id=source.repo_id,
            revision=source.revision,
            output_dir=source.output_dir,
            manifest_path=source.manifest_path,
            audit_path=source.audit_path,
            results_path=source.results_path,
            predictions_path=source.predictions_path,
            expected_real=source.expected_real_count,
            expected_fake=source.expected_fake_count,
        )
    if name == "wildfake_broad":
        source = config.wildfake_evaluation
    elif name == "sid_set":
        source = config.sid_evaluation
    else:
        raise ValueError(f"Unsupported external evaluation dataset: {name}")
    return EvaluationDatasetSpec(
        name=name,
        repo_id=source.repo_id,
        revision=source.revision,
        output_dir=source.output_dir,
        manifest_path=source.manifest_path,
        audit_path=source.audit_path,
        results_path=source.results_path,
        predictions_path=source.predictions_path,
        expected_real=source.target_real,
        expected_fake=source.target_fake,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_COMPOSED_OPERATIONS: dict[str, list[str]] = {
    "combo_social_resize_0.5_jpeg_70": ["resize_0.5", "jpeg_70"],
    "combo_repost_jpeg_90_resize_0.5_jpeg_70": [
        "jpeg_90",
        "resize_0.5",
        "jpeg_70",
    ],
    "combo_crop_0.80_resize_0.5_jpeg_70": [
        "center_crop_0.80",
        "resize_0.5",
        "jpeg_70",
    ],
    "combo_blur_1.0_resize_0.5_jpeg_50": ["blur_1.0", "resize_0.5", "jpeg_50"],
    "combo_edit_color_0.20_noise_0.02_jpeg_70": [
        "color_jitter_0.20",
        "noise_0.02",
        "jpeg_70",
    ],
    "combo_stress_crop_0.80_blur_1.0_resize_0.25_jpeg_30": [
        "center_crop_0.80",
        "blur_1.0",
        "resize_0.25",
        "jpeg_30",
    ],
}


def _single_operation(image: Image.Image, operation: str, rng: random.Random) -> Image.Image:
    """Apply one deterministic operation while preserving the canvas size."""
    if operation == "clean":
        return image
    if operation.startswith("jpeg_"):
        quality = int(operation.split("_", 1)[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if operation.startswith("blur_"):
        return image.filter(ImageFilter.GaussianBlur(float(operation.split("_", 1)[1])))
    if operation.startswith("resize_"):
        scale = float(operation.split("_", 1)[1])
        width, height = image.size
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BICUBIC,
        )
        return reduced.resize((width, height), resample=Image.Resampling.BICUBIC)
    if operation.startswith("noise_"):
        sigma = float(operation.split("_", 1)[1])
        array = np.asarray(image, dtype=np.float32) / 255.0
        generator = np.random.default_rng(rng.getrandbits(64))
        noisy = np.clip(array + generator.normal(0.0, sigma, array.shape), 0.0, 1.0)
        return Image.fromarray(np.round(noisy * 255.0).astype(np.uint8), mode="RGB")
    if operation == "color_jitter_0.20":
        operations = [
            (ImageEnhance.Brightness, rng.uniform(0.8, 1.2)),
            (ImageEnhance.Contrast, rng.uniform(0.8, 1.2)),
            (ImageEnhance.Color, rng.uniform(0.8, 1.2)),
        ]
        rng.shuffle(operations)
        for enhancer, factor in operations:
            image = enhancer(image).enhance(factor)
        return image
    if operation == "center_crop_0.80":
        width, height = image.size
        crop_width, crop_height = max(1, round(width * 0.8)), max(1, round(height * 0.8))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)
    raise ValueError(f"Unsupported external evaluation operation: {operation}")


def _scenario_image(image: Image.Image, scenario: str, rng: random.Random) -> Image.Image:
    """Apply one single or ordered composed evaluation scenario."""
    operations = _COMPOSED_OPERATIONS.get(scenario, [scenario])
    for operation in operations:
        image = _single_operation(image, operation, rng)
    return image


def _balanced_stable_sample(
    records: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic label-balanced diagnostic subset."""
    if maximum is None or maximum >= len(records):
        return records
    grouped = {
        label: [record for record in records if int(record["label"]) == label]
        for label in (0, 1)
    }
    if not all(grouped.values()):
        raise RuntimeError("Fast external evaluation requires both labels.")
    targets = {0: maximum // 2, 1: maximum - maximum // 2}
    if any(targets[label] > len(grouped[label]) for label in (0, 1)):
        raise ValueError("Fast sample count exceeds the available size of one label.")

    def rank(record: dict[str, Any]) -> str:
        return hashlib.sha256(
            f"{seed}:external-fast:{record['id']}".encode("utf-8")
        ).hexdigest()

    selected = [
        record
        for label in (0, 1)
        for record in sorted(grouped[label], key=rank)[: targets[label]]
    ]
    return sorted(selected, key=lambda record: str(record["id"]))


class ExternalEvaluationDataset(Dataset[dict[str, Any]]):
    """Load a common external manifest under one deterministic scenario."""

    def __init__(
        self,
        config: AppConfig,
        spec: EvaluationDatasetSpec,
        scenario: str,
        *,
        max_samples: int | None = None,
    ) -> None:
        self.config = config
        self.spec = spec
        self.scenario = scenario
        self.root = Path(spec.output_dir)
        manifest_path = Path(spec.manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest_path}")
        records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.records = _balanced_stable_sample(records, max_samples, config.project.seed)
        self.transform = RobustPairTransform(config)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root / record["path"]
        try:
            with Image.open(image_path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to decode external evaluation image: {image_path}") from exc
        seed_digest = hashlib.sha256(
            f"{self.config.project.seed}:{record['id']}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(seed_digest[:16], 16))
        image = canonical_rgb(image, self.config.views.padding_color)
        image = self.transform.standardize(image, rng)
        global_geometry, local_geometry = self.transform._geometries(image, rng)
        transformed = _scenario_image(image.copy(), self.scenario, rng)
        views = torch.stack(
            [
                self.transform._tensor(self.transform._render(transformed, geometry))
                for geometry in (global_geometry, local_geometry)
            ]
        )
        return {
            "views": views,
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "id": str(record["id"]),
            "path": str(record["path"]),
            "source_name": str(record.get("source_name", "unknown")),
        }


def _loader(
    config: AppConfig,
    spec: EvaluationDatasetSpec,
    scenario: str,
    *,
    max_samples: int | None = None,
) -> DataLoader[Any]:
    evaluation = config.evaluation
    arguments: dict[str, Any] = {
        "dataset": ExternalEvaluationDataset(
            config, spec, scenario, max_samples=max_samples
        ),
        "batch_size": evaluation.batch_size,
        "shuffle": False,
        "num_workers": evaluation.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if evaluation.num_workers > 0:
        arguments["prefetch_factor"] = evaluation.prefetch_factor
        arguments["persistent_workers"] = False
    return DataLoader(**arguments)


def _external_cache_identity(
    config: AppConfig,
    spec: EvaluationDatasetSpec,
    scenario: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe every input that can change frozen external features."""
    return {
        "schema_version": 1,
        "dataset": spec.name,
        "manifest_sha256": _file_sha256(Path(spec.manifest_path)),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(str(record["id"]) for record in records).encode("utf-8")
        ).hexdigest(),
        "scenario": scenario,
        "seed": config.project.seed,
        "feature_model": {
            "backbone_name": config.model.backbone_name,
            "pretrained": config.model.pretrained,
            "embedding_dim": config.model.embedding_dim,
            "intermediate_layers": config.model.intermediate_layers,
            "intermediate_dim": config.model.intermediate_dim,
            "residual_statistics_version": RESIDUAL_STATISTICS_VERSION,
        },
        "views": dataclasses.asdict(config.views),
        "standardization": dataclasses.asdict(config.standardization),
        "dtype": config.evaluation.external_feature_cache_dtype,
    }


def _external_cache_path(
    config: AppConfig, spec: EvaluationDatasetSpec, identity: dict[str, Any]
) -> Path:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return Path(config.evaluation.external_feature_cache_dir) / spec.name / f"{key}.pt"


def _compatible_external_identity(existing: dict[str, Any], current: dict[str, Any]) -> bool:
    """Accept older head-coupled cache identities when frozen features are identical."""
    existing_core = {
        key: value for key, value in existing.items() if key not in {"model", "feature_model"}
    }
    current_core = {
        key: value for key, value in current.items() if key not in {"model", "feature_model"}
    }
    if existing_core != current_core:
        return False
    existing_model = existing.get("feature_model") or existing.get("model")
    if not isinstance(existing_model, dict):
        return False
    return all(
        existing_model.get(key) == value
        for key, value in current["feature_model"].items()
        if key != "residual_statistics_version"
    )


@torch.inference_mode()
def _load_or_create_external_features(
    model: FrozenClipDetector,
    config: AppConfig,
    device: torch.device,
    spec: EvaluationDatasetSpec,
    scenario: str,
    *,
    max_samples: int | None,
) -> dict[str, Any]:
    """Load or atomically cache checkpoint-independent frozen scenario features."""
    loader = _loader(config, spec, scenario, max_samples=max_samples)
    dataset = loader.dataset
    if not isinstance(dataset, ExternalEvaluationDataset):
        raise TypeError("External feature cache received an unexpected dataset type.")
    identity = _external_cache_identity(config, spec, scenario, dataset.records)
    path = _external_cache_path(config, spec, identity)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("identity") != identity:
            raise RuntimeError(f"External feature cache identity mismatch: {path}")
        LOGGER.info("Using external feature cache: %s", path)
        return payload
    cache_directory = path.parent
    if cache_directory.is_dir():
        for candidate in sorted(cache_directory.glob("*.pt")):
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            existing_identity = payload.get("identity")
            if (
                isinstance(existing_identity, dict)
                and _compatible_external_identity(existing_identity, identity)
                and payload.get("intermediate") is not None
                and payload.get("residual_statistics") is not None
            ):
                LOGGER.info("Using compatible external feature cache: %s", candidate)
                return payload

    dtype = (
        torch.float16
        if config.evaluation.external_feature_cache_dtype == "float16"
        else torch.float32
    )
    final: list[torch.Tensor] = []
    intermediate: list[torch.Tensor] = []
    residual: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[str] = []
    paths: list[str] = []
    source_names: list[str] = []
    residual_extractor = (
        None
        if model.residual_extractor is not None
        else ResidualStatisticsExtractor().to(device).eval()
    )
    for batch in tqdm(loader, desc=f"Cache {spec.name}/{scenario}"):
        views = batch["views"].to(device, non_blocking=True)
        with _autocast(config, device):
            encoded = model.encode_views(views)
        final.append(encoded.final.to(dtype=dtype, device="cpu"))
        if encoded.intermediate is not None:
            intermediate.append(encoded.intermediate.to(dtype=dtype, device="cpu"))
        residual_statistics = encoded.residual_statistics
        if residual_statistics is None:
            if residual_extractor is None:
                raise RuntimeError("Residual-statistics extractor is unavailable.")
            residual_statistics = residual_extractor(views)
        residual.append(residual_statistics.to(dtype=dtype, device="cpu"))
        labels.append(batch["label"].float().cpu())
        ids.extend(str(value) for value in batch["id"])
        paths.extend(str(value) for value in batch["path"])
        source_names.extend(str(value) for value in batch["source_name"])
    payload = {
        "identity": identity,
        "final": torch.cat(final),
        "intermediate": torch.cat(intermediate) if intermediate else None,
        "residual_statistics": torch.cat(residual),
        "label": torch.cat(labels),
        "id": ids,
        "path": paths,
        "source_name": source_names,
    }
    atomic_torch_save(payload, path)
    LOGGER.info("Saved external feature cache: %s", path)
    return payload


def _autocast(config: AppConfig, device: torch.device) -> Any:
    if not config.training.amp or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _extended_metrics(
    labels: list[float], probabilities: list[float], threshold: float
) -> dict[str, float | int]:
    result: dict[str, float | int] = binary_metrics(labels, probabilities, threshold)
    targets = np.asarray(labels, dtype=np.int64)
    predictions = (np.asarray(probabilities) >= threshold).astype(np.int64)
    true_positive = int(np.sum((targets == 1) & (predictions == 1)))
    true_negative = int(np.sum((targets == 0) & (predictions == 0)))
    false_positive = int(np.sum((targets == 0) & (predictions == 1)))
    false_negative = int(np.sum((targets == 1) & (predictions == 0)))
    result.update(
        {
            "count": int(targets.size),
            "accuracy": float(np.mean(targets == predictions)),
            "fake_recall": true_positive / max(1, true_positive + false_negative),
            "real_recall": true_negative / max(1, true_negative + false_positive),
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    )
    return result


def _source_group_metrics(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Report threshold behavior for source groups that usually contain one class."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        grouped.setdefault(str(item["source_name"]), []).append(item)
    result: dict[str, Any] = {}
    for name, items in sorted(grouped.items()):
        labels = np.asarray([int(item["label"]) for item in items], dtype=np.int64)
        scores = np.asarray([float(item["pred"]) for item in items], dtype=np.float64)
        predicted = (scores >= threshold).astype(np.int64)
        result[name] = {
            "count": len(items),
            "real_count": int(np.sum(labels == 0)),
            "fake_count": int(np.sum(labels == 1)),
            "mean_probability": float(np.mean(scores)),
            "predicted_fake_rate": float(np.mean(predicted)),
            "accuracy": float(np.mean(predicted == labels)),
        }
    return result


@torch.no_grad()
def _evaluate_scenario(
    model: FrozenClipDetector,
    config: AppConfig,
    device: torch.device,
    spec: EvaluationDatasetSpec,
    scenario: str,
    *,
    calibrator: GlobalCalibrator | None = None,
    max_samples: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels: list[float] = []
    probabilities: list[float] = []
    predictions: list[dict[str, Any]] = []
    if config.evaluation.external_feature_cache_enabled:
        cached = _load_or_create_external_features(
            model,
            config,
            device,
            spec,
            scenario,
            max_samples=max_samples,
        )
        batch_source: Any = range(0, len(cached["id"]), config.evaluation.batch_size)
    else:
        cached = None
        batch_source = _loader(config, spec, scenario, max_samples=max_samples)

    for item in tqdm(batch_source, desc=f"Evaluate {spec.name}/{scenario}"):
        if cached is None:
            batch = item
            views = batch["views"].to(device, non_blocking=True)
            with _autocast(config, device):
                output = model(views)
            batch_labels = batch["label"].float().tolist()
            batch_ids = batch["id"]
            batch_paths = batch["path"]
            batch_sources = batch["source_name"]
        else:
            start = int(item)
            end = min(start + config.evaluation.batch_size, len(cached["id"]))
            encoded = EncodedViews(
                final=cached["final"][start:end].to(device, non_blocking=True),
                intermediate=(
                    cached["intermediate"][start:end].to(device, non_blocking=True)
                    if cached["intermediate"] is not None
                    else None
                ),
                residual_statistics=(
                    cached["residual_statistics"][start:end].to(
                        device, non_blocking=True
                    )
                    if cached["residual_statistics"] is not None
                    else None
                ),
            )
            with _autocast(config, device):
                output = model.forward_encoded(encoded)
            batch_labels = cached["label"][start:end].float().tolist()
            batch_ids = cached["id"][start:end]
            batch_paths = cached["path"][start:end]
            batch_sources = cached["source_name"][start:end]
        raw_scores = torch.sigmoid(output.logits).float().cpu().tolist()
        calibrated_scores = (
            calibrator.probabilities(output.logits).float().cpu().tolist()
            if calibrator is not None
            else raw_scores
        )
        labels.extend(batch_labels)
        probabilities.extend(calibrated_scores)
        predictions.extend(
            {
                "id": record_id,
                "image_path": path,
                "label": int(label),
                "source_name": source_name,
                "scenario": scenario,
                "pred": float(score),
                **(
                    {"raw_pred": float(raw_score)}
                    if calibrator is not None
                    else {}
                ),
            }
            for record_id, path, label, source_name, score, raw_score in zip(
                batch_ids,
                batch_paths,
                batch_labels,
                batch_sources,
                calibrated_scores,
                raw_scores,
            )
        )
    threshold = calibrator.threshold if calibrator is not None else config.training.threshold
    metrics: dict[str, Any] = _extended_metrics(
        labels, probabilities, threshold
    )
    metrics["source_groups"] = _source_group_metrics(
        predictions, threshold
    )
    return metrics, predictions


def evaluate_external(
    config: AppConfig,
    name: str,
    *,
    fast: bool = False,
    checkpoint_path: str | Path | None = None,
    hf_repo_id: str | None = None,
    hf_revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared manifest without dataset-specific inference logic."""
    spec = dataset_spec(config, name)
    audit_path = Path(spec.audit_path)
    if not audit_path.is_file():
        raise FileNotFoundError("Evaluation audit is missing; run the matching preparation first.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("complete")):
        raise RuntimeError("External evaluation preparation is incomplete.")
    expected_counts = {"real": spec.expected_real, "fake": spec.expected_fake}
    if audit.get("counts") != expected_counts:
        raise RuntimeError("External evaluation audit count does not match the configuration.")

    checkpoint_path = resolve_inference_checkpoint(
        config,
        checkpoint_path,
        hf_repo_id=hf_repo_id,
        hf_revision=hf_revision,
    )
    device = resolve_device(config)
    config, checkpoint = load_inference_checkpoint(config, checkpoint_path)
    model = create_detector(config.model)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.to(device).eval()
    seed_everything(config.project.seed)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    calibrator = load_global_calibrator(
        config, checkpoint_path, checkpoint_sha256=checkpoint_sha256
    )

    scenarios = (
        list(config.evaluation.fast_scenarios)
        if fast
        else list(config.evaluation.scenarios)
    )
    if not fast and config.evaluation.enable_composed_scenarios:
        scenarios.extend(config.evaluation.composed_scenarios)
    max_samples = config.evaluation.fast_max_samples if fast else None
    scenario_results: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []
    for scenario in scenarios:
        metrics, predictions = _evaluate_scenario(
            model,
            config,
            device,
            spec,
            scenario,
            calibrator=calibrator,
            max_samples=max_samples,
        )
        scenario_results[scenario] = metrics
        if config.evaluation.save_predictions:
            all_predictions.extend(predictions)
    clean_auroc = float(scenario_results.get("clean", {}).get("auroc", math.nan))
    single_aurocs = [
        float(scenario_results[item]["auroc"])
        for item in scenarios
        if item != "clean" and item not in config.evaluation.composed_scenarios
    ]
    composed_aurocs = [
        float(scenario_results[item]["auroc"])
        for item in config.evaluation.composed_scenarios
        if item in scenario_results
    ]
    result = {
        "schema_version": 3,
        "mode": "fast" if fast else "full",
        "dataset": {
            "name": spec.name,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "counts": audit["counts"],
            "sampling": audit.get("sampling"),
            "archives": audit.get("archives"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        },
        "threshold": calibrator.threshold if calibrator else config.training.threshold,
        "calibration": (
            calibrator.to_metadata()
            if calibrator is not None
            else {"applied": False, "method": "none"}
        ),
        "evaluated_sample_count": int(
            next(iter(scenario_results.values())).get("count", 0)
        ),
        "scenarios": scenario_results,
        "summary": {
            "clean_auroc": clean_auroc,
            "mean_single_transform_auroc": (
                float(np.mean(single_aurocs)) if single_aurocs else math.nan
            ),
            "worst_single_transform_auroc": (
                float(np.min(single_aurocs)) if single_aurocs else math.nan
            ),
            "mean_transformed_auroc": (
                float(np.mean(single_aurocs)) if single_aurocs else math.nan
            ),
            "worst_transformed_auroc": (
                float(np.min(single_aurocs)) if single_aurocs else math.nan
            ),
            "mean_composed_transform_auroc": (
                float(np.mean(composed_aurocs)) if composed_aurocs else math.nan
            ),
            "worst_composed_transform_auroc": (
                float(np.min(composed_aurocs)) if composed_aurocs else math.nan
            ),
        },
    }
    results_path = Path(spec.results_path)
    predictions_path = Path(spec.predictions_path)
    if fast:
        results_path = results_path.with_name(f"{results_path.stem}.fast{results_path.suffix}")
        predictions_path = predictions_path.with_name(
            f"{predictions_path.stem}.fast{predictions_path.suffix}"
        )
    _atomic_write(
        results_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if config.evaluation.save_predictions:
        lines = "".join(json.dumps(item, sort_keys=True) + "\n" for item in all_predictions)
        _atomic_write(predictions_path, lines.encode("utf-8"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _main(name: str, description: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(description)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Evaluate a deterministic balanced subset on representative severe scenarios.",
    )
    model_source = parser.add_mutually_exclusive_group()
    model_source.add_argument("--checkpoint", default=None, help="Local checkpoint override.")
    model_source.add_argument(
        "--hf-repo",
        default=None,
        help="Download a checkpoint-bound model package from this Hugging Face repository.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Immutable Hugging Face revision or branch. Defaults to the configuration.",
    )
    args = parser.parse_args()
    result = evaluate_external(
        load_config(args.config, args.set),
        name,
        fast=args.fast,
        checkpoint_path=args.checkpoint,
        hf_repo_id=args.hf_repo,
        hf_revision=args.hf_revision,
    )
    LOGGER.info(
        "External evaluation completed: dataset=%s clean AUROC=%.6f.",
        name,
        result["summary"]["clean_auroc"],
    )


def main_wildfake() -> None:
    _main("wildfake_broad", "Evaluate a detector on the broad WildFake sample.")


def main_sid() -> None:
    _main("sid_set", "Evaluate a detector on the SID-Set real/full-synthetic sample.")
