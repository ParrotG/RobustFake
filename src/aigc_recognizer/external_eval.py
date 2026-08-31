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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.transforms import RobustPairTransform, canonical_rgb
from aigc_recognizer.metrics import binary_metrics
from aigc_recognizer.model import FrozenClipDetector, create_detector
from aigc_recognizer.train import resolve_device
from aigc_recognizer.utils import seed_everything, seed_worker

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


class ExternalEvaluationDataset(Dataset[dict[str, Any]]):
    """Load a common external manifest under one deterministic scenario."""

    def __init__(self, config: AppConfig, spec: EvaluationDatasetSpec, scenario: str) -> None:
        self.config = config
        self.spec = spec
        self.scenario = scenario
        self.root = Path(spec.output_dir)
        manifest_path = Path(spec.manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest_path}")
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
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


def _loader(config: AppConfig, spec: EvaluationDatasetSpec, scenario: str) -> DataLoader[Any]:
    evaluation = config.evaluation
    arguments: dict[str, Any] = {
        "dataset": ExternalEvaluationDataset(config, spec, scenario),
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels: list[float] = []
    probabilities: list[float] = []
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(_loader(config, spec, scenario), desc=f"Evaluate {spec.name}/{scenario}"):
        views = batch["views"].to(device, non_blocking=True)
        with _autocast(config, device):
            output = model(views)
        scores = torch.sigmoid(output.logits).float().cpu().tolist()
        batch_labels = batch["label"].float().tolist()
        labels.extend(batch_labels)
        probabilities.extend(scores)
        predictions.extend(
            {
                "id": record_id,
                "image_path": path,
                "label": int(label),
                "source_name": source_name,
                "scenario": scenario,
                "pred": float(score),
            }
            for record_id, path, label, source_name, score in zip(
                batch["id"], batch["path"], batch_labels, batch["source_name"], scores
            )
        )
    metrics: dict[str, Any] = _extended_metrics(
        labels, probabilities, config.training.threshold
    )
    metrics["source_groups"] = _source_group_metrics(
        predictions, config.training.threshold
    )
    return metrics, predictions


def evaluate_external(config: AppConfig, name: str) -> dict[str, Any]:
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

    checkpoint_path = Path(config.evaluation.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint_path}")
    device = resolve_device(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_backbone = {"name": config.model.backbone_name, "pretrained": config.model.pretrained}
    if checkpoint.get("backbone") != expected_backbone:
        raise RuntimeError("Checkpoint backbone does not match the evaluation configuration.")
    model = create_detector(config.model)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.to(device).eval()
    seed_everything(config.project.seed)

    scenarios = list(config.evaluation.scenarios)
    if config.evaluation.enable_composed_scenarios:
        scenarios.extend(config.evaluation.composed_scenarios)
    scenario_results: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []
    for scenario in scenarios:
        metrics, predictions = _evaluate_scenario(model, config, device, spec, scenario)
        scenario_results[scenario] = metrics
        if config.evaluation.save_predictions:
            all_predictions.extend(predictions)
    clean_auroc = float(scenario_results.get("clean", {}).get("auroc", math.nan))
    single_aurocs = [
        float(scenario_results[item]["auroc"])
        for item in config.evaluation.scenarios
        if item != "clean"
    ]
    composed_aurocs = [
        float(scenario_results[item]["auroc"])
        for item in config.evaluation.composed_scenarios
        if item in scenario_results
    ]
    result = {
        "schema_version": 2,
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
            "sha256": _file_sha256(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        },
        "threshold": config.training.threshold,
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
    _atomic_write(
        Path(spec.results_path),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if config.evaluation.save_predictions:
        lines = "".join(json.dumps(item, sort_keys=True) + "\n" for item in all_predictions)
        _atomic_write(Path(spec.predictions_path), lines.encode("utf-8"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _main(name: str, description: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(description)
    args = parser.parse_args()
    result = evaluate_external(load_config(args.config, args.set), name)
    LOGGER.info(
        "External evaluation completed: dataset=%s clean AUROC=%.6f.",
        name,
        result["summary"]["clean_auroc"],
    )


def main_wildfake() -> None:
    _main("wildfake_broad", "Evaluate a detector on the broad WildFake sample.")


def main_sid() -> None:
    _main("sid_set", "Evaluate a detector on the SID-Set real/full-synthetic sample.")
