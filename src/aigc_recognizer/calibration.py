"""Fit and apply one checkpoint-bound global affine probability calibrator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from aigc_recognizer.checkpoint import load_inference_checkpoint
from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.dataset import AIGCManifestDataset, validate_preparation
from aigc_recognizer.feature_cache import CachedFeatureDataset, cache_directory
from aigc_recognizer.metrics import binary_metrics
from aigc_recognizer.model import create_cached_detector, create_detector
from aigc_recognizer.train import _forward_pair, _move_batch, resolve_device
from aigc_recognizer.utils import seed_everything, seed_worker

LOGGER = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True)
class GlobalCalibrator:
    """Map a raw binary logit through a learned affine Platt transform."""

    coefficient: float
    intercept: float
    threshold: float
    path: str
    checkpoint_sha256: str

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits.float() * self.coefficient + self.intercept)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "applied": True,
            "method": "affine_platt",
            "path": self.path,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
            "threshold": self.threshold,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def calibration_path(config: AppConfig, checkpoint_path: str | Path) -> Path:
    """Resolve an explicit path or the calibration beside the checkpoint."""
    if config.evaluation.calibration_path:
        return Path(config.evaluation.calibration_path)
    return Path(checkpoint_path).with_name("calibration.json")


def load_global_calibrator(
    config: AppConfig,
    checkpoint_path: str | Path,
    *,
    checkpoint_sha256: str | None = None,
) -> GlobalCalibrator | None:
    """Load a compatible calibrator, treating an absent automatic path as disabled."""
    path = calibration_path(config, checkpoint_path)
    if not path.is_file():
        if config.evaluation.calibration_path:
            raise FileNotFoundError(f"Configured calibration file does not exist: {path}")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"coefficient", "intercept", "threshold", "checkpoint_sha256"}
    if payload.get("schema_version") != 1 or not required <= set(payload):
        raise RuntimeError(f"Calibration file has an unsupported schema: {path}")
    actual_sha256 = checkpoint_sha256 or _file_sha256(Path(checkpoint_path))
    if payload["checkpoint_sha256"] != actual_sha256:
        raise RuntimeError("Calibration file was fitted for a different checkpoint.")
    coefficient = float(payload["coefficient"])
    threshold = float(payload["threshold"])
    if coefficient <= 0 or not 0.0 <= threshold <= 1.0:
        raise RuntimeError("Calibration coefficient or threshold is invalid.")
    return GlobalCalibrator(
        coefficient=coefficient,
        intercept=float(payload["intercept"]),
        threshold=threshold,
        path=str(path),
        checkpoint_sha256=actual_sha256,
    )


def _validation_loaders(config: AppConfig) -> tuple[list[DataLoader[Any]], bool]:
    cache_manifest = cache_directory(config) / "cache_manifest.json"
    cached = cache_manifest.is_file()
    dataset_type = CachedFeatureDataset if cached else AIGCManifestDataset
    if not cached:
        LOGGER.info("Compatible feature cache was not found; calibration will encode images online.")
    loaders: list[DataLoader[Any]] = []
    for split in ("val_id", "val_dg"):
        dataset = dataset_type(config, split)
        workers = 0 if cached else config.evaluation.num_workers
        arguments: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": config.evaluation.batch_size,
            "shuffle": False,
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
        }
        if workers > 0:
            arguments["prefetch_factor"] = config.evaluation.prefetch_factor
            arguments["persistent_workers"] = False
        loaders.append(DataLoader(**arguments))
    return loaders, cached


@torch.inference_mode()
def _collect_logits(
    model: Any, loaders: list[DataLoader[Any]], device: torch.device
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    logits: list[float] = []
    labels: list[float] = []
    counts: dict[str, int] = {}
    for split, loader in zip(("val_id", "val_dg"), loaders):
        split_count = 0
        for batch in tqdm(loader, desc=f"Calibrate {split}", leave=False):
            clean, transformed, targets = _move_batch(batch, device)
            clean_output, transformed_output = _forward_pair(model, clean, transformed)
            batch_size = targets.numel()
            logits.extend(clean_output.logits.float().cpu().tolist())
            labels.extend(targets.float().cpu().tolist())
            logits.extend(transformed_output.logits.float().cpu().tolist())
            labels.extend(targets.float().cpu().tolist())
            split_count += batch_size
        counts[f"{split}_base_images"] = split_count
        counts[f"{split}_predictions"] = split_count * 2
    return (
        np.asarray(logits, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        counts,
    )


def fit_global_calibration(
    config: AppConfig,
    *,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Fit one affine calibrator on pooled clean/transformed ID and DG validation."""
    validate_preparation(config)
    seed_everything(config.project.seed)
    selected_checkpoint = Path(checkpoint_path or config.evaluation.checkpoint_path)
    config, checkpoint = load_inference_checkpoint(config, selected_checkpoint)
    loaders, cached = _validation_loaders(config)
    device = resolve_device(config)
    model = create_cached_detector(config.model) if cached else create_detector(config.model)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    model.to(device).eval()
    raw_logits, labels, counts = _collect_logits(model, loaders, device)

    estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1_000)
    estimator.fit(raw_logits.reshape(-1, 1), labels)
    coefficient = float(estimator.coef_[0, 0])
    intercept = float(estimator.intercept_[0])
    if coefficient <= 0:
        raise RuntimeError("Fitted calibration reverses the detector ordering.")
    calibrated = 1.0 / (1.0 + np.exp(-(coefficient * raw_logits + intercept)))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, calibrated)
    balanced = 0.5 * (true_positive_rate + 1.0 - false_positive_rate)
    finite = np.isfinite(thresholds)
    best_index = int(np.flatnonzero(finite)[np.argmax(balanced[finite])])
    threshold = float(thresholds[best_index])
    raw_probabilities = 1.0 / (1.0 + np.exp(-raw_logits))
    checkpoint_sha256 = _file_sha256(selected_checkpoint)
    destination = Path(output_path) if output_path else calibration_path(config, selected_checkpoint)
    payload = {
        "schema_version": 1,
        "method": "affine_platt",
        "checkpoint_path": str(selected_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "coefficient": coefficient,
        "intercept": intercept,
        "threshold": threshold,
        "fit_counts": counts,
        "feature_cache_used": cached,
        "raw_metrics_at_0_5": binary_metrics(labels, raw_probabilities, 0.5),
        "calibrated_metrics": {
            **binary_metrics(labels, calibrated, threshold),
            "balanced_accuracy_at_selected_threshold": float(
                balanced_accuracy_score(labels, calibrated >= threshold)
            ),
        },
    }
    _atomic_json(destination, payload)
    LOGGER.info(
        "Global calibration saved: path=%s coefficient=%.6f intercept=%.6f threshold=%.6f",
        destination,
        coefficient,
        intercept,
        threshold,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return destination


def main() -> None:
    """Fit a global calibration file for one trained checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(
        "Fit global affine calibration on internal clean/transformed ID and DG validation."
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override.")
    parser.add_argument("--output", default=None, help="Calibration JSON override.")
    arguments = parser.parse_args()
    fit_global_calibration(
        load_config(arguments.config, arguments.set),
        checkpoint_path=arguments.checkpoint,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
