"""Evaluate a trained detector on the prescribed WildFake subset and severities."""

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


def _atomic_write(path: Path, content: bytes) -> None:
    """Atomically replace one evaluation artifact."""
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


def _scenario_image(
    image: Image.Image,
    scenario: str,
    rng: random.Random,
) -> Image.Image:
    """Apply one exact challenge severity before spatial view rendering."""
    if scenario == "clean":
        return image
    if scenario.startswith("jpeg_"):
        quality = int(scenario.split("_", 1)[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if scenario.startswith("blur_"):
        return image.filter(ImageFilter.GaussianBlur(float(scenario.split("_", 1)[1])))
    if scenario.startswith("resize_"):
        scale = float(scenario.split("_", 1)[1])
        width, height = image.size
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=Image.Resampling.BICUBIC,
        )
        return reduced.resize((width, height), resample=Image.Resampling.BICUBIC)
    if scenario.startswith("noise_"):
        sigma = float(scenario.split("_", 1)[1])
        array = np.asarray(image, dtype=np.float32) / 255.0
        generator = np.random.default_rng(rng.getrandbits(64))
        noisy = np.clip(array + generator.normal(0.0, sigma, array.shape), 0.0, 1.0)
        return Image.fromarray(np.round(noisy * 255.0).astype(np.uint8), mode="RGB")
    if scenario == "color_jitter_0.20":
        operations = [
            (ImageEnhance.Brightness, rng.uniform(0.8, 1.2)),
            (ImageEnhance.Contrast, rng.uniform(0.8, 1.2)),
            (ImageEnhance.Color, rng.uniform(0.8, 1.2)),
        ]
        rng.shuffle(operations)
        for enhancer, factor in operations:
            image = enhancer(image).enhance(factor)
        return image
    if scenario == "center_crop_0.80":
        width, height = image.size
        crop_width = max(1, round(width * 0.8))
        crop_height = max(1, round(height * 0.8))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), resample=Image.Resampling.BICUBIC)
    raise ValueError(f"Unsupported official evaluation scenario: {scenario}")


class OfficialEvaluationDataset(Dataset[dict[str, Any]]):
    """Load the isolated official manifest under one deterministic scenario."""

    def __init__(self, config: AppConfig, scenario: str) -> None:
        self.config = config
        self.scenario = scenario
        self.root = Path(config.official_evaluation.output_dir)
        manifest_path = Path(config.official_evaluation.manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Official evaluation manifest does not exist: {manifest_path}")
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
            raise RuntimeError(f"Failed to decode official evaluation image: {image_path}") from exc
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
            "id": record["id"],
            "path": record["path"],
            "source_name": record["source_name"],
        }


def _loader(config: AppConfig, scenario: str) -> DataLoader[Any]:
    official = config.official_evaluation
    arguments: dict[str, Any] = {
        "dataset": OfficialEvaluationDataset(config, scenario),
        "batch_size": official.batch_size,
        "shuffle": False,
        "num_workers": official.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if official.num_workers > 0:
        arguments["prefetch_factor"] = official.prefetch_factor
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


@torch.no_grad()
def _evaluate_scenario(
    model: FrozenClipDetector,
    config: AppConfig,
    device: torch.device,
    scenario: str,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    labels: list[float] = []
    probabilities: list[float] = []
    predictions: list[dict[str, Any]] = []
    loader = _loader(config, scenario)
    for batch in tqdm(loader, desc=f"Evaluate {scenario}"):
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
                batch["id"],
                batch["path"],
                batch_labels,
                batch["source_name"],
                scores,
            )
        )
    return _extended_metrics(labels, probabilities, config.training.threshold), predictions


def evaluate_official(config: AppConfig) -> dict[str, Any]:
    """Evaluate best.pt without mutating the model, checkpoint, or training state."""
    official = config.official_evaluation
    audit_path = Path(official.audit_path)
    if not audit_path.is_file():
        raise FileNotFoundError("Official evaluation audit is missing; run preparation first.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("complete")):
        raise RuntimeError("Official evaluation preparation is incomplete.")
    expected_counts = {
        "real": official.expected_real_count,
        "fake": official.expected_fake_count,
    }
    if audit.get("counts") != expected_counts:
        raise RuntimeError("Official evaluation audit count does not match the configuration.")

    checkpoint_path = Path(official.checkpoint_path)
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

    scenario_results: dict[str, dict[str, float | int]] = {}
    all_predictions: list[dict[str, Any]] = []
    for scenario in official.scenarios:
        metrics, predictions = _evaluate_scenario(model, config, device, scenario)
        scenario_results[scenario] = metrics
        if official.save_predictions:
            all_predictions.extend(predictions)
    clean_auroc = float(scenario_results.get("clean", {}).get("auroc", math.nan))
    transformed_aurocs = [
        float(metrics["auroc"])
        for name, metrics in scenario_results.items()
        if name != "clean"
    ]
    result = {
        "schema_version": 1,
        "dataset": {
            "repo_id": official.repo_id,
            "revision": official.revision,
            "counts": audit["counts"],
            "archives": audit["archives"],
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
            "mean_transformed_auroc": (
                float(np.mean(transformed_aurocs)) if transformed_aurocs else math.nan
            ),
            "worst_transformed_auroc": (
                float(np.min(transformed_aurocs)) if transformed_aurocs else math.nan
            ),
        },
    }
    _atomic_write(
        Path(official.results_path),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if official.save_predictions:
        prediction_lines = "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in all_predictions
        )
        _atomic_write(Path(official.predictions_path), prediction_lines.encode("utf-8"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    """Run the challenge-prescribed external evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Evaluate a detector on the official WildFake subset.")
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    result = evaluate_official(config)
    LOGGER.info(
        "Official evaluation completed: clean AUROC=%.6f mean transformed AUROC=%.6f.",
        result["summary"]["clean_auroc"],
        result["summary"]["mean_transformed_auroc"],
    )


if __name__ == "__main__":
    main()
