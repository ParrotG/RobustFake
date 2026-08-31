"""Evaluate official CNNDetection and UnivFD baselines on shared manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import functional as tvf
from tqdm import tqdm

from aigc_recognizer.config import AppConfig, config_argument_parser, load_config
from aigc_recognizer.data.transforms import canonical_rgb
from aigc_recognizer.external_eval import (
    EvaluationDatasetSpec,
    _atomic_write,
    _balanced_stable_sample,
    _extended_metrics,
    _scenario_image,
    _source_group_metrics,
    dataset_spec,
)
from aigc_recognizer.train import resolve_device
from aigc_recognizer.utils import atomic_torch_save, seed_everything, seed_worker

LOGGER = logging.getLogger(__name__)

SUPPORTED_BASELINES = ("cnndetection", "univfd")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_local_artifact(path: Path, expected_sha256: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {path}")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Baseline checkpoint SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, received {actual}."
        )
    return path


def _download_verified_artifact(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    maximum_bytes: int,
    retries: int,
    retry_backoff_seconds: float,
) -> Path:
    """Download one immutable external checkpoint and atomically publish it."""
    if destination.is_file():
        return _verified_local_artifact(destination, expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                declared_size = int(response.headers.get("content-length", "0") or 0)
                if declared_size > maximum_bytes:
                    raise RuntimeError(
                        f"Baseline checkpoint exceeds the configured download limit: {url}"
                    )
                downloaded = 0
                digest = hashlib.sha256()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > maximum_bytes:
                            raise RuntimeError(
                                f"Baseline checkpoint exceeds the configured download limit: {url}"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    "Downloaded baseline checkpoint SHA-256 mismatch: "
                    f"expected {expected_sha256}, received {actual_sha256}."
                )
            os.replace(temporary, destination)
            break
        except (requests.RequestException, OSError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed to download baseline checkpoint: {url}") from exc
            delay = retry_backoff_seconds * (2**attempt)
            LOGGER.warning(
                "Baseline download failed; retrying in %.1f seconds: %s", delay, exc
            )
            time.sleep(delay)
        finally:
            if temporary.exists():
                temporary.unlink()
    LOGGER.info("Baseline checkpoint ready: %s", destination)
    return destination


def resolve_baseline_checkpoint(config: AppConfig, name: str) -> Path:
    """Resolve an optional local checkpoint or download the pinned official artifact."""
    baseline = config.baseline_evaluation
    maximum_bytes = round(baseline.max_download_gb * 1024**3)
    cache = Path(baseline.cache_dir)
    if name == "cnndetection":
        if baseline.cnndetection_checkpoint_path:
            path = Path(baseline.cnndetection_checkpoint_path)
        else:
            path = cache / "cnndetection" / "blur_jpg_prob0.5.pth"
            return _download_verified_artifact(
                baseline.cnndetection_url,
                path,
                baseline.cnndetection_sha256,
                maximum_bytes=maximum_bytes,
                retries=baseline.download_retries,
                retry_backoff_seconds=baseline.retry_backoff_seconds,
            )
        return _verified_local_artifact(path, baseline.cnndetection_sha256)
    if name == "univfd":
        if baseline.univfd_checkpoint_path:
            path = Path(baseline.univfd_checkpoint_path)
        else:
            path = cache / "univfd" / "fc_weights.pth"
            return _download_verified_artifact(
                baseline.univfd_url,
                path,
                baseline.univfd_sha256,
                maximum_bytes=maximum_bytes,
                retries=baseline.download_retries,
                retry_backoff_seconds=baseline.retry_backoff_seconds,
            )
        return _verified_local_artifact(path, baseline.univfd_sha256)
    raise ValueError(f"Unsupported baseline: {name}")


class CNNDetectionBaseline(nn.Module):
    """Official CVPR 2020 CNNDetection ResNet-50 binary classifier."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError("CNNDetection checkpoint does not contain model weights.")
        self.model.load_state_dict(state, strict=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).squeeze(-1)


class UnivFDBaseline(nn.Module):
    """Official UnivFD linear head on an OpenAI CLIP ViT-L/14 image encoder."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        backbone_name: str,
        pretrained: str,
        *,
        visual_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if visual_encoder is None:
            import open_clip

            clip_model = open_clip.create_model(backbone_name, pretrained=pretrained)
            visual_encoder = clip_model.visual
        self.visual_encoder = visual_encoder
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()
        self.classifier = nn.Linear(768, 1)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError("UnivFD checkpoint must contain its linear-head state dictionary.")
        self.classifier.load_state_dict(state, strict=True)

    def train(self, mode: bool = True) -> "UnivFDBaseline":
        super().train(mode)
        self.visual_encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.visual_encoder(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if not isinstance(features, torch.Tensor) or features.shape[-1] != 768:
            raise RuntimeError("UnivFD requires 768-dimensional CLIP image features.")
        return self.classifier(features.float()).squeeze(-1)


@dataclass(frozen=True)
class LoadedBaseline:
    """A baseline model together with its exact public artifact identity."""

    name: str
    display_name: str
    model: nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    repository_revision: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    metadata: dict[str, Any]


def load_baseline(config: AppConfig, name: str) -> LoadedBaseline:
    """Load one pinned official baseline without changing its learned weights."""
    checkpoint = resolve_baseline_checkpoint(config, name)
    baseline = config.baseline_evaluation
    if name == "cnndetection":
        model: nn.Module = CNNDetectionBaseline(checkpoint)
        return LoadedBaseline(
            name=name,
            display_name="CNNDetection (CVPR 2020)",
            model=model,
            checkpoint_path=checkpoint,
            checkpoint_sha256=baseline.cnndetection_sha256,
            repository_revision=baseline.cnndetection_repository_revision,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            metadata={
                "architecture": "ResNet-50",
                "checkpoint": "blur_jpg_prob0.5.pth",
                "paper": "CNN-generated images are surprisingly easy to spot... for now",
                "paper_url": "https://openaccess.thecvf.com/content_CVPR_2020/html/"
                "Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_"
                "CVPR_2020_paper.html",
                "repository": "https://github.com/PeterWang512/CNNDetection",
                "preprocessing": "center crop 224; ImageNet normalization",
                "score": "sigmoid of the official uncalibrated synthetic logit",
            },
        )
    if name == "univfd":
        model = UnivFDBaseline(
            checkpoint,
            baseline.univfd_backbone_name,
            baseline.univfd_pretrained,
        )
        return LoadedBaseline(
            name=name,
            display_name="UnivFD (CVPR 2023)",
            model=model,
            checkpoint_path=checkpoint,
            checkpoint_sha256=baseline.univfd_sha256,
            repository_revision=baseline.univfd_repository_revision,
            mean=CLIP_MEAN,
            std=CLIP_STD,
            metadata={
                "architecture": "OpenAI CLIP ViT-L/14 plus linear head",
                "checkpoint": "fc_weights.pth",
                "paper": "Towards Universal Fake Image Detectors That Generalize Across Generative Models",
                "paper_url": "https://openaccess.thecvf.com/content/CVPR2023/html/"
                "Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_"
                "Generative_Models_CVPR_2023_paper.html",
                "repository": "https://github.com/WisconsinAIVision/UniversalFakeDetect",
                "preprocessing": "center crop 224; OpenAI CLIP normalization",
                "score": "sigmoid of the official uncalibrated synthetic logit",
            },
        )
    raise ValueError(f"Unsupported baseline: {name}")


class BaselineEvaluationDataset(Dataset[dict[str, Any]]):
    """Apply shared scenarios followed by one baseline's official preprocessing."""

    def __init__(
        self,
        config: AppConfig,
        spec: EvaluationDatasetSpec,
        scenario: str,
        *,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        max_samples: int | None = None,
    ) -> None:
        self.config = config
        self.spec = spec
        self.scenario = scenario
        self.mean = mean
        self.std = std
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

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root / str(record["path"])
        try:
            with Image.open(image_path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to decode baseline evaluation image: {image_path}") from exc
        seed_digest = hashlib.sha256(
            f"{self.config.project.seed}:{record['id']}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(seed_digest[:16], 16))
        image = canonical_rgb(image, self.config.views.padding_color)
        image = _scenario_image(image, self.scenario, rng)
        image = tvf.center_crop(
            image,
            [
                self.config.baseline_evaluation.input_size,
                self.config.baseline_evaluation.input_size,
            ],
        )
        tensor = tvf.pil_to_tensor(image).float().div_(255.0)
        tensor = tvf.normalize(tensor, self.mean, self.std)
        return {
            "image": tensor,
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "id": str(record["id"]),
            "path": str(record["path"]),
            "source_name": str(record.get("source_name", "unknown")),
        }


def _baseline_loader(
    config: AppConfig,
    loaded: LoadedBaseline,
    spec: EvaluationDatasetSpec,
    scenario: str,
    *,
    max_samples: int | None,
) -> DataLoader[Any]:
    workers = config.evaluation.num_workers
    arguments: dict[str, Any] = {
        "dataset": BaselineEvaluationDataset(
            config,
            spec,
            scenario,
            mean=loaded.mean,
            std=loaded.std,
            max_samples=max_samples,
        ),
        "batch_size": config.evaluation.batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        arguments["prefetch_factor"] = config.evaluation.prefetch_factor
        arguments["persistent_workers"] = False
    return DataLoader(**arguments)


def _score_cache_identity(
    config: AppConfig,
    loaded: LoadedBaseline,
    spec: EvaluationDatasetSpec,
    scenario: str,
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    record_ids = [str(record["id"]) for record in records]
    return {
        "schema_version": 1,
        "baseline": loaded.name,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "repository_revision": loaded.repository_revision,
        "dataset": spec.name,
        "manifest_sha256": _file_sha256(Path(spec.manifest_path)),
        "record_ids_sha256": hashlib.sha256("\n".join(record_ids).encode()).hexdigest(),
        "scenario": scenario,
        "seed": config.project.seed,
        "input_size": config.baseline_evaluation.input_size,
        "mean": loaded.mean,
        "std": loaded.std,
        "metadata": loaded.metadata,
    }


def _score_cache_path(config: AppConfig, identity: dict[str, Any]) -> Path:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    key = hashlib.sha256(serialized.encode()).hexdigest()
    return (
        Path(config.baseline_evaluation.score_cache_dir)
        / str(identity["dataset"])
        / str(identity["baseline"])
        / f"{key}.pt"
    )


@torch.inference_mode()
def _load_or_score_scenario(
    config: AppConfig,
    loaded: LoadedBaseline,
    spec: EvaluationDatasetSpec,
    scenario: str,
    device: torch.device,
    *,
    max_samples: int | None,
) -> dict[str, Any]:
    loader = _baseline_loader(
        config, loaded, spec, scenario, max_samples=max_samples
    )
    dataset = loader.dataset
    if not isinstance(dataset, BaselineEvaluationDataset):
        raise TypeError("Baseline evaluator received an unexpected dataset type.")
    identity = _score_cache_identity(
        config, loaded, spec, scenario, dataset.records
    )
    cache_path = _score_cache_path(config, identity)
    if config.baseline_evaluation.score_cache_enabled and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("identity") != identity:
            raise RuntimeError(f"Baseline score-cache identity mismatch: {cache_path}")
        LOGGER.info("Using baseline score cache: %s", cache_path)
        return cached

    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    identifiers: list[str] = []
    paths: list[str] = []
    source_names: list[str] = []
    started = time.perf_counter()
    for batch in tqdm(loader, desc=f"Evaluate {loaded.name}/{spec.name}/{scenario}"):
        images = batch["image"].to(device, non_blocking=True)
        if config.training.amp and device.type == "cuda":
            dtype = torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = loaded.model(images)
        else:
            output = loaded.model(images)
        logits.append(output.float().cpu())
        labels.append(batch["label"].float().cpu())
        identifiers.extend(str(value) for value in batch["id"])
        paths.extend(str(value) for value in batch["path"])
        source_names.extend(str(value) for value in batch["source_name"])
    elapsed = time.perf_counter() - started
    payload = {
        "identity": identity,
        "logits": torch.cat(logits),
        "label": torch.cat(labels),
        "id": identifiers,
        "path": paths,
        "source_name": source_names,
        "elapsed_seconds": elapsed,
    }
    if config.baseline_evaluation.score_cache_enabled:
        atomic_torch_save(payload, cache_path)
    return payload


def _scenario_metrics(
    payload: dict[str, Any], scenario: str, threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = payload["label"].float().tolist()
    probabilities = torch.sigmoid(payload["logits"].float()).tolist()
    predictions = [
        {
            "id": record_id,
            "image_path": path,
            "label": int(label),
            "source_name": source_name,
            "scenario": scenario,
            "pred": float(score),
        }
        for record_id, path, label, source_name, score in zip(
            payload["id"],
            payload["path"],
            labels,
            payload["source_name"],
            probabilities,
        )
    ]
    metrics = _extended_metrics(labels, probabilities, threshold)
    metrics["source_groups"] = _source_group_metrics(predictions, threshold)
    elapsed = float(payload.get("elapsed_seconds", math.nan))
    metrics["inference_seconds"] = elapsed
    metrics["images_per_second"] = (
        len(labels) / elapsed if elapsed > 0 and math.isfinite(elapsed) else math.nan
    )
    return metrics, predictions


def _validate_prepared_evaluation(spec: EvaluationDatasetSpec) -> dict[str, Any]:
    audit_path = Path(spec.audit_path)
    if not audit_path.is_file():
        raise FileNotFoundError(
            "Evaluation audit is missing; run the matching preparation first."
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool(audit.get("complete")):
        raise RuntimeError("External evaluation preparation is incomplete.")
    expected = {"real": spec.expected_real, "fake": spec.expected_fake}
    if audit.get("counts") != expected:
        raise RuntimeError("External evaluation audit count does not match the configuration.")
    return audit


def _result_paths(
    config: AppConfig, spec: EvaluationDatasetSpec, name: str, fast: bool
) -> tuple[Path, Path]:
    root = Path(config.baseline_evaluation.results_dir) / spec.name / name
    suffix = ".fast" if fast else ""
    return root / f"results{suffix}.json", root / f"predictions{suffix}.jsonl"


def evaluate_baseline(
    config: AppConfig,
    name: str,
    dataset_name: str,
    *,
    fast: bool = False,
    loaded_baseline: LoadedBaseline | None = None,
) -> dict[str, Any]:
    """Evaluate one official baseline on one prepared external dataset."""
    if name not in SUPPORTED_BASELINES:
        raise ValueError(f"Unsupported baseline: {name}")
    spec = dataset_spec(config, dataset_name)
    audit = _validate_prepared_evaluation(spec)
    seed_everything(config.project.seed)
    loaded = loaded_baseline or load_baseline(config, name)
    device = resolve_device(config)
    loaded.model.to(device).eval()
    scenarios = (
        list(config.evaluation.fast_scenarios)
        if fast
        else list(config.evaluation.scenarios)
    )
    if not fast and config.evaluation.enable_composed_scenarios:
        scenarios.extend(config.evaluation.composed_scenarios)
    max_samples = config.evaluation.fast_max_samples if fast else None
    threshold = config.baseline_evaluation.threshold
    scenario_results: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []
    for scenario in scenarios:
        payload = _load_or_score_scenario(
            config,
            loaded,
            spec,
            scenario,
            device,
            max_samples=max_samples,
        )
        metrics, predictions = _scenario_metrics(payload, scenario, threshold)
        scenario_results[scenario] = metrics
        if config.evaluation.save_predictions:
            all_predictions.extend(predictions)

    single = [
        float(scenario_results[scenario]["auroc"])
        for scenario in scenarios
        if scenario != "clean" and scenario not in config.evaluation.composed_scenarios
    ]
    composed = [
        float(scenario_results[scenario]["auroc"])
        for scenario in config.evaluation.composed_scenarios
        if scenario in scenario_results
    ]
    result = {
        "schema_version": 1,
        "mode": "fast" if fast else "full",
        "baseline": {
            "name": loaded.name,
            "display_name": loaded.display_name,
            "checkpoint_path": str(loaded.checkpoint_path),
            "checkpoint_sha256": loaded.checkpoint_sha256,
            "repository_revision": loaded.repository_revision,
            **loaded.metadata,
        },
        "dataset": {
            "name": spec.name,
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "counts": audit["counts"],
        },
        "threshold": threshold,
        "calibration": {"applied": False, "method": "none"},
        "evaluated_sample_count": int(
            next(iter(scenario_results.values())).get("count", 0)
        ),
        "scenarios": scenario_results,
        "summary": {
            "clean_auroc": float(scenario_results["clean"]["auroc"]),
            "mean_single_transform_auroc": float(np.mean(single)) if single else math.nan,
            "worst_single_transform_auroc": float(np.min(single)) if single else math.nan,
            "mean_composed_transform_auroc": (
                float(np.mean(composed)) if composed else math.nan
            ),
            "worst_composed_transform_auroc": (
                float(np.min(composed)) if composed else math.nan
            ),
        },
    }
    results_path, predictions_path = _result_paths(config, spec, name, fast)
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


def evaluate_baselines(
    config: AppConfig,
    names: list[str],
    dataset_name: str,
    *,
    fast: bool = False,
) -> dict[str, Any]:
    """Evaluate multiple baselines and publish one presentation-friendly comparison."""
    results = {
        name: evaluate_baseline(config, name, dataset_name, fast=fast)
        for name in names
    }
    comparison = {
        "schema_version": 1,
        "mode": "fast" if fast else "full",
        "dataset": dataset_name,
        "baselines": {
            name: result["summary"] for name, result in results.items()
        },
    }
    suffix = ".fast" if fast else ""
    destination = (
        Path(config.baseline_evaluation.results_dir)
        / dataset_name
        / f"comparison{suffix}.json"
    )
    _atomic_write(
        destination,
        (json.dumps(comparison, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return results


def download_baselines(config: AppConfig, names: Iterable[str]) -> dict[str, Path]:
    """Download and verify the small detector checkpoints without evaluating data."""
    return {name: resolve_baseline_checkpoint(config, name) for name in names}


def main() -> None:
    """Run official external baselines through the shared transformation matrix."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(
        "Evaluate pinned CNNDetection and UnivFD baselines on a prepared external dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=("wildfake_official", "wildfake_broad", "sid_set"),
        default="wildfake_official",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=SUPPORTED_BASELINES,
        default=[],
        help="Baseline to evaluate; repeat to select both. Defaults to both.",
    )
    parser.add_argument("--fast", action="store_true")
    arguments = parser.parse_args()
    names = arguments.baseline or list(SUPPORTED_BASELINES)
    results = evaluate_baselines(
        load_config(arguments.config, arguments.set),
        names,
        arguments.dataset,
        fast=arguments.fast,
    )
    for name, result in results.items():
        LOGGER.info(
            "%s clean AUROC=%.6f mean single-transform AUROC=%.6f",
            name,
            result["summary"]["clean_auroc"],
            result["summary"]["mean_single_transform_auroc"],
        )


def main_download() -> None:
    """Download and verify the pinned baseline detector checkpoints."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser("Download pinned external baseline checkpoints.")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=SUPPORTED_BASELINES,
        default=[],
        help="Baseline to download; repeat to select both. Defaults to both.",
    )
    arguments = parser.parse_args()
    names = arguments.baseline or list(SUPPORTED_BASELINES)
    for name, path in download_baselines(
        load_config(arguments.config, arguments.set), names
    ).items():
        LOGGER.info("%s checkpoint: %s", name, path)


if __name__ == "__main__":
    main()
