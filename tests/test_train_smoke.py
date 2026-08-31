import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

from aigc_recognizer.config import load_config
from aigc_recognizer.data.dataset import (
    canonical_generator_family,
    canonical_group_name,
)
from aigc_recognizer.model import FrozenClipDetector
from aigc_recognizer.feature_cache import CachedFeatureDataset, precompute_features
from aigc_recognizer.train import (
    _consistency_scale,
    _group_metrics,
    make_loaders,
    run_training,
)


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


class DummyIntermediateEncoder(DummyVisualEncoder):
    def __init__(self, output_dim: int, intermediate_dim: int) -> None:
        super().__init__(output_dim)
        self.intermediate_projection = nn.Linear(3, intermediate_dim)

    def forward_intermediates(
        self,
        images: torch.Tensor,
        *,
        indices: list[int],
        **_kwargs: object,
    ) -> dict[str, object]:
        pooled = images.mean(dim=(-1, -2))
        prefix = self.intermediate_projection(pooled).unsqueeze(1)
        return {
            "image_features": self.projection(pooled),
            "image_intermediates_prefix": [prefix + index for index in indices],
        }


def make_manifest(root: Path) -> Path:
    records = []
    for split in ("train", "val_id", "val_dg"):
        for label in (0, 1):
            for index in range(2):
                record_id = f"{split}-{label}-{index}"
                relative = Path("images") / split / str(label) / f"{record_id}.png"
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                base = 35 if label == 0 else 220
                image = Image.new("RGB", (24, 24), (base, base, base))
                ImageDraw.Draw(image).line((0, index * 5, 23, 23 - index * 5), fill="red", width=2)
                image.save(destination)
                records.append(
                    {
                        "id": record_id,
                        "path": str(relative),
                        "label": label,
                        "split": split,
                        "source_revision": "smoke-revision",
                    }
                )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (root / "audit.json").write_text(
        json.dumps({"complete": True, "selected": len(records)}) + "\n",
        encoding="utf-8",
    )
    return manifest


def detector_for(config) -> FrozenClipDetector:
    return FrozenClipDetector(DummyVisualEncoder(8), config.model)


def test_reporting_group_aliases_are_canonical() -> None:
    assert canonical_group_name("SD15", "architecture") == "sd15"
    assert canonical_group_name("sd15", "architecture") == "sd15"
    assert canonical_group_name("ImageNet", "real_source") == "imagenet"
    assert canonical_group_name("laion5b", "real_source") == "laion"
    assert canonical_group_name("GAN_based", "generator_family") == "gan"
    assert canonical_generator_family("LatDiff", "LatDiff") == "diffusion"
    assert canonical_generator_family("unknown", "SD15") == "diffusion"


def test_training_loader_covers_every_prepared_record_once(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(make_manifest(Path(config.data.output_dir)))
    config.training.batch_size = 2
    config.training.num_workers = 0

    train_loader, _val_id_loader, _val_dg_loader = make_loaders(config)
    indices = list(train_loader.sampler)

    assert len(indices) == len(train_loader.dataset)
    assert len(set(indices)) == len(train_loader.dataset)


def test_single_class_group_metrics_report_recall_instead_of_skipping() -> None:
    fake = _group_metrics([1.0, 1.0], [0.9, 0.2], 0.5)
    real = _group_metrics([0.0, 0.0], [0.7, 0.1], 0.5)

    assert fake["fake_count"] == 2
    assert fake["fake_recall"] == 0.5
    assert "auroc" not in fake
    assert real["real_count"] == 2
    assert real["real_recall"] == 0.5
    assert real["false_positive_rate"] == 0.5


def test_consistency_scale_ramps_over_opening_epochs() -> None:
    assert _consistency_scale(0, 0, 10, 3) > 0.0
    assert _consistency_scale(0, 9, 10, 3) == 1 / 3
    assert _consistency_scale(1, 9, 10, 3) == 2 / 3
    assert _consistency_scale(2, 9, 10, 3) == 1.0
    assert _consistency_scale(0, 0, 10, 0) == 1.0


def test_cpu_smoke_training_and_resume(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(make_manifest(Path(config.data.output_dir)))
    config.data.audit_path = str(Path(config.data.output_dir) / "audit.json")
    config.views.input_size = 16
    config.model.embedding_dim = 8
    config.model.intermediate_layers = []
    config.model.head_dim = 6
    config.model.projection_dim = 4
    config.training.device = "cpu"
    config.training.amp = False
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 1
    config.training.num_workers = 0
    config.training.early_stopping_patience = 2
    config.output.root_dir = str(tmp_path / "runs")
    config.project.run_name = "smoke"

    best = run_training(config, detector_for(config))
    last = best.parent / "last.pt"
    assert best.is_file()
    assert last.is_file()
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["source_revision"] == "smoke-revision"
    assert all(not key.startswith("visual_encoder.") for key in checkpoint["trainable_model"])

    config.training.epochs = 2
    config.training.resume_from = str(last)
    resumed_best = run_training(config, detector_for(config))
    assert resumed_best.is_file()
    metric_lines = (best.parent / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metric_lines) == 2
    latest = json.loads(metric_lines[-1])
    assert "validation_id" in latest
    assert "validation_dg" in latest


def test_feature_cache_is_resumable_and_trains_without_backbone(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(make_manifest(Path(config.data.output_dir)))
    config.data.audit_path = str(Path(config.data.output_dir) / "audit.json")
    config.views.input_size = 16
    config.model.embedding_dim = 8
    config.model.intermediate_layers = [0, 1]
    config.model.intermediate_dim = 5
    config.model.head_dim = 6
    config.model.projection_dim = 4
    config.training.device = "cpu"
    config.training.amp = False
    config.training.epochs = 1
    config.training.batch_size = 2
    config.training.num_workers = 0
    config.output.root_dir = str(tmp_path / "runs")
    config.project.run_name = "cached-smoke"
    config.feature_cache.root_dir = str(tmp_path / "feature-cache")
    config.feature_cache.train_variants = 2
    config.feature_cache.shard_size = 3
    config.feature_cache.batch_size = 2
    config.feature_cache.num_workers = 0

    encoder = DummyIntermediateEncoder(8, 5)
    directory = precompute_features(
        config, FrozenClipDetector(encoder, config.model)
    )
    manifest_path = directory / "cache_manifest.json"
    first_manifest = manifest_path.read_bytes()
    precompute_features(config, FrozenClipDetector(encoder, config.model))
    assert manifest_path.read_bytes() == first_manifest

    cached = CachedFeatureDataset(config, "train")
    assert len(cached) == 4
    assert cached.tensors["clean_final"].shape == (2, 4, 2, 8)
    assert cached.tensors["clean_intermediate"].shape == (2, 4, 2, 2, 5)

    config.feature_cache.use_for_training = True
    best = run_training(config)
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert any(
        key.startswith("intermediate_projections.")
        for key in checkpoint["trainable_model"]
    )
