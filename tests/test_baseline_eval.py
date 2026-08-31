import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from aigc_recognizer import baseline_eval
from aigc_recognizer.baseline_eval import (
    BaselineEvaluationDataset,
    CNNDetectionBaseline,
    CLIP_MEAN,
    CLIP_STD,
    UnivFDBaseline,
    _scenario_metrics,
)
from aigc_recognizer.config import ConfigError, load_config
from aigc_recognizer.external_eval import EvaluationDatasetSpec


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(3, 2)
        self.fc = nn.Linear(2, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(images.mean(dim=(-1, -2))))


class TinyVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 768)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


def test_cnndetection_loads_official_checkpoint_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = TinyCNN()
    source.fc = nn.Linear(2, 1)
    checkpoint = tmp_path / "cnn.pth"
    torch.save({"model": source.state_dict()}, checkpoint)
    monkeypatch.setattr(baseline_eval, "resnet50", lambda weights=None: TinyCNN())

    model = CNNDetectionBaseline(checkpoint).eval()

    assert model(torch.zeros(2, 3, 8, 8)).shape == (2,)


def test_univfd_loads_linear_head_and_freezes_visual_encoder(tmp_path: Path) -> None:
    head = nn.Linear(768, 1)
    checkpoint = tmp_path / "univfd.pth"
    torch.save(head.state_dict(), checkpoint)
    visual = TinyVisual()

    model = UnivFDBaseline(
        checkpoint,
        "ViT-L-14-quickgelu",
        "openai",
        visual_encoder=visual,
    ).eval()

    assert model(torch.zeros(2, 3, 8, 8)).shape == (2,)
    assert not any(parameter.requires_grad for parameter in visual.parameters())


def test_baseline_dataset_applies_scenario_then_official_crop(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    root.mkdir()
    Image.new("RGB", (300, 260), color=(90, 120, 150)).save(root / "sample.png")
    record = {
        "id": "sample",
        "path": "sample.png",
        "label": 1,
        "source_name": "generator",
    }
    manifest = root / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    spec = EvaluationDatasetSpec(
        name="smoke",
        repo_id="local",
        revision="revision",
        output_dir=str(root),
        manifest_path=str(manifest),
        audit_path=str(root / "audit.json"),
        results_path=str(tmp_path / "results.json"),
        predictions_path=str(tmp_path / "predictions.jsonl"),
        expected_real=0,
        expected_fake=1,
    )
    config = load_config(DEFAULT_CONFIG)

    dataset = BaselineEvaluationDataset(
        config,
        spec,
        "jpeg_30",
        mean=CLIP_MEAN,
        std=CLIP_STD,
    )
    first = dataset[0]
    second = dataset[0]

    assert first["image"].shape == (3, 224, 224)
    assert torch.equal(first["image"], second["image"])
    assert first["label"].item() == 1.0


def test_baseline_metrics_keep_uncalibrated_predictions() -> None:
    payload = {
        "logits": torch.tensor([-2.0, 2.0]),
        "label": torch.tensor([0.0, 1.0]),
        "id": ["real", "fake"],
        "path": ["real.png", "fake.png"],
        "source_name": ["real-source", "fake-source"],
        "elapsed_seconds": 0.5,
    }

    metrics, predictions = _scenario_metrics(payload, "clean", 0.5)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["images_per_second"] == pytest.approx(4.0)
    assert predictions[0]["pred"] == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())


def test_invalid_baseline_hash_is_rejected() -> None:
    with pytest.raises(ConfigError, match="CNNDetection baseline SHA-256"):
        load_config(
            DEFAULT_CONFIG,
            ["baseline_evaluation.cnndetection_sha256=invalid"],
        )
