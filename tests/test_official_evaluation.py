import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from aigc_recognizer.config import load_config
from aigc_recognizer.external_eval import (
    EvaluationDatasetSpec,
    _balanced_stable_sample,
    _load_or_create_external_features,
)
from aigc_recognizer.model import FrozenClipDetector
from aigc_recognizer.official_data import select_official_rows
from aigc_recognizer.official_eval import _extended_metrics, _scenario_image


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


def test_official_row_selection_is_exact_and_safe() -> None:
    dalle = [
        {"IsAdvanced": "1", "IsFake": "1", "Image_path": "./Diffusion_based/DALLE/Advanced/a.jpg"},
        {"IsAdvanced": "0", "IsFake": "1", "Image_path": "./Diffusion_based/DALLE/Typical/b.jpg"},
    ]
    coco = [
        {"IsAdvanced": "0", "IsFake": "0", "Image_path": "./Real/coco/coco2017/val2017/c.jpg"},
        {"IsAdvanced": "0", "IsFake": "0", "Image_path": "./Real/coco/coco2017/train2017/d.jpg"},
    ]

    selected = select_official_rows(dalle, coco)

    assert selected == [
        {"label": 1, "source_name": "DALL-E Advanced", "member": "DALLE/Advanced/a.jpg"},
        {"label": 0, "source_name": "COCO val2017", "member": "coco/coco2017/val2017/c.jpg"},
    ]


@pytest.mark.parametrize(
    "scenario",
    [
        "clean",
        "jpeg_30",
        "blur_2.0",
        "resize_0.25",
        "noise_0.10",
        "color_jitter_0.20",
        "center_crop_0.80",
    ],
)
def test_official_scenarios_preserve_image_shape_and_range(scenario: str) -> None:
    image = Image.fromarray(np.full((24, 32, 3), 127, dtype=np.uint8), mode="RGB")
    first = _scenario_image(image.copy(), scenario, random.Random(7))
    second = _scenario_image(image.copy(), scenario, random.Random(7))

    assert first.size == image.size
    assert first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.asarray(first).min() >= 0
    assert np.asarray(first).max() <= 255


def test_extended_metrics_include_confusion_counts() -> None:
    metrics = _extended_metrics([0.0, 0.0, 1.0, 1.0], [0.1, 0.8, 0.7, 0.2], 0.5)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["count"] == 4


def test_fast_external_sample_is_deterministic_and_label_balanced() -> None:
    records = [
        {"id": f"real-{index}", "label": 0} for index in range(9)
    ] + [{"id": f"fake-{index}", "label": 1} for index in range(13)]

    first = _balanced_stable_sample(records, 10, 2026)
    second = _balanced_stable_sample(records, 10, 2026)

    assert first == second
    assert len(first) == 10
    assert sum(int(record["label"]) == 0 for record in first) == 5
    assert sum(int(record["label"]) == 1 for record in first) == 5


def test_external_frozen_features_are_cached_and_reused(tmp_path: Path) -> None:
    image_root = tmp_path / "evaluation"
    image_root.mkdir()
    records = []
    for label in (0, 1):
        for index in range(2):
            name = f"{label}-{index}.png"
            Image.new("RGB", (20, 24), color=label * 200).save(image_root / name)
            records.append(
                {
                    "id": f"id-{label}-{index}",
                    "path": name,
                    "label": label,
                    "source_name": f"source-{label}",
                }
            )
    manifest = image_root / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    spec = EvaluationDatasetSpec(
        name="smoke",
        repo_id="local",
        revision="revision",
        output_dir=str(image_root),
        manifest_path=str(manifest),
        audit_path=str(image_root / "audit.json"),
        results_path=str(tmp_path / "results.json"),
        predictions_path=str(tmp_path / "predictions.jsonl"),
        expected_real=2,
        expected_fake=2,
    )
    config = load_config(DEFAULT_CONFIG)
    config.views.input_size = 16
    config.model.embedding_dim = 8
    config.model.intermediate_layers = []
    config.model.head_dim = 6
    config.model.projection_dim = 4
    config.training.device = "cpu"
    config.training.amp = False
    config.evaluation.batch_size = 2
    config.evaluation.num_workers = 0
    config.evaluation.external_feature_cache_dir = str(tmp_path / "cache")
    model = FrozenClipDetector(DummyVisualEncoder(8), config.model).eval()

    first = _load_or_create_external_features(
        model, config, torch.device("cpu"), spec, "clean", max_samples=None
    )
    second = _load_or_create_external_features(
        model, config, torch.device("cpu"), spec, "clean", max_samples=None
    )

    assert first["final"].shape == (4, 2, 8)
    assert torch.equal(first["final"], second["final"])
    assert len(list((tmp_path / "cache").rglob("*.pt"))) == 1


@pytest.mark.parametrize(
    "scenario",
    [
        "combo_social_resize_0.5_jpeg_70",
        "combo_repost_jpeg_90_resize_0.5_jpeg_70",
        "combo_crop_0.80_resize_0.5_jpeg_70",
        "combo_blur_1.0_resize_0.5_jpeg_50",
        "combo_edit_color_0.20_noise_0.02_jpeg_70",
        "combo_stress_crop_0.80_blur_1.0_resize_0.25_jpeg_30",
    ],
)
def test_composed_scenarios_are_deterministic_and_preserve_shape(scenario: str) -> None:
    pixels = np.arange(32 * 24 * 3, dtype=np.uint8).reshape(24, 32, 3)
    image = Image.fromarray(pixels, mode="RGB")

    first = _scenario_image(image.copy(), scenario, random.Random(19))
    second = _scenario_image(image.copy(), scenario, random.Random(19))

    assert first.size == image.size
    assert first.mode == "RGB"
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(image))
