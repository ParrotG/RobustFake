import random

import numpy as np
import pytest
from PIL import Image

from aigc_recognizer.official_data import select_official_rows
from aigc_recognizer.official_eval import _extended_metrics, _scenario_image


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
