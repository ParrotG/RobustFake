from pathlib import Path

import torch
from PIL import Image

from aigc_recognizer.config import load_config
from aigc_recognizer.data.transforms import RobustPairTransform


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_paired_transform_has_expected_shape_and_is_deterministic() -> None:
    config = load_config(DEFAULT_CONFIG)
    config.views.input_size = 32
    transform = RobustPairTransform(config)
    image = Image.new("RGBA", (60, 40), (100, 150, 200, 128))
    first = transform(image, seed=123)
    second = transform(image, seed=123)
    assert first["clean_views"].shape == (2, 3, 32, 32)
    assert first["transformed_views"].shape == (2, 3, 32, 32)
    assert torch.equal(first["clean_views"], second["clean_views"])
    assert torch.equal(first["transformed_views"], second["transformed_views"])
    assert torch.isfinite(first["transformed_views"]).all()
