import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from aigc_recognizer.config import load_config
from aigc_recognizer.model import FrozenClipDetector
from aigc_recognizer.predict import discover_images, predict_directory


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


def test_directory_prediction_writes_stable_required_json(tmp_path: Path) -> None:
    input_directory = tmp_path / "images"
    nested = input_directory / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (24, 20), "white").save(input_directory / "b.PNG")
    Image.new("RGB", (20, 24), "black").save(nested / "a.jpg")
    (input_directory / "ignored.txt").write_text("not an image", encoding="utf-8")

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
    model = FrozenClipDetector(DummyVisualEncoder(8), config.model).eval()
    output = tmp_path / "predictions.json"

    first = predict_directory(config, input_directory, output, model=model)
    second = predict_directory(config, input_directory, output, model=model)

    assert first == second
    assert json.loads(output.read_text(encoding="utf-8")) == first
    assert [set(item) for item in first] == [
        {"image_path", "pred"},
        {"image_path", "pred"},
    ]
    assert [Path(item["image_path"]).name for item in first] == ["b.PNG", "a.jpg"]
    assert all(0.0 <= item["pred"] <= 1.0 for item in first)


def test_image_discovery_rejects_empty_or_missing_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no supported images"):
        discover_images(tmp_path)
    with pytest.raises(NotADirectoryError):
        discover_images(tmp_path / "missing")
