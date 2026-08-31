import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn

from aigc_recognizer.config import load_config
from aigc_recognizer.model import FrozenClipDetector
from aigc_recognizer.train import run_training


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


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


def test_cpu_smoke_training_and_resume(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(make_manifest(Path(config.data.output_dir)))
    config.data.audit_path = str(Path(config.data.output_dir) / "audit.json")
    config.views.input_size = 16
    config.model.embedding_dim = 8
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
