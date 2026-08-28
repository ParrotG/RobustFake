import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from aigc_recognizer.config import load_config
from aigc_recognizer.data.prepare import sample_stream, stable_bucket


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def image_bytes(index: int) -> bytes:
    image = Image.new("RGB", (32, 32), (index * 13 % 255, index * 29 % 255, index * 47 % 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((index % 12, index % 9, 20 + index % 10, 24), outline="white", width=2)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def identity_for_split(prefix: str, expected: str, train_percent: int = 50) -> str:
    for index in range(10_000):
        value = f"{prefix}-{index}"
        actual = "train" if stable_bucket(value) < train_percent else "val"
        if actual == expected:
            return value
    raise AssertionError("Unable to construct a stable split identity.")


def test_sampler_is_balanced_generator_disjoint_and_idempotent(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(tmp_path / "dataset" / "manifest.jsonl")
    config.data.audit_path = str(tmp_path / "dataset" / "audit.json")
    config.data.train_per_class = 4
    config.data.val_per_class = 4
    config.data.train_generator_percent = 50
    config.data.architecture_ratios = {
        "LatDiff": 0.25,
        "GAN": 0.25,
        "PixDiff": 0.25,
        "other": 0.25,
    }
    config.data.max_real_source_fraction = 0.5
    config.data.perceptual_deduplication = False
    config.data.max_scanned = 100

    rows = []
    index = 0
    for split in ("train", "val"):
        for architecture in ("LatDiff", "GAN", "PixDiff", "other"):
            model_name = identity_for_split(f"{split}-{architecture}", split)
            rows.append(
                {
                    "image_name": f"fake-{index}.png",
                    "image_data": image_bytes(index),
                    "format": "PNG",
                    "model_name": model_name,
                    "nsfw_flag": False,
                    "real_source": "LAION",
                    "subset": "Systematic",
                    "label": 1,
                    "architecture": architecture,
                }
            )
            index += 1
        for source in ("VISION", "FFHQ"):
            needed = 2
            found = 0
            candidate = 0
            while found < needed:
                image_name = f"{split}-real-{source}-{candidate}.png"
                identity = f"{image_name}|{source}"
                actual = "train" if stable_bucket(identity) < 50 else "val"
                candidate += 1
                if actual != split:
                    continue
                rows.append(
                    {
                        "image_name": image_name,
                        "image_data": image_bytes(index),
                        "format": "PNG",
                        "model_name": "real",
                        "nsfw_flag": False,
                        "real_source": source,
                        "subset": "real",
                        "label": 0,
                        "architecture": "real",
                    }
                )
                index += 1
                found += 1

    audit = sample_stream(rows, config, "test-revision")
    first_manifest = Path(config.data.manifest_path).read_text(encoding="utf-8")
    second_audit = sample_stream(rows, config, "test-revision")
    second_manifest = Path(config.data.manifest_path).read_text(encoding="utf-8")
    records = [json.loads(line) for line in first_manifest.splitlines()]

    assert audit["complete"] is True
    assert second_audit["selected"] == 16
    assert first_manifest == second_manifest
    assert len(records) == 16
    train_models = {r["model_name"] for r in records if r["label"] == 1 and r["split"] == "train"}
    val_models = {r["model_name"] for r in records if r["label"] == 1 and r["split"] == "val"}
    assert train_models.isdisjoint(val_models)
