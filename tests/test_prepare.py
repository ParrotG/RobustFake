import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from aigc_recognizer.config import load_config
from aigc_recognizer.data.prepare import (
    _iter_local_shards,
    effective_real_source,
    is_forbidden,
    sample_rows,
    stable_bucket,
)


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

    audit = sample_rows(rows, config, "test-revision")
    first_manifest = Path(config.data.manifest_path).read_text(encoding="utf-8")
    second_audit = sample_rows(rows, config, "test-revision")
    second_manifest = Path(config.data.manifest_path).read_text(encoding="utf-8")
    records = [json.loads(line) for line in first_manifest.splitlines()]

    assert audit["complete"] is True
    assert second_audit["selected"] == 16
    assert first_manifest == second_manifest
    assert len(records) == 16
    train_models = {r["model_name"] for r in records if r["label"] == 1 and r["split"] == "train"}
    val_models = {r["model_name"] for r in records if r["label"] == 1 and r["split"] == "val"}
    assert train_models.isdisjoint(val_models)


def test_interrupted_sampler_checkpoints_and_resumes(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(tmp_path / "dataset" / "manifest.jsonl")
    config.data.audit_path = str(tmp_path / "dataset" / "audit.json")
    config.data.train_per_class = 1
    config.data.val_per_class = 1
    config.data.train_generator_percent = 50
    config.data.architecture_ratios = {
        "LatDiff": 1.0,
        "GAN": 0.0,
        "PixDiff": 0.0,
        "other": 0.0,
    }
    config.data.max_real_source_fraction = 1.0
    config.data.perceptual_deduplication = False
    config.data.checkpoint_every_scanned = 1

    rows = []
    for index, split in enumerate(("train", "val")):
        rows.append(
            {
                "image_name": f"fake-{split}.png",
                "image_data": image_bytes(index),
                "format": "PNG",
                "model_name": identity_for_split(f"resume-model-{split}", split),
                "nsfw_flag": False,
                "real_source": "LAION",
                    "subset": "Systematic",
                "label": 1,
                "architecture": "LatDiff",
            }
        )
    for index, split in enumerate(("train", "val"), start=2):
        source = "VISION"
        candidate = 0
        while True:
            image_name = f"resume-real-{split}-{candidate}"
            assigned = (
                "train"
                if stable_bucket(f"{image_name}|{source}") < config.data.train_generator_percent
                else "val"
            )
            if assigned == split:
                break
            candidate += 1
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

    def interrupted_rows():
        yield from rows[:2]
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        sample_rows(interrupted_rows(), config, "resume-revision")
    partial_audit = json.loads(Path(config.data.audit_path).read_text(encoding="utf-8"))
    assert partial_audit["complete"] is False
    assert partial_audit["selected"] == 2

    resumed_audit = sample_rows(rows, config, "resume-revision")
    assert resumed_audit["complete"] is True
    assert resumed_audit["selected"] == 4


def test_real_source_falls_back_to_model_name_and_excludes_coco() -> None:
    config = load_config(DEFAULT_CONFIG)
    coco = {"label": 0, "real_source": "N/A", "model_name": "COCO"}
    vision = {"label": 0, "real_source": "N/A", "model_name": "VISION"}

    assert effective_real_source(coco) == "COCO"
    assert effective_real_source(vision) == "VISION"
    assert is_forbidden(coco, config) is True
    assert is_forbidden(vision, config) is False


def test_local_shard_reader_uses_completed_hub_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    local_shard = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "image_name": ["one.png"],
                "image_data": [image_bytes(41)],
                "label": [1],
                "architecture": ["GAN"],
                "model_name": ["test-generator"],
                "subset": ["Manual"],
                "real_source": ["N/A"],
            }
        ),
        local_shard,
    )

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(local_shard)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    config = load_config(DEFAULT_CONFIG)
    config.data.shard_indices = [68]
    rows = list(
        _iter_local_shards(
            config,
            "test-revision",
            "test-token",
            ["data/HFCF_small_68.parquet"],
        )
    )

    assert len(rows) == 1
    assert rows[0]["architecture"] == "GAN"
    assert calls[0]["revision"] == "test-revision"
    assert calls[0]["local_dir"] == config.data.shard_cache_dir
