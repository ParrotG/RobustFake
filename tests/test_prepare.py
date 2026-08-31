import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from aigc_recognizer.config import load_config
from aigc_recognizer.data.prepare import (
    PreparationError,
    _checkpoint,
    _describe_image,
    _excluded_parents,
    _extract_selected_rows,
    _source_shards,
    select_paired_metadata,
)


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def metadata_rows(parent_count: int = 12) -> list[dict]:
    generators = [
        "sd15",
        "sdxl",
        "flux_schnell",
        "kandinsky22",
        "pixart_sigma",
        "wuerstchen",
    ]
    rows = []
    for index in range(parent_count):
        parent = f"real_{index:06d}"
        split = "train" if index < 6 else "val" if index < 9 else "test"
        for generator in ["real", *generators]:
            image_id = parent if generator == "real" else f"{generator}_{index:06d}"
            rows.append(
                {
                    "image_id": image_id,
                    "source_real_id": parent,
                    "generator": generator,
                    "label": int(generator != "real"),
                    "sha256": hashlib.sha256(image_id.encode()).hexdigest(),
                    "split": split,
                }
            )
    return rows


def png_bytes(value: int, size: int = 16) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (value, value // 2, 255 - value)).save(buffer, "PNG")
    return buffer.getvalue()


def test_pair_selection_is_stable_balanced_and_split_safe() -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.expected_parent_count = 12
    rows = metadata_rows()

    first = select_paired_metadata(rows, config)
    second = select_paired_metadata(reversed(rows), config)

    assert first == second
    assert len(first) == 24
    for split, expected in (("train", 6), ("val", 3), ("test", 3)):
        selected = [row for row in first.values() if row["split"] == split]
        assert Counter(row["label"] for row in selected) == {0: expected, 1: expected}
        generator_counts = Counter(row["generator"] for row in selected if row["label"] == 1)
        assert max(generator_counts.values()) - min(generator_counts.values()) <= 1
    split_by_parent = {}
    for row in first.values():
        split_by_parent.setdefault(row["source_real_id"], row["split"])
        assert split_by_parent[row["source_real_id"]] == row["split"]


def test_pair_selection_rejects_missing_partner() -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.expected_parent_count = 12
    rows = metadata_rows()
    rows.pop()
    with pytest.raises(PreparationError, match="does not have one real and six fakes"):
        select_paired_metadata(rows, config)


def test_source_shard_preflight_is_bounded() -> None:
    config = load_config(DEFAULT_CONFIG)
    siblings = [
        SimpleNamespace(rfilename="real/train/a.parquet", size=100),
        SimpleNamespace(rfilename="sd15/val/b.parquet", size=200),
        SimpleNamespace(rfilename="metadata/manifest.parquet", size=10),
    ]
    assert _source_shards(siblings, config) == ["real/train/a.parquet", "sd15/val/b.parquet"]
    config.data.max_download_gb = 1e-12
    with pytest.raises(PreparationError, match="exceeding data.max_download_gb"):
        _source_shards(siblings, config)


def test_selected_parquet_row_is_validated_and_written(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.expected_image_size = 16
    content = png_bytes(80)
    digest = hashlib.sha256(content).hexdigest()
    source = {
        "image": {"bytes": content, "path": None},
        "image_id": "real_000001",
        "source_real_id": "real_000001",
        "label": 0,
        "generator": "real",
        "source_dataset": "imagenet",
        "split": "validation",
        "prompt": "a test image",
        "width": 16,
        "height": 16,
        "pipeline_version": "1.2",
        "sha256": digest,
    }
    parquet = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([source]), parquet)
    expected = {
        "real_000001": {
            "image_id": "real_000001",
            "source_real_id": "real_000001",
            "label": 0,
            "generator": "real",
            "split": "val",
            "sha256": digest,
        }
    }
    records = {}

    _extract_selected_rows(parquet, expected, config, records)

    assert records["real_000001"]["split"] == "val"
    assert records["real_000001"]["source_split"] == "validation"
    assert records["real_000001"]["content_sha256"] == digest
    assert (Path(config.data.output_dir) / records["real_000001"]["path"]).is_file()


def test_checkpoint_is_idempotent_and_preserves_expected_ids(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.output_dir = str(tmp_path / "dataset")
    config.data.manifest_path = str(tmp_path / "dataset" / "manifest.jsonl")
    config.data.audit_path = str(tmp_path / "dataset" / "audit.json")
    config.data.state_path = str(tmp_path / "dataset" / "state.json")
    image = Path(config.data.output_dir) / "images/train/real/a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(png_bytes(30))
    record = {
        "id": "a",
        "path": "images/train/real/a.png",
        "label": 0,
        "split": "train",
        "generator": "real",
        "source_real_id": "a",
    }
    arguments = (config, "fingerprint", {"one.parquet"}, ["one.parquet"], {"a"}, {"a": record})

    _checkpoint(*arguments, False, "test")
    first = Path(config.data.manifest_path).read_bytes()
    _checkpoint(*arguments, False, "test")

    assert Path(config.data.manifest_path).read_bytes() == first
    state = json.loads(Path(config.data.state_path).read_text())
    assert state["expected_ids"] == ["a"]
    assert state["completed_shards"] == ["one.parquet"]


def test_official_match_excludes_entire_parent_pair(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.data.official_leakage_root = str(tmp_path / "official")
    config.data.official_leakage_manifest = str(tmp_path / "official" / "manifest.jsonl")
    official_image = tmp_path / "official" / "images" / "real.png"
    official_image.parent.mkdir(parents=True)
    official_image.write_bytes(png_bytes(120))
    Path(config.data.official_leakage_manifest).write_text(
        json.dumps({"id": "official", "path": "images/real.png", "label": 0}) + "\n"
    )
    description = _describe_image(official_image.read_bytes(), config)
    common = {
        "source_real_id": "real_1",
        "content_sha256": hashlib.sha256(official_image.read_bytes()).hexdigest(),
        "perceptual_hash": description["perceptual_hash"],
        "difference_hash": description["difference_hash"],
    }
    records = {
        "real_1": {**common, "label": 0, "source_dataset": "coco"},
        "fake_1": {
            **common,
            "label": 1,
            "source_dataset": "sd15",
            "content_sha256": "f" * 64,
        },
    }

    assert _excluded_parents(records, config) == {"real_1"}
