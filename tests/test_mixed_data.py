import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from aigc_recognizer.config import load_config
from aigc_recognizer.data.mixed import (
    DedupIndex,
    PreparationError,
    _cleanup_project_payload,
    _community_row_candidate,
    _load_source_cache,
    _matches_generator_alias,
    _save_source_cache,
    assign_tiny,
    commit_candidates,
    load_external_denylist,
    nuisance_buckets,
    select_shanmuk,
    square_root_allocation,
)


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 20), (value, value // 2, 255 - value))
    draw = ImageDraw.Draw(image)
    draw.line((value % 7, 0, 23, value % 19), fill=(255 - value, value, 30), width=2)
    image.save(path)


def test_default_mixed_quotas_are_exact() -> None:
    config = load_config(DEFAULT_CONFIG)
    counts = Counter()
    for splits in config.mixed_data.source_quotas.values():
        for split, labels in splits.items():
            for label, amount in labels.items():
                counts[(split, int(label))] += amount
    assert counts == {
        ("train", 0): 32_000,
        ("train", 1): 32_000,
        ("val_id", 0): 4_000,
        ("val_id", 1): 4_000,
        ("val_dg", 0): 4_000,
        ("val_dg", 1): 4_000,
    }


def test_square_root_allocation_is_exact_bounded_and_deterministic() -> None:
    first = square_root_allocation(12, {"large": 100, "small": 4, "medium": 25})
    second = square_root_allocation(12, {"medium": 25, "small": 4, "large": 100})
    assert first == second
    assert sum(first.values()) == 12
    assert first["large"] > first["medium"] > first["small"]
    assert first["small"] <= 4


def test_nuisance_buckets_cover_locked_boundaries() -> None:
    assert nuisance_buckets(191, 400, "JPEG", 1000)["resolution_bucket"] == "lt192"
    assert nuisance_buckets(256, 256, "PNG", 1000)["resolution_bucket"] == "256_383"
    assert nuisance_buckets(800, 500, "WEBP", 1000)["aspect_bucket"] == "landscape"


def test_heldout_alias_does_not_match_unrelated_image_name() -> None:
    assert _matches_generator_alias("MAGE-v2", "mage")
    assert not _matches_generator_alias("ImageMagick", "mage")


def test_source_candidate_checkpoint_is_idempotent_and_invalidates_missing_files(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"candidate")
    state = {
        "revision": "a" * 40,
        "completed_units": ["shard-1", "shard-1"],
        "row_count": 10,
        "candidates": [
            {"source_id": "id-1", "local_path": str(staged)},
            {"source_id": "id-1", "local_path": str(staged)},
        ],
    }
    _save_source_cache(tmp_path, state)
    loaded = _load_source_cache(tmp_path, "a" * 40)
    assert loaded["completed_units"] == ["shard-1"]
    assert len(loaded["candidates"]) == 1
    staged.unlink()
    invalidated = _load_source_cache(tmp_path, "a" * 40)
    assert invalidated["completed_units"] == []
    assert invalidated["row_count"] == 0


def test_external_deny_index_is_cached_and_resumable(tmp_path: Path, monkeypatch) -> None:
    config = load_config(DEFAULT_CONFIG)
    evaluation = tmp_path / "evaluation"
    image_path = evaluation / "image.png"
    _image(image_path, 110)
    manifest = evaluation / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"id": "external-1", "path": "image.png", "label": 0}) + "\n",
        encoding="utf-8",
    )
    config.mixed_data.external_deny_manifests = [str(manifest)]
    config.mixed_data.cache_dir = str(tmp_path / "cache")
    first = load_external_denylist(config)
    assert len(first) == 1
    assert first[0]["crop_resistant_hash"]

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("The cached external hash should be reused.")

    monkeypatch.setattr("aigc_recognizer.data.mixed.describe_path", fail_if_recomputed)
    second = load_external_denylist(config)
    assert second == first


def test_dedup_bk_tree_stores_one_node_per_unique_phash() -> None:
    config = load_config(DEFAULT_CONFIG)
    shared_hash = "8" + "0" * 63
    denied = [
        {
            "id": f"external-{index}",
            "label": 0,
            "split": "external_test",
            "perceptual_hash": shared_hash,
            "difference_hash": "0" * 64,
        }
        for index in range(100)
    ]
    index = DedupIndex(config, denied)
    assert len(index.hashes[int(shared_hash, 16)]) == 100
    assert index.hash_tree is not None
    assert index.hash_tree[1] == {}


def test_community_row_decode_and_project_payload_cleanup(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    encoded = tmp_path / "source.png"
    _image(encoded, 140)
    candidate = _community_row_candidate(
        {
            "label": 1,
            "split": "train",
            "nsfw_flag": False,
            "model_name": "example/model",
            "subset": "Systematic",
            "architecture": "LatDiff",
            "image_name": "sample.png",
            "image_data": encoded.read_bytes(),
        },
        config,
        tmp_path / "community",
    )
    assert candidate is not None
    assert Path(candidate["local_path"]).is_file()

    payload = tmp_path / "community" / "payload"
    (payload / "nested").mkdir(parents=True)
    (payload / "nested" / "shard.parquet").write_bytes(b"payload")
    _cleanup_project_payload(payload)
    assert not any(path.is_file() for path in payload.rglob("*"))


def test_tiny_assignment_accepts_only_the_pinned_empty_sd14_class() -> None:
    config = load_config(DEFAULT_CONFIG)
    active = [name for name in config.mixed_data.tiny_generators if name != "SD14"]
    candidates = []
    for index in range(10_000):
        candidates.append(
            {
                "source_dataset": "tiny_genimage",
                "source_revision": config.mixed_data.tiny_genimage_revision,
                "source_id": f"real-{index}",
                "label": 0,
                "generator": "real",
                "real_source": "ImageNet",
                "architecture": "real",
            }
        )
    for generator in active:
        for index in range(1_500):
            candidates.append(
                {
                    "source_dataset": "tiny_genimage",
                    "source_revision": config.mixed_data.tiny_genimage_revision,
                    "source_id": f"{generator}-{index}",
                    "label": 1,
                    "generator": generator,
                    "real_source": "",
                    "architecture": generator,
                }
            )
    assigned = assign_tiny(candidates, config)
    primary = Counter(
        (item["split"], int(item["label"]))
        for item in assigned
        if int(item.get("selection_tier", 0)) == 0
    )
    assert primary == {
        ("train", 0): 6_000,
        ("train", 1): 6_000,
        ("val_id", 0): 2_000,
        ("val_id", 1): 1_000,
        ("val_dg", 1): 1_000,
    }
    assert "SD14" not in {item["generator"] for item in assigned if item["label"] == 1}

    with pytest.raises(PreparationError, match="empty-class exception"):
        assign_tiny([item for item in candidates if item.get("generator") != "ADM"], config)


def test_shanmuk_selection_keeps_parent_pairs_and_is_stable(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    root = tmp_path / "source"
    config.mixed_data.shanmuk_root = str(root)
    records = []
    # reserve_multiplier is reduced so a compact synthetic source can exercise the same logic.
    config.mixed_data.reserve_multiplier = 1.0
    shared = root / "images" / "shared.png"
    _image(shared, 80)
    for index in range(5_000):
        parent = f"p{index:05d}"
        for label in (0, 1):
            records.append(
                {
                    "id": f"{parent}-{label}",
                    "source_real_id": parent,
                    "path": str(shared.relative_to(root)),
                    "label": label,
                    "generator": "real" if label == 0 else config.data.generators[index % 6],
                    "source_dataset": "coco" if index % 2 == 0 else "imagenet",
                    "source_revision": config.mixed_data.shanmuk_revision,
                    "source_split": "train",
                    "split": "train",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    first = select_shanmuk(config)
    second = select_shanmuk(config)
    assert [item["source_id"] for item in first] == [item["source_id"] for item in second]
    assert Counter(item["split"] for item in first) == {"train": 8_000, "val_id": 2_000}
    assert all(len([item for item in first if item["pair_id"] == parent]) == 2 for parent in {item["pair_id"] for item in first})
    fake_counts = Counter(item["generator"] for item in first if int(item["label"]) == 1)
    real_counts = Counter(item["real_source"] for item in first if int(item["label"]) == 0)
    assert max(fake_counts.values()) - min(fake_counts.values()) <= 1
    assert max(real_counts.values()) - min(real_counts.values()) <= 1


def test_commit_uses_reserve_after_same_label_duplicate(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.mixed_data.output_dir = str(tmp_path / "mixed")
    config.mixed_data.cache_dir = str(tmp_path / "cache")
    config.mixed_data.checkpoint_every = 100
    config.mixed_data.target_total = 2
    config.mixed_data.target_real = 1
    config.mixed_data.target_fake = 1
    config.mixed_data.source_quotas = {
        source: {
            split: {"0": int(source == "tiny_genimage" and split == "train"), "1": int(source == "tiny_genimage" and split == "train")}
            for split in ("train", "val_id", "val_dg")
        }
        for source in ("shanmuk", "wildfake", "community_forensics", "tiny_genimage")
    }
    real = tmp_path / "real.png"
    fake = tmp_path / "fake.png"
    reserve = tmp_path / "reserve.png"
    _image(real, 30)
    _image(fake, 220)
    _image(reserve, 180)

    def candidate(identity: str, label: int, path: Path) -> dict:
        return {
            "source_dataset": "tiny_genimage",
            "source_revision": config.mixed_data.tiny_genimage_revision,
            "source_id": identity,
            "split": "train",
            "label": label,
            "generator": "real" if label == 0 else "ADM",
            "architecture": "real" if label == 0 else "ADM",
            "model_id": "real" if label == 0 else "ADM",
            "local_path": str(path),
        }

    denied = [
        {
            **candidate("external", 1, fake),
            "id": "external",
            "split": "external_test",
            **__import__("aigc_recognizer.data.mixed", fromlist=["describe_path"]).describe_path(fake, config),
        }
    ]
    selected, _report = commit_candidates(
        [candidate("real", 0, real), candidate("duplicate", 1, fake), candidate("reserve", 1, reserve)],
        config,
        denied,
    )
    assert len(selected) == 2
    assert {item["source_id"] for item in selected} == {"real", "reserve"}


def test_commit_rejects_conflicting_exact_duplicate(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config.mixed_data.output_dir = str(tmp_path / "mixed")
    config.mixed_data.cache_dir = str(tmp_path / "cache")
    config.mixed_data.target_total = 1
    config.mixed_data.target_real = 0
    config.mixed_data.target_fake = 1
    config.mixed_data.source_quotas = {
        source: {
            split: {"0": 0, "1": int(source == "tiny_genimage" and split == "train")}
            for split in ("train", "val_id", "val_dg")
        }
        for source in ("shanmuk", "wildfake", "community_forensics", "tiny_genimage")
    }
    path = tmp_path / "same.png"
    _image(path, 90)
    from aigc_recognizer.data.mixed import describe_path

    denied = [{"id": "real", "label": 0, "split": "external_test", **describe_path(path, config)}]
    candidate = {
        "source_dataset": "tiny_genimage",
        "source_revision": config.mixed_data.tiny_genimage_revision,
        "source_id": "fake",
        "split": "train",
        "label": 1,
        "generator": "ADM",
        "architecture": "ADM",
        "model_id": "ADM",
        "local_path": str(path),
    }
    with pytest.raises(PreparationError, match="Conflicting labels"):
        commit_candidates([candidate], config, denied)
