import collections
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from aigc_recognizer.config import AppConfig
from aigc_recognizer.external_data import (
    _detect_corrupt_gan_members,
    _select_sid_shard,
    _wildfake_checkpoint,
    select_wildfake_rows,
)


def _wildfake_row(
    family: str, architecture: str, index: int, *, label: int
) -> dict[str, str]:
    level = "Real" if label == 0 else family
    return {
        "Generator": family,
        "Architecture": architecture,
        "Weight": architecture,
        "Category": architecture,
        "IsAdvanced": str(index % 2),
        "IsFake": str(label),
        "Image_path": f"./{level}/{architecture}/{index:04d}.png",
        "Num": str(index),
    }


def test_wildfake_selection_is_balanced_hierarchical_and_reproducible() -> None:
    config = AppConfig()
    broad = config.wildfake_evaluation
    broad.target_real = 8
    broad.target_fake = 12
    broad.real_sources = ["real_a", "real_b"]
    broad.fake_families = ["GAN_based", "Diffusion_based"]
    broad.fake_architectures = ["gan_a", "gan_b", "diff_a", "diff_b"]
    rows = []
    for source in broad.real_sources:
        rows.extend(_wildfake_row("Real", source, index, label=0) for index in range(20))
    for family, architectures in {
        "GAN_based": ["gan_a", "gan_b"],
        "Diffusion_based": ["diff_a", "diff_b"],
    }.items():
        for architecture in architectures:
            rows.extend(_wildfake_row(family, architecture, index, label=1) for index in range(20))

    first = select_wildfake_rows(rows, config)
    second = select_wildfake_rows(reversed(rows), config)

    assert first == second
    assert collections.Counter(item["label"] for item in first) == {0: 8, 1: 12}
    assert collections.Counter(item["family"] for item in first if item["label"] == 1) == {
        "Diffusion_based": 6,
        "GAN_based": 6,
    }
    assert collections.Counter(item["architecture"] for item in first if item["label"] == 1) == {
        "diff_a": 3,
        "diff_b": 3,
        "gan_a": 3,
        "gan_b": 3,
    }

    excluded = first[-1]["source_path"]
    broad.excluded_source_paths = [excluded]
    replacement = select_wildfake_rows(rows, config)
    assert len(replacement) == len(first)
    assert excluded not in {item["source_path"] for item in replacement}
    assert {item["source_path"] for item in replacement} != {
        item["source_path"] for item in first
    }

    broad.excluded_source_paths = []
    additional = select_wildfake_rows(rows, config, [excluded])
    assert excluded not in {item["source_path"] for item in additional}
    assert len(additional) == len(first)


def test_sid_per_shard_hash_selection_is_exact_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "validation-00000.parquet"
    table = pa.table(
        {
            "img_id": [f"real-{index}" for index in range(20)]
            + [f"fake-{index}" for index in range(20)]
            + [f"tampered-{index}" for index in range(20)],
            "label": [0] * 20 + [1] * 20 + [2] * 20,
        }
    )
    pq.write_table(table, path)

    first = _select_sid_shard(path, path.name, 5, 7, 2026)
    second = _select_sid_shard(path, path.name, 5, 7, 2026)

    assert first == second
    assert len(first[0]) == 5
    assert len(first[1]) == 7
    assert all(item.startswith("real-") for item in first[0])
    assert all(item.startswith("fake-") for item in first[1])


def test_wildfake_checkpoint_uses_persisted_generator_family_field(tmp_path: Path) -> None:
    config = AppConfig()
    broad = config.wildfake_evaluation
    broad.manifest_path = str(tmp_path / "manifest.jsonl")
    broad.audit_path = str(tmp_path / "audit.json")
    broad.state_path = str(tmp_path / "state.json")
    broad.integrity_cache_path = str(tmp_path / "missing-integrity.json")
    record = {
        "id": "record-1",
        "path": "images/fake/record-1.png",
        "label": 1,
        "generator_family": "GAN_based",
        "architecture": "BigGAN",
        "bytes": 123,
    }

    audit = _wildfake_checkpoint(
        config,
        {"record-1": record},
        {"record-1"},
        {"Images/GAN_based.zip": "archive-hash"},
        "fingerprint",
        False,
        "test checkpoint",
    )

    assert audit["sampling"]["strata"] == {"GAN_based/BigGAN": 1}
    assert Path(broad.manifest_path).is_file()
    assert Path(broad.audit_path).is_file()
    assert Path(broad.state_path).is_file()


def test_wildfake_integrity_cache_avoids_remote_rescan(tmp_path: Path) -> None:
    config = AppConfig()
    cache_path = tmp_path / "archive_integrity.json"
    config.wildfake_evaluation.integrity_cache_path = str(cache_path)
    cache_path.write_text(
        json.dumps(
            {
                "archive_path": "Images/GAN_based.zip",
                "archive_sha256": "archive-hash",
                "compression_ratio": 0.02,
                "suspicious_members_checked": 2,
                "corrupt_source_paths": ["GAN_based/Advanced/GigaGAN/bad.png"],
            }
        ),
        encoding="utf-8",
    )

    rejected = _detect_corrupt_gan_members(config, "archive-hash")

    assert rejected == {"GAN_based/Advanced/GigaGAN/bad.png"}
