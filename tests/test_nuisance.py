import json
from pathlib import Path

import numpy as np
from PIL import Image

from aigc_recognizer.config import load_config
from aigc_recognizer.data.nuisance import (
    FEATURE_NAMES,
    _group_metrics,
    extract_nuisance_features,
    run_nuisance_audit,
)


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_nuisance_features_are_finite_and_fixed_width() -> None:
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    features = extract_nuisance_features(image, 32)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_nuisance_single_class_group_reports_class_recall() -> None:
    result = _group_metrics(
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([0.8, 0.1], dtype=np.float64),
    )

    assert result["fake_count"] == 2
    assert result["fake_recall"] == 0.5
    assert "auroc" not in result


def test_nuisance_report_detects_obvious_low_level_bias(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    root = tmp_path / "dataset"
    config.data.output_dir = str(root)
    config.data.manifest_path = str(root / "manifest.jsonl")
    config.data.nuisance_report_path = str(root / "nuisance.json")
    config.nuisance_audit.feature_size = 24
    config.nuisance_audit.max_iter = 20
    config.nuisance_audit.min_samples_leaf = 2
    config.nuisance_audit.permutation_repeats = 1
    records = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            for index in range(12):
                record_id = f"{split}-{label}-{index}"
                relative = Path("images") / split / f"{record_id}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                value = 20 + index if label == 0 else 220 - index
                Image.new("RGB", (24, 24), (value, value, value)).save(path)
                records.append(
                    {
                        "id": record_id,
                        "path": str(relative),
                        "label": label,
                        "split": split,
                        "generator": "real" if label == 0 else config.data.generators[index % 6],
                        "source_dataset": "coco" if label == 0 else "synthetic",
                        "format": "png",
                        "width": 24,
                        "height": 24,
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    Path(config.data.manifest_path).write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = run_nuisance_audit(config)

    assert report["informational_only"] is True
    assert report["raw_canonical"]["splits"]["val"]["auroc"] > 0.9
    assert Path(config.data.nuisance_report_path).is_file()
