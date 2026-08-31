import json
from pathlib import Path

import pytest
import torch

import numpy as np

from aigc_recognizer.calibration import _select_robust_threshold, load_global_calibrator
from aigc_recognizer.config import load_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_global_calibrator_applies_affine_logits_and_checks_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    import hashlib

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coefficient": 2.0,
                "intercept": -1.0,
                "threshold": 0.7,
                "checkpoint_sha256": checkpoint_sha256,
            }
        ),
        encoding="utf-8",
    )
    config = load_config(DEFAULT_CONFIG)
    config.evaluation.calibration_path = str(calibration)

    loaded = load_global_calibrator(config, checkpoint)

    assert loaded is not None
    expected = torch.sigmoid(torch.tensor([-1.0, 1.0]))
    assert torch.allclose(loaded.probabilities(torch.tensor([0.0, 1.0])), expected)
    assert loaded.threshold == pytest.approx(0.7)

    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(RuntimeError, match="different checkpoint"):
        load_global_calibrator(config, checkpoint)


def test_absent_automatic_calibration_is_optional(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = load_config(DEFAULT_CONFIG)

    assert load_global_calibrator(config, checkpoint) is None


def test_calibration_can_be_disabled_even_when_an_artifact_exists(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "calibration.json").write_text("not read", encoding="utf-8")
    config = load_config(DEFAULT_CONFIG)
    config.evaluation.calibration_enabled = False

    assert load_global_calibrator(config, checkpoint) is None


def test_robust_threshold_protects_clean_groups_and_improves_worst_group() -> None:
    probabilities = np.asarray(
        [
            0.10, 0.20, 0.60, 0.70,
            0.15, 0.25, 0.65, 0.75,
            0.40, 0.55, 0.70, 0.80,
            0.45, 0.60, 0.75, 0.85,
        ]
    )
    labels = np.tile(np.asarray([0, 0, 1, 1]), 4)
    groups = np.repeat(
        np.asarray(["val_id_clean", "val_dg_clean", "val_id_transformed", "val_dg_transformed"]),
        4,
    )

    threshold, diagnostics = _select_robust_threshold(
        probabilities, labels, groups, max_clean_drop=0.0
    )

    assert 0.25 < threshold <= 0.60
    assert diagnostics["strategy"] == "constrained_minimax"
    assert diagnostics["selected_clean_macro_balanced_accuracy"] == pytest.approx(1.0)
    assert set(diagnostics["selected_group_balanced_accuracy"]) == {
        "val_id_clean",
        "val_dg_clean",
        "val_id_transformed",
        "val_dg_transformed",
    }
