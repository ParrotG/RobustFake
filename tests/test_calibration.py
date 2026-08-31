import json
from pathlib import Path

import pytest
import torch

from aigc_recognizer.calibration import load_global_calibrator
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
