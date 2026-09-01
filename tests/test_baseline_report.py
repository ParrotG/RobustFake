import json
from pathlib import Path

import pytest

from aigc_recognizer.baseline_report import _SCENARIOS, build_report


def _result(path: Path, *, dataset: str = "official", count: int = 4, offset: float = 0.0) -> Path:
    scenarios = {
        name: {
            "auroc": 0.8 + offset,
            "count": count,
        }
        for name, _label in _SCENARIOS
    }
    payload = {
        "mode": "full",
        "dataset": {"name": dataset},
        "evaluated_sample_count": count,
        "scenarios": scenarios,
        "summary": {
            "clean_auroc": 0.9 + offset,
            "mean_single_transform_auroc": 0.8 + offset,
            "worst_single_transform_auroc": 0.7 + offset,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_report_writes_validated_machine_and_visual_outputs(tmp_path: Path) -> None:
    full = _result(tmp_path / "full.json", offset=0.05)
    baseline = _result(tmp_path / "baseline.json")

    rows = build_report(
        [("RobustFake", full), ("Baseline", baseline)], tmp_path / "report"
    )

    assert [row["name"] for row in rows] == ["RobustFake", "Baseline"]
    assert (tmp_path / "report" / "summary.json").is_file()
    assert (tmp_path / "report" / "summary.csv").is_file()
    svg = (tmp_path / "report" / "robustness_comparison.svg").read_text(
        encoding="utf-8"
    )
    assert "RobustFake retains its ranking advantage" in svg
    assert "0.9500" in svg


def test_build_report_rejects_different_datasets(tmp_path: Path) -> None:
    full = _result(tmp_path / "full.json", dataset="official")
    baseline = _result(tmp_path / "baseline.json", dataset="other")

    with pytest.raises(RuntimeError, match="different or unidentified datasets"):
        build_report(
            [("RobustFake", full), ("Baseline", baseline)], tmp_path / "report"
        )
