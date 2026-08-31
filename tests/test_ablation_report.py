import json
from pathlib import Path

from aigc_recognizer.ablation_report import build_report


def _result(path: Path, clean: float, transformed: float) -> None:
    payload = {
        "checkpoint": {"sha256": "identity"},
        "calibration": {"applied": True},
        "scenarios": {
            "clean": {
                "auroc": clean,
                "balanced_accuracy": clean - 0.05,
                "real_recall": clean - 0.04,
            },
            "blur_2.0": {
                "auroc": transformed,
                "balanced_accuracy": transformed - 0.05,
                "real_recall": transformed - 0.10,
            },
            "combo_stress": {
                "auroc": transformed - 0.10,
                "balanced_accuracy": transformed - 0.15,
                "real_recall": transformed - 0.20,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_ablation_report_writes_machine_readable_and_svg_outputs(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    removal = tmp_path / "removal.json"
    _result(full, 0.95, 0.90)
    _result(removal, 0.93, 0.85)

    rows = build_report(
        [("Full", full), ("Without method", removal)], tmp_path / "report"
    )

    assert rows[1]["delta_mean_single_auroc"] < 0
    assert (tmp_path / "report" / "summary.json").is_file()
    assert (tmp_path / "report" / "summary.csv").is_file()
    svg = (tmp_path / "report" / "auroc_comparison.svg").read_text(encoding="utf-8")
    assert "RobustFake ablation AUROC" in svg
    assert "Without method" in svg
