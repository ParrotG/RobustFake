"""Validate baseline results and build a presentation-ready AUROC comparison."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


_SUMMARY_METRICS = (
    ("clean_auroc", "Clean"),
    ("mean_single_transform_auroc", "Mean prescribed transform"),
    ("worst_single_transform_auroc", "Worst prescribed transform"),
)

_SCENARIOS = (
    ("clean", "Clean"),
    ("jpeg_90", "JPEG 90"),
    ("jpeg_70", "JPEG 70"),
    ("jpeg_50", "JPEG 50"),
    ("jpeg_30", "JPEG 30"),
    ("blur_0.5", "Blur .5"),
    ("blur_1.0", "Blur 1"),
    ("blur_2.0", "Blur 2"),
    ("resize_0.5", "Resize .5"),
    ("resize_0.25", "Resize .25"),
    ("noise_0.02", "Noise .02"),
    ("noise_0.05", "Noise .05"),
    ("noise_0.10", "Noise .10"),
    ("color_jitter_0.20", "Color"),
    ("center_crop_0.80", "Crop .8"),
)

_COLORS = ("#0F766E", "#2563EB", "#EA580C", "#7C3AED")


def _parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Result must use NAME=PATH syntax.")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Result name and path must not be empty.")
    return name, Path(raw_path)


def _load_result(name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "full":
        raise RuntimeError(f"Comparison requires a full evaluation result: {path}")
    scenarios = payload.get("scenarios", {})
    missing = [scenario for scenario, _label in _SCENARIOS if scenario not in scenarios]
    if missing:
        raise RuntimeError(f"Result is missing prescribed scenarios {missing}: {path}")
    count = int(payload.get("evaluated_sample_count", 0))
    if count <= 0 or any(int(scenarios[key].get("count", -1)) != count for key, _ in _SCENARIOS):
        raise RuntimeError(f"Result has inconsistent scenario sample counts: {path}")
    summary = payload.get("summary", {})
    row: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "dataset": payload.get("dataset", {}).get("name"),
        "evaluated_sample_count": count,
    }
    for key, _label in _SUMMARY_METRICS:
        value = float(summary[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"Result contains an invalid {key}: {path}")
        row[key] = value
    row["scenarios"] = {
        key: float(scenarios[key]["auroc"]) for key, _label in _SCENARIOS
    }
    return row


def _validate_comparability(rows: list[dict[str, Any]]) -> None:
    if len(rows) < 2:
        raise ValueError("Baseline reporting requires the full model and at least one baseline.")
    datasets = {row["dataset"] for row in rows}
    counts = {row["evaluated_sample_count"] for row in rows}
    if len(datasets) != 1 or None in datasets:
        raise RuntimeError(f"Results use different or unidentified datasets: {datasets}")
    if len(counts) != 1:
        raise RuntimeError(f"Results use different sample counts: {counts}")


def _write_svg(rows: list[dict[str, Any]], destination: Path) -> None:
    width, height = 1600, 1000
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="70" y="58" font-family="sans-serif" font-size="34" font-weight="700" fill="#111827">RobustFake retains its ranking advantage after redistribution</text>',
        '<text x="70" y="91" font-family="sans-serif" font-size="17" fill="#4B5563">WildFake official demonstration set · 13,841 images · complete prescribed single-transform matrix · AUROC</text>',
    ]

    legend_x = 70
    for index, row in enumerate(rows):
        color = _COLORS[index]
        elements.extend(
            [
                f'<circle cx="{legend_x + 8}" cy="127" r="8" fill="{color}"/>',
                f'<text x="{legend_x + 24}" y="133" font-family="sans-serif" font-size="16" font-weight="600" fill="#374151">{html.escape(str(row["name"]))}</text>',
            ]
        )
        legend_x += 210

    elements.append('<text x="70" y="178" font-family="sans-serif" font-size="21" font-weight="700" fill="#111827">A. Aggregate comparison</text>')
    card_width = 470
    for metric_index, (metric, label) in enumerate(_SUMMARY_METRICS):
        x = 70 + metric_index * 500
        elements.extend(
            [
                f'<rect x="{x}" y="198" width="{card_width}" height="260" rx="14" fill="#F8FAFC" stroke="#E2E8F0"/>',
                f'<text x="{x + 22}" y="232" font-family="sans-serif" font-size="18" font-weight="700" fill="#1F2937">{html.escape(label)}</text>',
            ]
        )
        for row_index, row in enumerate(rows):
            y = 270 + row_index * 62
            value = float(row[metric])
            bar_x, bar_width = x + 145, 255
            color = _COLORS[row_index]
            elements.extend(
                [
                    f'<text x="{x + 22}" y="{y + 17}" font-family="sans-serif" font-size="15" fill="#374151">{html.escape(str(row["name"]))}</text>',
                    f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="22" rx="6" fill="#E5E7EB"/>',
                    f'<rect x="{bar_x}" y="{y}" width="{bar_width * value:.1f}" height="22" rx="6" fill="{color}"/>',
                    f'<text x="{x + 450}" y="{y + 17}" text-anchor="end" font-family="sans-serif" font-size="16" font-weight="700" fill="#111827">{value:.4f}</text>',
                ]
            )

    elements.extend(
        [
            '<text x="70" y="510" font-family="sans-serif" font-size="21" font-weight="700" fill="#111827">B. Every prescribed scenario</text>',
            '<text x="1530" y="510" text-anchor="end" font-family="sans-serif" font-size="14" fill="#6B7280">Axis starts at 0.35 for readability</text>',
        ]
    )
    left, right, top, bottom = 105.0, 1530.0, 545.0, 880.0
    y_min, y_max = 0.35, 1.0

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (bottom - top)

    for tick in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        y = y_position(tick)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#D1D5DB" stroke-width="1"/>',
                f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="14" fill="#6B7280">{tick:.1f}</text>',
            ]
        )
    step = (right - left) / (len(_SCENARIOS) - 1)
    for scenario_index, (_key, label) in enumerate(_SCENARIOS):
        x = left + scenario_index * step
        elements.append(
            f'<text x="{x:.1f}" y="{bottom + 31}" transform="rotate(35 {x:.1f} {bottom + 31})" text-anchor="start" font-family="sans-serif" font-size="13" fill="#4B5563">{html.escape(label)}</text>'
        )
    for row_index, row in enumerate(rows):
        color = _COLORS[row_index]
        points = [
            (left + index * step, y_position(float(row["scenarios"][key])))
            for index, (key, _label) in enumerate(_SCENARIOS)
        ]
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{5 if row_index == 0 else 3}" stroke-linejoin="round" stroke-linecap="round" opacity="{1.0 if row_index == 0 else 0.9}"/>'
        )
        for x, y in points:
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if row_index == 0 else 4}" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>'
            )
    elements.extend(
        [
            '<text x="70" y="972" font-family="sans-serif" font-size="13" fill="#6B7280">Fairness note: all detectors use the same images and transformations. Public baselines retain their official 224 px preprocessing and uncalibrated heads; AUROC is threshold-independent.</text>',
            '</svg>',
        ]
    )
    destination.write_text("\n".join(elements) + "\n", encoding="utf-8")


def build_report(results: list[tuple[str, Path]], output_directory: Path) -> list[dict[str, Any]]:
    """Validate comparable results and write JSON, CSV, and SVG summaries."""
    rows = [_load_result(name, path) for name, path in results]
    _validate_comparability(rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    serializable = [{key: value for key, value in row.items() if key != "scenarios"} for row in rows]
    (output_directory / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)
    _write_svg(rows, output_directory / "robustness_comparison.svg")
    return rows


def main() -> None:
    """Build the full-model versus public-baseline comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=_parse_result,
        metavar="NAME=PATH",
        help="Full evaluation result; pass RobustFake first, then repeat for baselines.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    build_report(arguments.result, arguments.output_dir)


if __name__ == "__main__":
    main()
