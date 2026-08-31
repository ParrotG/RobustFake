"""Build compact machine-readable and SVG summaries for RobustFake ablations."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Result must use NAME=PATH syntax.")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Result name and path must not be empty.")
    return name, Path(raw_path)


def _summary(name: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", {})
    if "clean" not in scenarios:
        raise RuntimeError(f"Ablation result does not contain a clean scenario: {path}")
    composed_names = set(
        name for name in scenarios if str(name).startswith("combo_")
    )
    single_names = [
        item for item in scenarios if item != "clean" and item not in composed_names
    ]
    if not single_names:
        raise RuntimeError(f"Ablation result does not contain transformed scenarios: {path}")
    clean = scenarios["clean"]
    singles = [scenarios[item] for item in single_names]
    composed = [scenarios[item] for item in composed_names]
    return {
        "name": name,
        "path": str(path),
        "checkpoint_sha256": payload.get("checkpoint", {}).get("sha256"),
        "calibration_applied": bool(payload.get("calibration", {}).get("applied")),
        "clean_auroc": float(clean["auroc"]),
        "mean_single_auroc": mean(float(item["auroc"]) for item in singles),
        "worst_single_auroc": min(float(item["auroc"]) for item in singles),
        "mean_composed_auroc": (
            mean(float(item["auroc"]) for item in composed) if composed else None
        ),
        "clean_balanced_accuracy": float(clean["balanced_accuracy"]),
        "mean_single_balanced_accuracy": mean(
            float(item["balanced_accuracy"]) for item in singles
        ),
        "worst_single_real_recall": min(float(item["real_recall"]) for item in singles),
    }


def _bar_svg(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write a dependency-free grouped AUROC bar chart suitable for slides."""
    metrics = [
        ("clean_auroc", "Clean"),
        ("mean_single_auroc", "Mean transformed"),
        ("worst_single_auroc", "Worst transformed"),
        ("mean_composed_auroc", "Mean composed"),
    ]
    metrics = [item for item in metrics if any(row[item[0]] is not None for row in rows)]
    width = 960
    left = 190
    top = 70
    bar_height = max(12, min(24, 60 // max(1, len(rows))))
    group_height = 34 + len(rows) * bar_height
    height = top + group_height * len(metrics) + 70
    chart_width = width - left - 70
    colors = ["#2563EB", "#F97316", "#16A34A", "#9333EA", "#DC2626", "#0891B2"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="36" font-family="sans-serif" font-size="24" '
        'font-weight="700">RobustFake ablation AUROC</text>',
    ]
    for tick in range(0, 11):
        value = tick / 10
        x = left + value * chart_width
        elements.append(
            f'<line x1="{x:.1f}" y1="52" x2="{x:.1f}" y2="{height - 45}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{height - 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#4B5563">{value:.1f}</text>'
        )
    for metric_index, (key, label) in enumerate(metrics):
        base_y = top + metric_index * group_height
        elements.append(
            f'<text x="{left - 14}" y="{base_y + 18}" text-anchor="end" '
            f'font-family="sans-serif" font-size="15" font-weight="600">{html.escape(label)}</text>'
        )
        for row_index, row in enumerate(rows):
            value = row[key]
            if value is None:
                continue
            y = base_y + 26 + row_index * bar_height
            bar_width = float(value) * chart_width
            color = colors[row_index % len(colors)]
            elements.append(
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_height - 3}" '
                f'fill="{color}" rx="2"/>'
            )
            elements.append(
                f'<text x="{left + bar_width + 7:.1f}" y="{y + bar_height - 7}" '
                f'font-family="sans-serif" font-size="12">{float(value):.4f} '
                f'{html.escape(str(row["name"]))}</text>'
            )
    elements.append("</svg>")
    destination.write_text("\n".join(elements) + "\n", encoding="utf-8")


def build_report(results: list[tuple[str, Path]], output_directory: Path) -> list[dict[str, Any]]:
    """Create JSON, CSV, and SVG summaries from evaluator result files."""
    if len(results) < 2:
        raise ValueError("Ablation reporting requires a full baseline and at least one removal.")
    rows = [_summary(name, path) for name, path in results]
    baseline = rows[0]
    for row in rows:
        row["delta_mean_single_auroc"] = row["mean_single_auroc"] - baseline["mean_single_auroc"]
        row["delta_worst_single_auroc"] = row["worst_single_auroc"] - baseline["worst_single_auroc"]
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _bar_svg(rows, output_directory / "auroc_comparison.svg")
    return rows


def main() -> None:
    """Build presentation-ready ablation summaries from evaluation JSON files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=_parse_result,
        metavar="NAME=PATH",
        help="Evaluation result; pass the full model first and repeat for removals.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    build_report(arguments.result, arguments.output_dir)


if __name__ == "__main__":
    main()
