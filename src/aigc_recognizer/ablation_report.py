"""Build compact machine-readable and SVG summaries for RobustFake ablations."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


_CHART_METRICS = [
    ("clean_auroc", "Clean", "#2563EB"),
    ("mean_single_auroc", "Mean single", "#16A34A"),
    ("worst_single_auroc", "Worst single", "#F59E0B"),
    ("mean_composed_auroc", "Mean composed", "#8B5CF6"),
]


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


def _chart_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep ranking ablations in the AUROC chart and separate calibration."""
    ranking_rows = [row for row in rows if row["calibration_applied"]]
    return ranking_rows or rows


def _display_name(name: str) -> str:
    aliases = {
        "NoResidual": "No residual",
        "NoMultilayer": "No multi-layer",
        "NoConsistency": "No consistency",
        "NoContrastive": "No contrastive",
        "NoCalibration": "No calibration",
    }
    return aliases.get(name, name)


def _chart_scale(rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = [
        float(row[key])
        for row in rows
        for key, _label, _color in _CHART_METRICS
        if row[key] is not None
    ]
    lower = max(0.0, math.floor((min(values) - 0.01) / 0.05) * 0.05)
    return lower, 1.0


def _bar_svg(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write a vertical grouped AUROC chart suitable for slides."""
    rows = _chart_rows(rows)
    width = 1500
    height = 900
    left, right, top, bottom = 105, 45, 175, 190
    chart_width = width - left - right
    chart_height = height - top - bottom
    y_min, y_max = _chart_scale(rows)
    group_width = chart_width / len(rows)
    bar_width = min(42.0, group_width / (len(_CHART_METRICS) + 1.5))

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="105" y="54" font-family="sans-serif" font-size="32" '
        'font-weight="700" fill="#111827">Leave-one-component-out robustness</text>',
        f'<text x="105" y="86" font-family="sans-serif" font-size="17" fill="#4B5563">'
        f'Full official matrix · grouped by model variant · AUROC axis starts at {y_min:.2f}</text>',
    ]

    legend_x = 105
    for _key, label, color in _CHART_METRICS:
        elements.append(
            f'<rect x="{legend_x}" y="112" width="18" height="18" rx="3" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{legend_x + 27}" y="127" font-family="sans-serif" '
            f'font-size="15" fill="#374151">{html.escape(label)}</text>'
        )
        legend_x += 150

    tick = math.ceil(y_min / 0.05) * 0.05
    while tick <= y_max + 1e-9:
        y = y_position(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            'stroke="#D1D5DB" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 15}" y="{y + 6:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="15" fill="#4B5563">{tick:.2f}</text>'
        )
        tick += 0.05

    baseline = rows[0]
    for row_index, row in enumerate(rows):
        group_left = left + row_index * group_width
        center = group_left + group_width / 2
        if row_index == 0:
            elements.append(
                f'<rect x="{group_left + 8:.1f}" y="{top}" width="{group_width - 16:.1f}" '
                f'height="{chart_height}" fill="#EFF6FF" rx="8"/>'
            )
        bars_width = len(_CHART_METRICS) * bar_width
        bars_left = center - bars_width / 2
        for metric_index, (key, _label, color) in enumerate(_CHART_METRICS):
            value = row[key]
            if value is None:
                continue
            x = bars_left + metric_index * bar_width
            y = y_position(float(value))
            rendered_height = top + chart_height - y
            elements.append(
                f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{bar_width - 4:.1f}" '
                f'height="{rendered_height:.1f}" fill="{color}" rx="4"/>'
            )
            elements.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12" font-weight="600" '
                f'fill="#374151">{float(value):.3f}</text>'
            )
        label = html.escape(_display_name(str(row["name"])))
        elements.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 35}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="17" font-weight="600" fill="#111827">{label}</text>'
        )
        if row_index == 0:
            delta_label = "Reference"
        else:
            delta = 100.0 * (
                float(row["worst_single_auroc"])
                - float(baseline["worst_single_auroc"])
            )
            delta_label = f"Δ worst single {delta:+.1f} pp"
        elements.append(
            f'<text x="{center:.1f}" y="{top + chart_height + 62}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" fill="#6B7280">{delta_label}</text>'
        )

    elements.append(
        f'<text x="{left}" y="{height - 38}" font-family="sans-serif" font-size="14" '
        'fill="#6B7280">Calibration ablation is reported separately because monotonic calibration does not change AUROC ranking.</text>'
    )
    elements.append("</svg>")
    destination.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/"
        + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    )
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _bar_png(rows: list[dict[str, Any]], destination: Path) -> None:
    """Write a high-resolution PNG counterpart of the grouped SVG chart."""
    rows = _chart_rows(rows)
    width, height = 1800, 1080
    left, right, top, bottom = 130, 55, 215, 230
    chart_width = width - left - right
    chart_height = height - top - bottom
    y_min, y_max = _chart_scale(rows)
    group_width = chart_width / len(rows)
    bar_width = min(52.0, group_width / (len(_CHART_METRICS) + 1.5))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def y_position(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * chart_height

    draw.text((left, 50), "Leave-one-component-out robustness", font=_font(39, bold=True), fill="#111827")
    draw.text(
        (left, 105),
        f"Full official matrix · grouped by model variant · AUROC axis starts at {y_min:.2f}",
        font=_font(21),
        fill="#4B5563",
    )
    legend_x = left
    for _key, label, color in _CHART_METRICS:
        draw.rounded_rectangle((legend_x, 155, legend_x + 23, 178), radius=4, fill=color)
        draw.text((legend_x + 34, 151), label, font=_font(18), fill="#374151")
        legend_x += 190

    tick = math.ceil(y_min / 0.05) * 0.05
    while tick <= y_max + 1e-9:
        y = y_position(tick)
        draw.line((left, y, width - right, y), fill="#D1D5DB", width=2)
        label = f"{tick:.2f}"
        box = draw.textbbox((0, 0), label, font=_font(17))
        draw.text((left - 18 - (box[2] - box[0]), y - 10), label, font=_font(17), fill="#4B5563")
        tick += 0.05

    baseline = rows[0]
    for row_index, row in enumerate(rows):
        group_left = left + row_index * group_width
        center = group_left + group_width / 2
        if row_index == 0:
            draw.rounded_rectangle(
                (group_left + 10, top, group_left + group_width - 10, top + chart_height),
                radius=10,
                fill="#EFF6FF",
            )
        bars_width = len(_CHART_METRICS) * bar_width
        bars_left = center - bars_width / 2
        for metric_index, (key, _label, color) in enumerate(_CHART_METRICS):
            value = row[key]
            if value is None:
                continue
            x = bars_left + metric_index * bar_width
            y = y_position(float(value))
            draw.rounded_rectangle(
                (x + 3, y, x + bar_width - 3, top + chart_height),
                radius=5,
                fill=color,
            )
            value_label = f"{float(value):.3f}"
            box = draw.textbbox((0, 0), value_label, font=_font(14, bold=True))
            draw.text(
                (x + bar_width / 2 - (box[2] - box[0]) / 2, y - 23),
                value_label,
                font=_font(14, bold=True),
                fill="#374151",
            )
        display_name = _display_name(str(row["name"]))
        box = draw.textbbox((0, 0), display_name, font=_font(20, bold=True))
        draw.text(
            (center - (box[2] - box[0]) / 2, top + chart_height + 35),
            display_name,
            font=_font(20, bold=True),
            fill="#111827",
        )
        delta_label = "Reference" if row_index == 0 else (
            "Δ worst single "
            f"{100.0 * (float(row['worst_single_auroc']) - float(baseline['worst_single_auroc'])):+.1f} pp"
        )
        box = draw.textbbox((0, 0), delta_label, font=_font(16))
        draw.text(
            (center - (box[2] - box[0]) / 2, top + chart_height + 72),
            delta_label,
            font=_font(16),
            fill="#6B7280",
        )
    draw.text(
        (left, height - 48),
        "Calibration ablation is reported separately because monotonic calibration does not change AUROC ranking.",
        font=_font(16),
        fill="#6B7280",
    )
    image.save(destination, format="PNG", optimize=True)


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
    _bar_png(rows, output_directory / "auroc_comparison.png")
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
