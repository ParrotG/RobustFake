"""Visualize RGBA channels and heuristic suspicious regions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


CHANNELS = ("R", "G", "B", "A")
BACKGROUND = (247, 248, 250)
TEXT = (25, 29, 36)
CELL_WIDTH = 260
CELL_HEIGHT = 280
MARGIN = 12


@dataclass(frozen=True)
class ChannelAnalysis:
    name: str
    channel: np.ndarray
    residual: np.ndarray
    suspicious: np.ndarray
    threshold: float
    note: str


def _font(size: int) -> ImageFont.ImageFont:
    font_candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    )
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_rgba(path: Path) -> tuple[Image.Image, bool]:
    """Load an image as RGBA while reporting whether alpha was present."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        had_alpha = "A" in image.getbands() or "transparency" in image.info
        return image.convert("RGBA").copy(), had_alpha


def _dilate(mask: np.ndarray, size: int = 5) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("Dilation size must be a positive odd number.")
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0


def _anomaly_threshold(residual: np.ndarray, quantile: float) -> float:
    maximum = float(residual.max())
    if maximum <= 0:
        return 0.0
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    robust = median + 4.0 * 1.4826 * mad
    percentile = float(np.quantile(residual, quantile))
    return max(robust, percentile)


def analyze_channel(
    channel: np.ndarray,
    name: str,
    *,
    blur_radius: float = 3.0,
    quantile: float = 0.985,
    dilation_size: int = 5,
) -> ChannelAnalysis:
    """Find channel-specific local anomalies with a high-pass heuristic."""
    values = np.asarray(channel, dtype=np.uint8)
    blurred = np.asarray(
        Image.fromarray(values, mode="L").filter(ImageFilter.GaussianBlur(blur_radius)),
        dtype=np.float32,
    )
    residual = np.abs(values.astype(np.float32) - blurred)
    threshold = _anomaly_threshold(residual, quantile)
    if name == "A":
        suspicious = (values < 250) | (residual >= threshold if threshold > 0 else False)
        note = "non-opaque pixels and abrupt alpha transitions"
    else:
        suspicious = residual >= threshold if threshold > 0 else np.zeros_like(values, dtype=bool)
        note = "unusually strong local high-frequency response"
    suspicious = _dilate(suspicious, dilation_size)
    return ChannelAnalysis(name, values, residual, suspicious, threshold, note)


def _channel_image(channel: np.ndarray) -> Image.Image:
    return Image.fromarray(channel.astype(np.uint8), mode="L").convert("RGB")


def _suspicious_image(analysis: ChannelAnalysis) -> Image.Image:
    base = np.asarray(_channel_image(analysis.channel), dtype=np.float32)
    mask = analysis.suspicious
    highlight = np.zeros_like(base)
    highlight[..., 0] = 255.0
    blended = base * 0.58 + highlight * 0.42
    blended[~mask] = base[~mask]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _cell(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    cell = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(cell)
    draw.text((16, 8), title, fill=TEXT, font=_font(18))
    if subtitle:
        draw.text((16, 30), subtitle, fill=(90, 97, 108), font=_font(13))
    cell.paste(image.resize((224, 224), resample=Image.Resampling.NEAREST), (18, 50))
    return cell


def render_channel_grid(
    analyses: list[ChannelAnalysis],
    destination: Path,
) -> None:
    """Render channel images on top and suspicious overlays below."""
    width = MARGIN * 3 + CELL_WIDTH * 4
    height = MARGIN * 3 + CELL_HEIGHT * 2
    panel = Image.new("RGB", (width, height), BACKGROUND)
    for index, analysis in enumerate(analyses):
        x = MARGIN + index * (CELL_WIDTH + MARGIN)
        panel.paste(_cell(_channel_image(analysis.channel), f"{analysis.name} channel"), (x, MARGIN))
        fraction = 100.0 * float(analysis.suspicious.mean())
        subtitle = f"suspicious: {fraction:.2f}%"
        panel.paste(
            _cell(_suspicious_image(analysis), f"{analysis.name} suspicious", subtitle),
            (x, MARGIN * 2 + CELL_HEIGHT),
        )
    panel.save(destination, format="PNG")


def render_residual_chart(analyses: list[ChannelAnalysis], destination: Path) -> None:
    """Render mean and high-percentile residual values for all channels."""
    width, height = 900, 520
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    draw.text((55, 24), "RGBA channel residual summary", fill=TEXT, font=_font(24))
    means = [float(analysis.residual.mean()) for analysis in analyses]
    p99s = [float(np.quantile(analysis.residual, 0.99)) for analysis in analyses]
    maximum = max(means + p99s + [1.0]) * 1.2
    left, top, right, bottom = 86, 85, 850, 420
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - round((bottom - top) * tick / 5)
        draw.line((left, y, right, y), fill=(225, 228, 233), width=1)
        draw.text((18, y - 8), f"{value:.1f}", fill=(80, 86, 95), font=_font(16))
    group_width = (right - left) / len(analyses)
    bar_width = 42
    for index, analysis in enumerate(analyses):
        center = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width / 2 - 4, means[index], (73, 113, 208)),
            (4, p99s[index], (229, 126, 73)),
        ):
            x0 = round(center + offset)
            x1 = x0 + bar_width
            y0 = bottom - round((bottom - top) * value / maximum)
            draw.rectangle((x0, y0, x1, bottom), fill=color)
            draw.text((x0, y0 - 22), f"{value:.1f}", fill=TEXT, font=_font(15))
        draw.text((round(center - 10), bottom + 14), analysis.name, fill=TEXT, font=_font(17))
    draw.rectangle((610, 48, 630, 68), fill=(73, 113, 208))
    draw.text((640, 47), "Mean |residual|", fill=TEXT, font=_font(16))
    draw.rectangle((760, 48, 780, 68), fill=(229, 126, 73))
    draw.text((790, 47), "P99", fill=TEXT, font=_font(16))
    draw.text(
        (86, 462),
        "RGB: local detail response; A: transparency and alpha-edge response.",
        fill=(80, 86, 95),
        font=_font(16),
    )
    chart.save(destination, format="PNG")


def build_report(
    image: Image.Image,
    had_alpha: bool,
    analyses: list[ChannelAnalysis],
    input_path: Path,
) -> dict[str, object]:
    """Build machine-readable channel statistics and interpretation notes."""
    channels: dict[str, object] = {}
    for analysis in analyses:
        values = analysis.channel.astype(np.float32)
        channels[analysis.name] = {
            "min": int(values.min()),
            "max": int(values.max()),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "residual_mean": round(float(analysis.residual.mean()), 4),
            "residual_p99": round(float(np.quantile(analysis.residual, 0.99)), 4),
            "threshold": round(float(analysis.threshold), 4),
            "suspicious_pixel_fraction": round(float(analysis.suspicious.mean()), 6),
            "heuristic": analysis.note,
        }
    return {
        "input": str(input_path),
        "size": {"width": image.width, "height": image.height},
        "had_alpha_channel": had_alpha,
        "alpha_note": (
            "The source image had no alpha channel; A is synthesized as 255."
            if not had_alpha
            else "A channel is read from the source image."
        ),
        "channels": channels,
        "warning": "Suspicious regions are heuristic visual cues, not proof of AI generation or tampering.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render RGBA channel images and heuristic suspicious-region overlays."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rgba_visualizations"),
        help="Directory for generated PNG and JSON files.",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=3.0,
        help="Gaussian radius used for local residual extraction.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.985,
        help="Residual quantile used for RGB anomaly thresholding.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input image does not exist: {args.input}")
    if not 0.9 < args.quantile < 1.0:
        parser.error("--quantile must be between 0.9 and 1.0.")
    image, had_alpha = load_rgba(args.input)
    array = np.asarray(image, dtype=np.uint8)
    analyses = [
        analyze_channel(
            array[:, :, index],
            name,
            blur_radius=args.blur_radius,
            quantile=args.quantile,
        )
        for index, name in enumerate(CHANNELS)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_dir / "rgba_channel_analysis.png"
    chart_path = args.output_dir / "rgba_residual_chart.png"
    report_path = args.output_dir / "rgba_channel_report.json"
    render_channel_grid(analyses, grid_path)
    render_residual_chart(analyses, chart_path)
    report_path.write_text(
        json.dumps(build_report(image, had_alpha, analyses, args.input), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"analysis_panel: {grid_path}")
    print(f"residual_chart: {chart_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
