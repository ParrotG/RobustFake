"""Render high-frequency residual maps and an energy comparison chart."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from torchvision.transforms import functional as tvf

from aigc_recognizer.config import config_argument_parser, load_config
from aigc_recognizer.data.transforms import canonical_rgb
from aigc_recognizer.model import CLIP_MEAN, CLIP_STD, HighFrequencyResidualBranch


IMAGE_SIZE = 224
CELL_WIDTH = 260
CELL_HEIGHT = 270
MARGIN = 12
BACKGROUND = (247, 248, 250)
TEXT = (25, 29, 36)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _make_demo_image(size: int = IMAGE_SIZE) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            wave = int(18 * torch.sin(torch.tensor(x / 11.0)).item())
            pixels[x, y] = (
                min(255, 35 + x + wave),
                min(255, 55 + y // 2),
                min(255, 105 + (x + y) // 3),
            )
    draw = ImageDraw.Draw(image)
    for offset in range(-size, size, 16):
        draw.line((offset, 0, offset + size, size), fill=(235, 220, 145), width=2)
    draw.rectangle((28, 30, 95, 98), outline=(245, 245, 245), width=4)
    draw.ellipse((130, 32, 198, 100), fill=(224, 112, 95), outline=(255, 235, 220), width=3)
    draw.line((28, 164, 198, 164), fill=(250, 250, 250), width=4)
    draw.line((28, 178, 172, 205), fill=(25, 235, 180), width=3)
    draw.text((34, 112), "HF DEMO", fill=(250, 250, 250), font=_font(20))
    return image


def _load_image(path: Path | None) -> Image.Image:
    if path is None:
        return _make_demo_image()
    with Image.open(path) as source:
        return canonical_rgb(source, 127).resize(
            (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.LANCZOS
        )


def _jpeg_resize(image: Image.Image) -> Image.Image:
    reduced = image.resize((56, 56), resample=Image.Resampling.BILINEAR)
    restored = reduced.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    restored.save(buffer, format="JPEG", quality=45, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def _normalized_tensor(image: Image.Image) -> torch.Tensor:
    tensor = tvf.pil_to_tensor(image).float().div_(255.0)
    return tvf.normalize(tensor, CLIP_MEAN, CLIP_STD).unsqueeze(0)


def _residual_maps(
    image: Image.Image, branch: HighFrequencyResidualBranch
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    normalized = _normalized_tensor(image)
    with torch.no_grad():
        mean = normalized.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = normalized.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        rgb = (normalized * std + mean).clamp(0.0, 1.0)
        residuals = torch.nn.functional.conv2d(
            rgb, branch.high_pass_kernels.float(), groups=3
        )
        embedding = branch(normalized)
    maps = {
        "Laplacian": residuals[:, 0::3].abs().mean(dim=1).squeeze(0),
        "Horizontal": residuals[:, 1::3].abs().mean(dim=1).squeeze(0),
        "Vertical": residuals[:, 2::3].abs().mean(dim=1).squeeze(0),
        "Combined": residuals.abs().mean(dim=1).squeeze(0),
    }
    return maps, embedding.squeeze(0)


def _heatmap(values: torch.Tensor) -> Image.Image:
    values = values.float()
    scale = torch.quantile(values, 0.99).clamp_min(1e-6)
    values = (values / scale).clamp(0.0, 1.0)
    red = (255.0 * values).to(torch.uint8)
    green = (255.0 * (1.0 - (2.0 * values - 1.0).abs())).clamp(0, 255).to(torch.uint8)
    blue = (255.0 * (1.0 - values)).to(torch.uint8)
    rgb = torch.stack([red, green, blue], dim=-1).cpu().numpy()
    return Image.fromarray(rgb, mode="RGB")


def _cell(image: Image.Image, title: str) -> Image.Image:
    cell = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
    draw = ImageDraw.Draw(cell)
    draw.text((16, 8), title, fill=TEXT, font=_font(17))
    cell.paste(image.resize((IMAGE_SIZE, IMAGE_SIZE)), (18, 34))
    return cell


def _render_panel(
    clean: Image.Image,
    degraded: Image.Image,
    clean_maps: dict[str, torch.Tensor],
    degraded_maps: dict[str, torch.Tensor],
    destination: Path,
) -> None:
    columns = ["Input", "Laplacian", "Horizontal", "Vertical", "Combined"]
    panel = Image.new(
        "RGB",
        (MARGIN * 3 + CELL_WIDTH * len(columns), MARGIN * 3 + CELL_HEIGHT * 2),
        BACKGROUND,
    )
    for row, (source, maps, prefix) in enumerate(
        ((clean, clean_maps, "Clean"), (degraded, degraded_maps, "JPEG45 + resize"))
    ):
        images = [source] + [_heatmap(maps[name]) for name in columns[1:]]
        for column, (title, image) in enumerate(zip(columns, images)):
            label = f"{prefix} / {title}" if column == 0 else f"{prefix} / {title}"
            x = MARGIN + column * (CELL_WIDTH + MARGIN)
            y = MARGIN + row * (CELL_HEIGHT + MARGIN)
            panel.paste(_cell(image, label), (x, y))
    panel.save(destination, format="PNG")


def _render_energy_chart(
    clean_maps: dict[str, torch.Tensor],
    degraded_maps: dict[str, torch.Tensor],
    destination: Path,
) -> None:
    width, height = 900, 520
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    title_font = _font(24)
    label_font = _font(16)
    draw.text((55, 24), "Mean absolute high-frequency residual energy", fill=TEXT, font=title_font)
    names = list(clean_maps)
    clean_values = [float(values.abs().mean()) for values in clean_maps.values()]
    degraded_values = [float(values.abs().mean()) for values in degraded_maps.values()]
    maximum = max(clean_values + degraded_values) * 1.25
    left, top, right, bottom = 86, 85, 850, 420
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - round((bottom - top) * tick / 5)
        draw.line((left, y, right, y), fill=(225, 228, 233), width=1)
        draw.text((12, y - 8), f"{value:.3f}", fill=(80, 86, 95), font=label_font)
    group_width = (right - left) / len(names)
    bar_width = 42
    for index, name in enumerate(names):
        center = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width / 2 - 4, clean_values[index], (73, 113, 208)),
            (4, degraded_values[index], (229, 126, 73)),
        ):
            x0 = round(center + offset)
            x1 = x0 + bar_width
            y0 = bottom - round((bottom - top) * value / maximum)
            draw.rectangle((x0, y0, x1, bottom), fill=color)
            draw.text((x0, y0 - 22), f"{value:.3f}", fill=TEXT, font=label_font)
        draw.text((round(center - 42), bottom + 14), name, fill=TEXT, font=label_font)
    draw.rectangle((610, 48, 630, 68), fill=(73, 113, 208))
    draw.text((640, 47), "Clean", fill=TEXT, font=label_font)
    draw.rectangle((710, 48, 730, 68), fill=(229, 126, 73))
    draw.text((740, 47), "Degraded", fill=TEXT, font=label_font)
    draw.text((86, 462), "Energy = mean(|fixed-filter response|); higher means more local high-frequency activity.", fill=(80, 86, 95), font=label_font)
    chart.save(destination, format="PNG")


def build_parser() -> argparse.ArgumentParser:
    parser = config_argument_parser("Visualize the high-frequency residual branch.")
    parser.add_argument("--input", type=Path, default=None, help="Optional image to visualize.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/visualizations"),
        help="Directory for generated PNG charts.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    image = _load_image(args.input)
    degraded = _jpeg_resize(image)
    branch = HighFrequencyResidualBranch(config.model).eval()
    clean_maps, clean_embedding = _residual_maps(image, branch)
    degraded_maps, degraded_embedding = _residual_maps(degraded, branch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "high_frequency_residual_demo.png"
    chart_path = args.output_dir / "residual_energy_chart.png"
    _render_panel(image, degraded, clean_maps, degraded_maps, panel_path)
    _render_energy_chart(clean_maps, degraded_maps, chart_path)
    print(f"panel: {panel_path}")
    print(f"chart: {chart_path}")
    print(f"clean_embedding_norm: {clean_embedding.norm().item():.6f}")
    print(f"degraded_embedding_norm: {degraded_embedding.norm().item():.6f}")


if __name__ == "__main__":
    main()
