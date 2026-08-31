"""Detect common redistribution degradations and apply conservative restoration.

The detector reuses the fixed RGB Laplacian/Sobel kernels from the original
high-frequency residual branch.  It is an analysis and preprocessing utility;
it is intentionally separate from the CLIP detector and training pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from .model import CLIP_MEAN, CLIP_STD, _make_high_pass_kernels


@dataclass(frozen=True)
class ArtifactEvidence:
    detected: bool
    confidence: float
    rationale: str


@dataclass(frozen=True)
class DegradationReport:
    input: str
    size: tuple[int, int]
    metrics: dict[str, float]
    artifacts: dict[str, ArtifactEvidence]
    operations: list[str]
    applied_operations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["size"] = {"width": self.size[0], "height": self.size[1]}
        return payload


@dataclass(frozen=True)
class GaussianSharpenParameters:
    radius: float
    percent: int
    threshold: int
    residual_gain: float
    clipping_increase: float


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _fixed_residuals(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RGB residual maps using the same fixed filters as the xyl branch."""
    import torch
    from torch.nn import functional as F

    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = tensor.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    kernels = _make_high_pass_kernels().float()
    with torch.no_grad():
        response = F.conv2d(normalized * std + mean, kernels, groups=3)
    values = response.squeeze(0).numpy()
    return values[0::3].mean(axis=0), values[1::3].mean(axis=0), values[2::3].mean(axis=0)


def _gradient_metrics(rgb: np.ndarray) -> tuple[float, float, float, float]:
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    padded = np.pad(gray, 1, mode="reflect")
    gx = padded[1:-1, 2:] - padded[1:-1, :-2]
    gy = padded[2:, 1:-1] - padded[:-2, 1:-1]
    gradient = np.hypot(gx, gy)
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * gray
    )
    edge_density = float((gradient > np.quantile(gradient, 0.85)).mean())
    return float(np.mean(gradient)), float(np.var(laplacian)), float(np.mean(np.abs(laplacian))), edge_density


def analyze_degradation(image: Image.Image, *, input_name: str = "<image>") -> DegradationReport:
    """Estimate blur, noise, resize and colour-jitter evidence from fixed residuals."""
    rgb = _to_rgb_array(image)
    laplacian, horizontal, vertical = _fixed_residuals(rgb)
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    smooth = np.asarray(Image.fromarray(np.clip(gray * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
    local_residual = gray - smooth
    gradient_mean, laplacian_variance, laplacian_abs_mean, edge_density = _gradient_metrics(rgb)
    hf_energy = float(np.mean(np.abs(np.stack([laplacian, horizontal, vertical]))))
    residual_sigma = float(np.median(np.abs(local_residual - np.median(local_residual))) / 0.6745)
    channel_means = rgb.reshape(-1, 3).mean(axis=0)
    channel_stds = rgb.reshape(-1, 3).std(axis=0)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    clipping = float(((rgb <= 0.01) | (rgb >= 0.99)).mean())
    channel_imbalance = float(np.std(channel_means) + 0.5 * np.std(channel_stds))
    periodicity = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])) / (gradient_mean + 1e-6))

    # Scores are intentionally conservative and expose the raw metrics in the report.
    blur_score = max(0.0, min(1.0, 0.58 * (1.0 - laplacian_variance / (laplacian_variance + 0.0035)) + 0.42 * (1.0 - hf_energy / (hf_energy + 0.018))))
    noise_score = max(0.0, min(1.0, _sigmoid((residual_sigma - 0.018) / 0.006) * (0.55 + 0.45 * (1.0 - min(edge_density / 0.18, 1.0)))))
    resize_score = max(0.0, min(1.0, _sigmoid((periodicity - 0.72) / 0.08) * (0.45 + 0.55 * min(gradient_mean / 0.16, 1.0))))
    color_score = max(0.0, min(1.0, _sigmoid((channel_imbalance - 0.035) / 0.012) * (0.55 + 0.45 * min(float(saturation.mean()) / 0.45, 1.0)) + 0.15 * min(clipping / 0.12, 1.0)))

    artifacts = {
        "gaussian_blur": ArtifactEvidence(blur_score >= 0.58, round(blur_score, 4), "low fixed-filter energy or Laplacian variance suggests softened edges"),
        "gaussian_noise": ArtifactEvidence(noise_score >= 0.58, round(noise_score, 4), "robust local residual sigma is elevated relative to edge density"),
        "resize": ArtifactEvidence(resize_score >= 0.58, round(resize_score, 4), "gradient periodicity is compatible with resampling/ringing"),
        "color_jitter": ArtifactEvidence(color_score >= 0.58, round(color_score, 4), "channel balance, saturation and clipping suggest a colour shift"),
    }
    operations = [name for name, evidence in artifacts.items() if evidence.detected]
    metrics = {
        "hf_energy": round(hf_energy, 8),
        "laplacian_variance": round(laplacian_variance, 8),
        "laplacian_abs_mean": round(laplacian_abs_mean, 8),
        "gradient_mean": round(gradient_mean, 8),
        "edge_density": round(edge_density, 8),
        "residual_sigma": round(residual_sigma, 8),
        "resize_periodicity": round(periodicity, 8),
        "channel_imbalance": round(channel_imbalance, 8),
        "mean_saturation": round(float(saturation.mean()), 8),
        "clipping_fraction": round(clipping, 8),
        "laplacian_residual_mean": round(float(np.mean(np.abs(laplacian))), 8),
    }
    return DegradationReport(input_name, image.size, metrics, artifacts, operations)


def _gray_world(image: Image.Image, strength: float = 0.45) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    means = array.reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    gains = target / np.maximum(means, 1e-3)
    gains = 1.0 + strength * (gains - 1.0)
    corrected = np.clip(array * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _high_frequency_energy(image: Image.Image) -> float:
    maps = _fixed_residuals(_to_rgb_array(image))
    return float(np.mean(np.abs(np.stack(maps))))


def _clipping_fraction(image: Image.Image) -> float:
    values = _to_rgb_array(image)
    return float(((values <= 0.01) | (values >= 0.99)).mean())


def _adaptive_gaussian_unsharp(
    image: Image.Image,
    confidence: float,
) -> tuple[Image.Image, GaussianSharpenParameters]:
    """Tune Gaussian unsharp parameters against the fixed residual response.

    Candidate selection rewards a controlled increase in Laplacian/Sobel
    energy and penalizes clipped pixels and excessive high-frequency gain.
    This makes the sharpening strength image-dependent while avoiding the
    unstable halos produced by blindly maximizing sharpness.
    """
    confidence = max(0.0, min(1.0, confidence))
    baseline_energy = max(_high_frequency_energy(image), 1e-8)
    baseline_clipping = _clipping_fraction(image)
    target_gain = 1.18 + 0.72 * confidence
    radii = sorted(
        {
            round(0.55 + 0.45 * confidence, 2),
            round(0.90 + 0.85 * confidence, 2),
            round(1.30 + 1.20 * confidence, 2),
        }
    )
    percents = sorted(
        {
            int(round(70 + 55 * confidence)),
            int(round(105 + 95 * confidence)),
            int(round(145 + 135 * confidence)),
        }
    )
    thresholds = (2, 4, 7)
    best_image = image.copy()
    best_parameters = GaussianSharpenParameters(0.0, 0, 0, 1.0, 0.0)
    best_score = -float("inf")
    for radius in radii:
        for percent in percents:
            for threshold in thresholds:
                candidate = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=radius,
                        percent=percent,
                        threshold=threshold,
                    )
                )
                gain = _high_frequency_energy(candidate) / baseline_energy
                clipping_increase = max(0.0, _clipping_fraction(candidate) - baseline_clipping)
                useful_gain = min(gain, target_gain) / target_gain
                overshoot = max(0.0, gain - target_gain)
                score = useful_gain - 0.55 * overshoot - 5.0 * clipping_increase
                # Prefer lower Gaussian strength when two candidates are
                # effectively equivalent, reducing ringing on broad edges.
                score -= 0.002 * radius + 0.00001 * percent
                if score > best_score:
                    best_score = score
                    best_image = candidate
                    best_parameters = GaussianSharpenParameters(
                        radius,
                        percent,
                        threshold,
                        round(gain, 4),
                        round(clipping_increase, 6),
                    )
    return best_image, best_parameters


def restore_image(image: Image.Image, report: DegradationReport | None = None) -> tuple[Image.Image, list[str]]:
    """Apply conservative, reversible restoration based on detected evidence."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    report = report or analyze_degradation(image)
    restored = image
    applied: list[str] = []
    if report.artifacts["gaussian_noise"].detected:
        try:
            import cv2

            bgr = cv2.cvtColor(np.asarray(restored), cv2.COLOR_RGB2BGR)
            h = int(round(3 + 8 * report.artifacts["gaussian_noise"].confidence))
            denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, 7, 21)
            restored = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        except ImportError:
            restored = restored.filter(ImageFilter.MedianFilter(size=3))
        applied.append("denoise")
    if report.artifacts["gaussian_blur"].detected or report.artifacts["resize"].detected:
        confidence = max(
            report.artifacts["gaussian_blur"].confidence,
            report.artifacts["resize"].confidence,
        )
        restored, parameters = _adaptive_gaussian_unsharp(restored, confidence)
        # A mild contrast lift recovers edge separation after resize without changing dimensions.
        restored = ImageEnhance.Contrast(restored).enhance(1.0 + 0.06 * confidence)
        applied.append(
            "gaussian_unsharp("
            f"radius={parameters.radius:.2f},"
            f"percent={parameters.percent},"
            f"threshold={parameters.threshold},"
            f"residual_gain={parameters.residual_gain:.4f}"
            ")"
        )
    if report.artifacts["color_jitter"].detected:
        restored = _gray_world(restored)
        restored = ImageEnhance.Color(restored).enhance(0.94)
        restored = ImageEnhance.Contrast(restored).enhance(1.03)
        applied.append("color_balance")
    if not applied:
        # Keep output deterministic while making the no-op explicit.
        restored = restored.copy()
        applied.append("no_change")
    return restored, applied


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _residual_heatmap(image: Image.Image) -> Image.Image:
    rgb = _to_rgb_array(image)
    maps = _fixed_residuals(rgb)
    values = np.mean(np.abs(np.stack(maps)), axis=0)
    scale = max(float(np.quantile(values, 0.99)), 1e-6)
    values = np.clip(values / scale, 0, 1)
    heat = np.stack([values, 1.0 - np.abs(2 * values - 1.0), 1.0 - values], axis=-1)
    return Image.fromarray(np.clip(heat * 255, 0, 255).astype(np.uint8), mode="RGB")


def render_example(original: Image.Image, restored: Image.Image, report: DegradationReport, destination: Path) -> None:
    width, height = 420, 340
    panel = Image.new("RGB", (width * 3, height), (247, 248, 250))
    titles = ["Input", "Fixed residual", "Restored"]
    images = [original, _residual_heatmap(original), restored]
    draw = ImageDraw.Draw(panel)
    for index, (title, image) in enumerate(zip(titles, images)):
        x = index * width
        draw.text((x + 16, 12), title, fill=(25, 29, 36), font=_font(20))
        preview = ImageOps.contain(image.convert("RGB"), (width - 32, height - 82))
        panel.paste(preview, (x + (width - preview.width) // 2, 48))
    detected = ", ".join(report.operations) if report.operations else "none"
    applied = ", ".join(report.applied_operations) if report.applied_operations else "none"
    draw.text((16, height - 40), f"Detected: {detected}", fill=(70, 76, 86), font=_font(14))
    draw.text((16, height - 22), f"Applied: {applied}", fill=(70, 76, 86), font=_font(14))
    panel.save(destination, format="PNG")


def _iter_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.casefold() in suffixes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect image degradations and create conservative restored examples.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/degradation_restore"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    inputs = _iter_inputs(args.input)
    if not inputs:
        raise SystemExit(f"No supported images found under {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in inputs:
        with Image.open(input_path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB").copy()
        report = analyze_degradation(original, input_name=str(input_path))
        restored, applied = restore_image(original, report)
        report = DegradationReport(
            report.input,
            report.size,
            report.metrics,
            report.artifacts,
            report.operations,
            applied,
        )
        restored_path = args.output_dir / f"{input_path.stem}-restored.png"
        panel_path = args.output_dir / f"{input_path.stem}-diagnostic.png"
        report_path = args.output_dir / f"{input_path.stem}-report.json"
        restored.save(restored_path, format="PNG")
        render_example(original, restored, report, panel_path)
        report_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"input: {input_path}")
        print(f"detected: {', '.join(report.operations) if report.operations else 'none'}")
        print(f"applied: {', '.join(report.applied_operations) if report.applied_operations else 'none'}")
        print(f"restored: {restored_path}")
        print(f"diagnostic: {panel_path}")
        print(f"report: {report_path}")


if __name__ == "__main__":
    main()
