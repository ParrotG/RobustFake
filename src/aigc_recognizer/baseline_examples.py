"""Create the light-background UnivFD transformation-flip comparison asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from aigc_recognizer.data.transforms import canonical_rgb
from aigc_recognizer.external_eval import _scenario_image


RECORD_ID = "79e97b6afe9816b0f579f601cd458eeb3b182d428c4a9c5f01627aabf4f0f8ce"
SCENARIO = "resize_0.25"
MODELS = ("UnivFD", "RobustFake")
COLORS = {"UnivFD": "#2563EB", "RobustFake": "#0F766E"}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _threshold(result_path: Path) -> float:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return float(payload.get("threshold", 0.5))


def _predictions(path: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if str(record["id"]) == RECORD_ID and str(record["scenario"]) in {
                "clean",
                SCENARIO,
            }:
                selected[str(record["scenario"])] = record
    if set(selected) != {"clean", SCENARIO}:
        raise RuntimeError(f"Prediction file does not contain the curated case: {path}")
    return selected


def _collect(predictions: dict[str, Path], results: dict[str, Path]) -> dict[str, Any]:
    if set(predictions) != set(MODELS) or set(results) != set(MODELS):
        raise ValueError(f"Models must be exactly {MODELS}.")
    loaded = {model: _predictions(path) for model, path in predictions.items()}
    label = int(loaded["RobustFake"]["clean"]["label"])
    if label != 1:
        raise RuntimeError("The curated comparison must remain AI-generated.")
    models: dict[str, Any] = {}
    for model in MODELS:
        clean = loaded[model]["clean"]
        transformed = loaded[model][SCENARIO]
        if int(clean["label"]) != label or int(transformed["label"]) != label:
            raise RuntimeError(f"Label mismatch in {model} predictions.")
        threshold = _threshold(results[model])
        clean_prediction = int(float(clean["pred"]) >= threshold)
        transformed_prediction = int(float(transformed["pred"]) >= threshold)
        models[model] = {
            "threshold": threshold,
            "clean_score": float(clean["pred"]),
            "transformed_score": float(transformed["pred"]),
            "clean_correct": clean_prediction == label,
            "transformed_correct": transformed_prediction == label,
        }
    if not models["UnivFD"]["clean_correct"] or models["UnivFD"]["transformed_correct"]:
        raise RuntimeError("Curated case no longer makes UnivFD flip from correct to wrong.")
    if not models["RobustFake"]["clean_correct"] or not models["RobustFake"]["transformed_correct"]:
        raise RuntimeError("RobustFake is no longer correct before and after transformation.")
    return {
        "record_id": RECORD_ID,
        "image_path": str(loaded["RobustFake"]["clean"]["image_path"]),
        "label": label,
        "ground_truth": "AI-generated",
        "scenario": SCENARIO,
        "scenario_label": "Resize to 25%, then upscale",
        "models": models,
    }


def _render_previews(
    case: dict[str, Any], dataset_root: Path, output_directory: Path, seed: int
) -> None:
    source_path = dataset_root / str(case["image_path"])
    with Image.open(source_path) as source:
        clean = canonical_rgb(source.copy(), 127)
    seed_digest = hashlib.sha256(f"{seed}:{RECORD_ID}".encode("utf-8")).hexdigest()
    rng = random.Random(int(seed_digest[:16], 16))
    transformed = _scenario_image(clean.copy(), SCENARIO, rng)
    clean_path = output_directory / "univfd-flip-clean.jpg"
    transformed_path = output_directory / "univfd-flip-transformed.jpg"
    clean.save(clean_path, format="JPEG", quality=94, subsampling=0)
    transformed.save(transformed_path, format="JPEG", quality=94, subsampling=0)
    case["clean_preview"] = str(clean_path)
    case["transformed_preview"] = str(transformed_path)
    case["preview_protocol"] = (
        "Canonical RGB followed by the deterministic shared resize scenario. "
        "RobustFake additionally applies its label-independent standardization and views."
    )


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def _pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    fill: str,
    foreground: str,
    font_size: int,
) -> None:
    draw.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=fill)
    font = _font(font_size, bold=True)
    bounds = draw.textbbox((0, 0), text, font=font)
    x = (box[0] + box[2] - (bounds[2] - bounds[0])) / 2
    y = (box[1] + box[3] - (bounds[3] - bounds[1])) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=foreground)


def _score_card(
    draw: ImageDraw.ImageDraw,
    model: str,
    metrics: dict[str, Any],
    *,
    y: int,
) -> None:
    x, width, height = 1050, 480, 205
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=20,
        fill="#FFFFFF",
        outline="#DCE3EA",
        width=2,
    )
    draw.rectangle((x, y, x + 8, y + height), fill=COLORS[model])
    draw.text((x + 30, y + 25), model, font=_font(26, bold=True), fill="#172033")
    draw.text((x + 30, y + 72), "AIGC probability", font=_font(16), fill="#64748B")
    clean = 100.0 * float(metrics["clean_score"])
    transformed = 100.0 * float(metrics["transformed_score"])
    draw.text((x + 30, y + 110), f"{clean:.1f}%", font=_font(32, bold=True), fill="#172033")
    draw.text((x + 150, y + 116), "→", font=_font(26, bold=True), fill="#94A3B8")
    draw.text((x + 205, y + 110), f"{transformed:.1f}%", font=_font(32, bold=True), fill="#172033")
    for offset, correct in (
        (30, bool(metrics["clean_correct"])),
        (205, bool(metrics["transformed_correct"])),
    ):
        _pill(
            draw,
            (x + offset, y + 162, x + offset + 140, y + 194),
            "CORRECT" if correct else "WRONG",
            fill="#DCFCE7" if correct else "#FEE2E2",
            foreground="#166534" if correct else "#B91C1C",
            font_size=14,
        )


def _draw_comparison(case: dict[str, Any], destination: Path) -> None:
    canvas = Image.new("RGB", (1600, 900), "#F6F8FB")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 48), "A routine resize should not change the answer", font=_font(38, bold=True), fill="#172033")
    draw.text((70, 101), "Ground truth: AI-generated  ·  Resize to 25%, then upscale", font=_font(20), fill="#64748B")
    _pill(draw, (1260, 54, 1530, 100), "AI-GENERATED", fill="#EDE9FE", foreground="#6D28D9", font_size=17)
    for x, key, label in (
        (70, "clean_preview", "ORIGINAL"),
        (540, "transformed_preview", "AFTER RESIZE"),
    ):
        with Image.open(case[key]) as image:
            preview = _fit(image.convert("RGB"), (420, 590))
        canvas.paste(preview, (x, 180))
        draw.rounded_rectangle((x, 180, x + 420, 770), radius=14, outline="#CBD5E1", width=3)
        _pill(draw, (x + 18, 198, x + 205, 238), label, fill="#FFFFFF", foreground="#334155", font_size=15)
    draw.text((1050, 180), "Original  →  Transformed", font=_font(20, bold=True), fill="#475569")
    _score_card(draw, "UnivFD", case["models"]["UnivFD"], y=225)
    _score_card(draw, "RobustFake", case["models"]["RobustFake"], y=455)
    draw.rounded_rectangle((1050, 690, 1530, 770), radius=16, fill="#ECFDF5", outline="#A7F3D0", width=2)
    draw.text((1074, 712), "UnivFD flips. RobustFake stays correct.", font=_font(20, bold=True), fill="#065F46")
    draw.text((70, 842), "Representative WildFake official case · record 79e97b6afe98 · complete benchmark results are reported below", font=_font(15), fill="#94A3B8")
    canvas.save(destination, format="PNG", optimize=True)


def build_example(
    *,
    dataset_root: Path,
    predictions: dict[str, Path],
    results: dict[str, Path],
    output_directory: Path,
    seed: int = 2026,
) -> dict[str, Any]:
    """Validate the curated flip and write one light comparison plus its data."""
    output_directory.mkdir(parents=True, exist_ok=True)
    case = _collect(predictions, results)
    _render_previews(case, dataset_root, output_directory, seed)
    _draw_comparison(case, output_directory / "univfd-transform-flip.png")
    (output_directory / "univfd-transform-flip.json").write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_directory / "univfd-transform-flip.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_id",
                "ground_truth",
                "scenario",
                "model",
                "threshold",
                "clean_score",
                "transformed_score",
                "clean_correct",
                "transformed_correct",
            ),
        )
        writer.writeheader()
        for model in MODELS:
            writer.writerow(
                {
                    "record_id": case["record_id"],
                    "ground_truth": case["ground_truth"],
                    "scenario": case["scenario"],
                    "model": model,
                    **case["models"][model],
                }
            )
    return case


def main() -> None:
    """Generate the UnivFD flip comparison from completed result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--robustfake-result", type=Path, required=True)
    parser.add_argument("--robustfake-predictions", type=Path, required=True)
    parser.add_argument("--univfd-result", type=Path, required=True)
    parser.add_argument("--univfd-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    build_example(
        dataset_root=arguments.dataset_root,
        predictions={
            "UnivFD": arguments.univfd_predictions,
            "RobustFake": arguments.robustfake_predictions,
        },
        results={
            "UnivFD": arguments.univfd_result,
            "RobustFake": arguments.robustfake_result,
        },
        output_directory=arguments.output_dir,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
