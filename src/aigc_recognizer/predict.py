"""Directory-to-JSON inference required by the hackathon submission."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from aigc_recognizer.checkpoint import load_inference_checkpoint
from aigc_recognizer.config import (
    AppConfig,
    config_argument_parser,
    load_config,
)
from aigc_recognizer.data.transforms import RobustPairTransform, canonical_rgb
from aigc_recognizer.model import FrozenClipDetector, create_detector
from aigc_recognizer.train import resolve_device
from aigc_recognizer.utils import seed_everything, seed_worker

LOGGER = logging.getLogger(__name__)
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def discover_images(input_directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return supported images in stable relative-path order."""
    root = Path(input_directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
    ]
    images.sort(key=lambda path: path.relative_to(root).as_posix().casefold())
    if not images:
        raise ValueError(f"Input directory contains no supported images: {root}")
    return images


class DirectoryInferenceDataset(Dataset[dict[str, Any]]):
    """Create deterministic clean global/local views for arbitrary images."""

    def __init__(self, config: AppConfig, root: Path, paths: list[Path]) -> None:
        self.config = config
        self.root = root
        self.paths = paths
        self.transform = RobustPairTransform(config)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        relative = path.relative_to(self.root).as_posix()
        try:
            with Image.open(path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to decode inference image: {path}") from exc
        digest = hashlib.sha256(
            f"{self.config.project.seed}:predict:{relative}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        image = canonical_rgb(image, self.config.views.padding_color)
        image = self.transform.standardize(image, rng)
        geometries = self.transform._geometries(image, rng)
        views = torch.stack(
            [
                self.transform._tensor(self.transform._render(image, geometry))
                for geometry in geometries
            ]
        )
        return {"views": views, "image_path": str(path)}


def _atomic_json(payload: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_inference_model(
    config: AppConfig, checkpoint_path: str | Path, device: torch.device
) -> FrozenClipDetector:
    """Load a detector using the architecture stored in its checkpoint."""
    inference_config, checkpoint = load_inference_checkpoint(config, checkpoint_path)
    model = create_detector(inference_config.model)
    model.load_trainable_state_dict(checkpoint["trainable_model"])
    return model.to(device).eval()


def _autocast(config: AppConfig, device: torch.device) -> Any:
    if not config.training.amp or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if config.training.amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def predict_directory(
    config: AppConfig,
    input_directory: str | Path,
    output_json: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    recursive: bool = True,
    model: FrozenClipDetector | None = None,
) -> list[dict[str, Any]]:
    """Score an image directory and atomically write the required JSON array."""
    seed_everything(config.project.seed)
    root = Path(input_directory)
    paths = discover_images(root, recursive=recursive)
    device = resolve_device(config)
    if model is None:
        config, checkpoint = load_inference_checkpoint(
            config, checkpoint_path or config.evaluation.checkpoint_path
        )
        detector = create_detector(config.model)
        detector.load_trainable_state_dict(checkpoint["trainable_model"])
    else:
        detector = model
    detector.to(device).eval()
    dataset = DirectoryInferenceDataset(config, root, paths)
    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.evaluation.batch_size,
        "shuffle": False,
        "num_workers": config.evaluation.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if config.evaluation.num_workers > 0:
        loader_arguments["prefetch_factor"] = config.evaluation.prefetch_factor
        loader_arguments["persistent_workers"] = False
    predictions: list[dict[str, Any]] = []
    for batch in tqdm(DataLoader(**loader_arguments), desc="Predict"):
        views = batch["views"].to(device, non_blocking=True)
        with _autocast(config, device):
            scores = torch.sigmoid(detector(views).logits).float().cpu().tolist()
        predictions.extend(
            {"image_path": path, "pred": float(score)}
            for path, score in zip(batch["image_path"], scores)
        )
    _atomic_json(predictions, Path(output_json))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def main() -> None:
    """Run the public directory-to-JSON inference command."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = config_argument_parser(
        "Score every image in a directory and write AIGC probabilities to JSON."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output-json", required=True, help="Destination JSON file.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Detector checkpoint. Defaults to evaluation.checkpoint_path.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Inspect only files directly inside the input directory.",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    predictions = predict_directory(
        config,
        arguments.input_dir,
        arguments.output_json,
        checkpoint_path=arguments.checkpoint,
        recursive=not arguments.no_recursive,
    )
    LOGGER.info(
        "Inference completed: images=%d output=%s",
        len(predictions),
        arguments.output_json,
    )


if __name__ == "__main__":
    main()
