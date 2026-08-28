"""Manifest-backed training dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from aigc_recognizer.config import AppConfig
from aigc_recognizer.data.transforms import RobustPairTransform


def load_manifest(path: str | Path, split: str) -> list[dict[str, Any]]:
    """Load and validate one split from an acquisition manifest."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"id", "path", "label", "split", "source_revision"}
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"Manifest line {line_number} is missing fields: {', '.join(sorted(missing))}"
            )
        if record["id"] in seen:
            raise ValueError(f"Manifest contains a duplicate id: {record['id']}")
        seen.add(record["id"])
        if record["split"] == split:
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records for split '{split}'.")
    return records


class AIGCManifestDataset(Dataset[dict[str, Any]]):
    """Load original images and create paired robust views online."""

    def __init__(self, config: AppConfig, split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError("Dataset split must be train or val.")
        self.config = config
        self.split = split
        self.records = load_manifest(config.data.manifest_path, split)
        self.root = Path(config.data.output_dir)
        self.transform = RobustPairTransform(config)

    def __len__(self) -> int:
        return len(self.records)

    def _validation_seed(self, record_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.config.project.seed}:{record_id}".encode("utf-8")
        ).hexdigest()
        return int(digest[:16], 16)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root / record["path"]
        try:
            with Image.open(image_path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to decode training image: {image_path}") from exc
        seed = self._validation_seed(record["id"]) if self.split == "val" else None
        views = self.transform(image, seed=seed)
        return {
            **views,
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "id": record["id"],
        }
