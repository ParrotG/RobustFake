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


def validate_preparation(config: AppConfig) -> dict[str, Any]:
    """Reject absent or incomplete acquisition state before model initialization."""
    manifest_path = Path(config.data.manifest_path)
    audit_path = Path(config.data.audit_path)
    if not manifest_path.is_file():
        object_roots = [
            Path(config.data.output_dir) / "images",
            Path(config.data.output_dir) / "objects",
        ]
        has_partial_images = any(
            root.is_dir() and any(path.is_file() for path in root.rglob("*"))
            for root in object_roots
        )
        detail = (
            " Image files are present, but preparation did not commit its manifest. "
            "This can happen after an interrupted preparation run. Rerun aigc-prepare; "
            "existing matching files will be reused."
            if has_partial_images
            else " Run aigc-prepare before training."
        )
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}.{detail}")
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Preparation audit does not exist: {audit_path}. Rerun aigc-prepare before training."
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Preparation audit cannot be read: {audit_path}") from exc
    if not bool(audit.get("complete")):
        raise RuntimeError(
            "Dataset preparation is incomplete "
            f"({audit.get('selected', 0)} images; {audit.get('stop_reason', 'unknown reason')}). "
            "Rerun aigc-prepare to resume before training."
        )
    if (
        config.mixed_data.enabled
        and "target_total" in audit
        and int(audit.get("selected", -1)) != config.mixed_data.target_total
    ):
        raise RuntimeError("Completed mixed-data audit does not contain the configured target total.")
    return audit


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
        if split not in {"train", "val", "val_id", "val_dg"}:
            raise ValueError("Dataset split must be train, val, val_id, or val_dg.")
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
        seed = self._validation_seed(record["id"]) if self.split != "train" else None
        views = self.transform(image, seed=seed)
        return {
            **views,
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "id": record["id"],
            "source_dataset": str(record.get("source_dataset", "unknown")),
            "real_source": str(record.get("real_source", "")),
            "generator_family": str(record.get("generator_family", "")),
            "architecture": str(record.get("architecture", record.get("generator", "unknown"))),
            "domain": str(
                record.get("real_source")
                if int(record["label"]) == 0
                else record.get("architecture", record.get("generator", "unknown"))
            ),
        }
