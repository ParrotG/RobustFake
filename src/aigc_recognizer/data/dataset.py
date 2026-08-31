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


def canonical_group_name(value: object, field: str) -> str:
    """Normalize reporting aliases without changing manifest provenance."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    compact = "".join(character for character in raw.casefold() if character.isalnum())
    if field == "real_source":
        aliases = {
            "na": "unknown",
            "imagenet": "imagenet",
            "laion": "laion",
            "laion5b": "laion",
            "landscapeshq": "landscapeshq",
            "vision": "vision",
            "celebahq": "celebahq",
            "celebhq": "celebahq",
        }
        return aliases.get(compact, compact)
    if field == "generator_family":
        aliases = {
            "diffusionbased": "diffusion",
            "ganbased": "gan",
            "otherbased": "other",
        }
        return aliases.get(compact, compact)
    if field == "source_dataset":
        return raw.casefold().replace("-", "_").replace(" ", "_")
    return compact


def canonical_generator_family(value: object, architecture: object) -> str:
    """Unify source-specific family taxonomies and infer known missing families."""
    family = canonical_group_name(value, "generator_family")
    if family in {"latdiff", "pixdiff"}:
        return "diffusion"
    if family not in {"", "unknown"}:
        return family
    normalized_architecture = canonical_group_name(architecture, "architecture")
    diffusion_architectures = {
        "adm",
        "ddim",
        "ddpm",
        "fluxschnell",
        "glide",
        "imagen",
        "kandinsky22",
        "latdiff",
        "pixartsigma",
        "pixdiff",
        "sd14",
        "sd15",
        "sdxl",
        "vqdm",
        "wuerstchen",
        "wukong",
    }
    if normalized_architecture in diffusion_architectures:
        return "diffusion"
    return family or "unknown"


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

    def __init__(
        self, config: AppConfig, split: str, *, deterministic_variant: int | None = None
    ) -> None:
        if split not in {"train", "val", "val_id", "val_dg"}:
            raise ValueError("Dataset split must be train, val, val_id, or val_dg.")
        self.config = config
        self.split = split
        self.records = load_manifest(config.data.manifest_path, split)
        self.root = Path(config.data.output_dir)
        self.transform = RobustPairTransform(config)
        self.deterministic_variant = deterministic_variant

    def __len__(self) -> int:
        return len(self.records)

    def _validation_seed(self, record_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.config.project.seed}:{record_id}".encode("utf-8")
        ).hexdigest()
        return int(digest[:16], 16)

    def _transform_seed(self, record_id: str) -> int | None:
        if self.split != "train":
            return self._validation_seed(record_id)
        if self.deterministic_variant is None:
            return None
        digest = hashlib.sha256(
            f"{self.config.project.seed}:cache:{self.deterministic_variant}:{record_id}".encode(
                "utf-8"
            )
        ).hexdigest()
        return int(digest[:16], 16)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = int(record["label"])
        image_path = self.root / record["path"]
        try:
            with Image.open(image_path) as source:
                image = source.copy()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to decode training image: {image_path}") from exc
        seed = self._transform_seed(record["id"])
        views = self.transform(image, seed=seed)
        return {
            **views,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "id": record["id"],
            "source_dataset": canonical_group_name(
                record.get("source_dataset", "unknown"), "source_dataset"
            ),
            "real_source": (
                canonical_group_name(record.get("real_source", "unknown"), "real_source")
                if label == 0
                else ""
            ),
            "generator_family": (
                canonical_generator_family(
                    record.get("generator_family", "unknown"),
                    record.get("architecture", record.get("generator", "unknown")),
                )
                if label == 1
                else ""
            ),
            "architecture": (
                canonical_group_name(
                    record.get("architecture", record.get("generator", "unknown")),
                    "architecture",
                )
                if label == 1
                else ""
            ),
            "domain": canonical_group_name(
                (
                    record.get("real_source", "unknown")
                    if label == 0
                    else record.get(
                        "architecture", record.get("generator", "unknown")
                    )
                ),
                "real_source" if label == 0 else "architecture",
            ),
        }
