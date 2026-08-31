"""Checkpoint loading helpers shared by evaluation and public inference."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

from aigc_recognizer.config import (
    AppConfig,
    ModelConfig,
    StandardizationConfig,
    ViewsConfig,
)


LOGGER = logging.getLogger(__name__)


def checkpoint_section(
    checkpoint: dict[str, Any], name: str, section_type: type[Any]
) -> Any:
    """Reconstruct one known configuration section from a checkpoint."""
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise RuntimeError("Checkpoint does not contain its training configuration.")
    raw_section = raw_config.get(name)
    if not isinstance(raw_section, dict):
        raise RuntimeError(f"Checkpoint does not contain a {name} configuration.")
    allowed = {field.name for field in dataclasses.fields(section_type)}
    return section_type(
        **{key: value for key, value in raw_section.items() if key in allowed}
    )


def load_inference_checkpoint(
    config: AppConfig, checkpoint_path: str | Path
) -> tuple[AppConfig, dict[str, Any]]:
    """Load a checkpoint and restore every preprocessing/model architecture field.

    Runtime evaluation settings remain controlled by the active configuration, while
    model and input preprocessing settings come from the checkpoint that produced the
    weights. This keeps checkpoints self-contained across architecture variants.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Detector checkpoint must contain a mapping payload.")

    restored_sections = {
        "model": checkpoint_section(checkpoint, "model", ModelConfig),
        "views": checkpoint_section(checkpoint, "views", ViewsConfig),
        "standardization": checkpoint_section(
            checkpoint, "standardization", StandardizationConfig
        ),
    }
    for name, restored in restored_sections.items():
        active = getattr(config, name)
        if dataclasses.asdict(active) != dataclasses.asdict(restored):
            LOGGER.info("Using %s configuration stored in checkpoint %s.", name, path)

    inference_config = dataclasses.replace(config, **restored_sections)
    inference_config.validate()
    expected_backbone = {
        "name": inference_config.model.backbone_name,
        "pretrained": inference_config.model.pretrained,
    }
    if checkpoint.get("backbone") != expected_backbone:
        raise RuntimeError("Checkpoint backbone metadata does not match its model configuration.")
    if not isinstance(checkpoint.get("trainable_model"), dict):
        raise RuntimeError("Checkpoint does not contain trainable model weights.")
    return inference_config, checkpoint
