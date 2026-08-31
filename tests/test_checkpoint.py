import dataclasses
from pathlib import Path

import pytest
import torch

from aigc_recognizer.checkpoint import load_inference_checkpoint
from aigc_recognizer.config import load_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def _checkpoint_payload(config: object) -> dict[str, object]:
    return {
        "config": config.to_dict(),
        "backbone": {
            "name": config.model.backbone_name,
            "pretrained": config.model.pretrained,
        },
        "trainable_model": {"weight": torch.ones(1)},
    }


def test_inference_checkpoint_restores_residual_architecture(tmp_path: Path) -> None:
    training_config = load_config(DEFAULT_CONFIG)
    training_config.model.residual_statistics_enabled = True
    checkpoint_path = tmp_path / "best.pt"
    torch.save(_checkpoint_payload(training_config), checkpoint_path)

    active_config = load_config(DEFAULT_CONFIG)
    active_config.model.residual_statistics_enabled = False
    assert not active_config.model.residual_statistics_enabled
    restored, checkpoint = load_inference_checkpoint(active_config, checkpoint_path)

    assert restored.model.residual_statistics_enabled
    assert dataclasses.asdict(restored.model) == dataclasses.asdict(training_config.model)
    assert restored.evaluation == active_config.evaluation
    assert checkpoint["trainable_model"]["weight"].item() == 1.0


def test_inference_checkpoint_rejects_inconsistent_backbone_metadata(
    tmp_path: Path,
) -> None:
    config = load_config(DEFAULT_CONFIG)
    payload = _checkpoint_payload(config)
    payload["backbone"] = {"name": "wrong", "pretrained": "openai"}
    checkpoint_path = tmp_path / "bad.pt"
    torch.save(payload, checkpoint_path)

    with pytest.raises(RuntimeError, match="backbone metadata"):
        load_inference_checkpoint(config, checkpoint_path)
