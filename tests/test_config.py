from pathlib import Path

import pytest

from aigc_recognizer.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_default_config_loads_and_override_is_typed() -> None:
    config = load_config(DEFAULT_CONFIG, ["training.batch_size=2"])
    assert config.training.batch_size == 2
    assert config.model.backbone_name == "ViT-B-16"


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("training:\n  unknown_field: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(path)


def test_invalid_probability_sum_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must sum to 1"):
        load_config(
            DEFAULT_CONFIG,
            ["augmentations.transformed_clean_probability=0.4"],
        )


def test_unknown_official_evaluation_scenario_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unsupported scenario"):
        load_config(DEFAULT_CONFIG, ["official_evaluation.scenarios=[clean, unknown]"])
