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


def test_unknown_external_evaluation_scenario_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unsupported scenario"):
        load_config(DEFAULT_CONFIG, ["evaluation.scenarios=[clean, unknown]"])


def test_composed_evaluation_can_be_disabled_from_central_config() -> None:
    config = load_config(DEFAULT_CONFIG, ["evaluation.enable_composed_scenarios=false"])
    assert config.evaluation.enable_composed_scenarios is False
    assert config.evaluation.composed_scenarios


def test_unknown_composed_evaluation_scenario_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unsupported scenario"):
        load_config(DEFAULT_CONFIG, ["evaluation.composed_scenarios=[unknown]"])


def test_unsafe_wildfake_exclusion_path_is_rejected() -> None:
    with pytest.raises(ConfigError, match="safe relative paths"):
        load_config(
            DEFAULT_CONFIG,
            ["wildfake_evaluation.excluded_source_paths=[../outside.png]"],
        )


def test_mixed_source_revision_must_be_immutable() -> None:
    with pytest.raises(ConfigError, match="40-character commit SHA"):
        load_config(DEFAULT_CONFIG, ["mixed_data.tiny_genimage_revision=main"])
