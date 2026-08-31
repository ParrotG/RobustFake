from pathlib import Path

import pytest

from aigc_recognizer.config import ConfigError, load_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_default_config_loads_and_override_is_typed() -> None:
    config = load_config(DEFAULT_CONFIG, ["training.batch_size=2"])
    assert config.training.batch_size == 2
    assert config.model.backbone_name == "ViT-B-16-quickgelu"


def test_default_training_profile_is_stability_oriented() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.project.run_name == "clip_b16_multilayer_v3"
    assert config.model.intermediate_layers == [3, 6, 9, 11]
    assert config.training.batch_size == 64
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.training.weight_decay == pytest.approx(1e-3)
    assert config.training.warmup_fraction == pytest.approx(0.05)
    assert config.training.num_workers == 12
    assert config.training.prefetch_factor == 1
    assert config.loss.contrastive_weight == pytest.approx(0.05)


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


def test_global_view_scale_must_not_be_smaller_than_local_view() -> None:
    with pytest.raises(ConfigError, match="View scales"):
        load_config(DEFAULT_CONFIG, ["views.global_crop_scale_min=0.4"])


def test_consistency_rampup_must_be_non_negative() -> None:
    with pytest.raises(ConfigError, match="rampup"):
        load_config(DEFAULT_CONFIG, ["loss.consistency_rampup_epochs=-1"])


def test_intermediate_layers_must_be_unique_ascending_block_indices() -> None:
    with pytest.raises(ConfigError, match="intermediate_layers"):
        load_config(DEFAULT_CONFIG, ["model.intermediate_layers=[3, 3, 12]"])


def test_feature_cache_dtype_is_validated() -> None:
    with pytest.raises(ConfigError, match="feature_cache.dtype"):
        load_config(DEFAULT_CONFIG, ["feature_cache.dtype=bfloat16"])


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
