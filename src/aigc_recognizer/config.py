"""Strict, centralized configuration loading for the project."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass
class ProjectConfig:
    seed: int = 2026
    run_name: str = "clip_b16_multiview"


@dataclass
class DataConfig:
    repo_id: str = "OwensLab/CommunityForensics-Small"
    revision: str | None = "6c539a534c07917307c381f5af4053c6091b5278"
    shard_cache_dir: str = "data/cache/community_forensics_shards"
    shard_indices: list[int] = field(
        default_factory=lambda: [68, 70, 77, 78, 83, 115, 116, 156, 157]
    )
    max_shard_cache_gb: float = 12.0
    output_dir: str = "data/processed/community_forensics_20k"
    manifest_path: str = "data/processed/community_forensics_20k/manifest.jsonl"
    audit_path: str = "data/processed/community_forensics_20k/audit.json"
    hf_auth: str = "auto"
    max_scanned: int = 150_000
    checkpoint_every_scanned: int = 1_000
    network_max_retries: int = 5
    network_retry_base_seconds: float = 5.0
    max_download_gb: float = 22.0
    max_image_pixels: int = 50_000_000
    train_per_class: int = 8_000
    val_per_class: int = 2_000
    train_generator_percent: int = 80
    architecture_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "LatDiff": 0.60,
            "GAN": 0.15,
            "PixDiff": 0.10,
            "other": 0.15,
        }
    )
    systematic_per_model_cap: int = 6
    non_systematic_per_model_cap: int = 2_000
    max_real_source_fraction: float = 0.70
    exclude_nsfw: bool = True
    excluded_generator_tokens: list[str] = field(
        default_factory=lambda: ["dall-e", "dalle", "openai"]
    )
    excluded_real_source_tokens: list[str] = field(default_factory=lambda: ["coco"])
    exact_deduplication: bool = True
    perceptual_deduplication: bool = True
    perceptual_hash_size: int = 8


@dataclass
class ViewsConfig:
    input_size: int = 224
    local_scale_min: float = 0.50
    local_scale_max: float = 0.90
    padding_color: int = 127
    random_interpolation: bool = True


@dataclass
class AugmentationsConfig:
    transformed_clean_probability: float = 0.25
    single_operation_probability: float = 0.50
    double_operation_probability: float = 0.25
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 95
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 2.0
    resize_scale_min: float = 0.25
    resize_scale_max: float = 1.0
    gaussian_noise_sigma_min: float = 0.0
    gaussian_noise_sigma_max: float = 0.10
    color_jitter_strength: float = 0.20
    center_crop_min_fraction: float = 0.70
    center_crop_max_fraction: float = 1.0
    double_jpeg_weight: float = 0.08
    webp_weight: float = 0.05
    enable_double_jpeg: bool = True
    enable_webp: bool = True


@dataclass
class ModelConfig:
    backbone_name: str = "ViT-B-16"
    pretrained: str = "openai"
    embedding_dim: int = 512
    head_dim: int = 256
    projection_dim: int = 128
    dropout: float = 0.20


@dataclass
class LossConfig:
    classification_weight: float = 1.0
    consistency_weight: float = 0.5
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.10


@dataclass
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.10
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = False
    amp: bool = True
    amp_dtype: str = "fp16"
    device: str = "auto"
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    threshold: float = 0.5
    resume_from: str | None = None
    pin_memory: bool = True


@dataclass
class OutputConfig:
    root_dir: str = "artifacts/runs"
    save_last: bool = True
    keep_best_only: bool = False


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Validate cross-field constraints before any expensive work starts."""
        if self.data.repo_id != "OwensLab/CommunityForensics-Small":
            raise ConfigError("Only OwensLab/CommunityForensics-Small is supported in v1.")
        if self.model.backbone_name != "ViT-B-16":
            raise ConfigError("Only the ViT-B-16 backbone is supported in v1.")
        if self.model.pretrained != "openai":
            raise ConfigError("Only the OpenAI pretrained weights are supported in v1.")
        if self.model.embedding_dim != 512:
            raise ConfigError("ViT-B-16 requires model.embedding_dim=512.")
        if self.views.input_size != 224:
            raise ConfigError("ViT-B-16 currently requires views.input_size=224.")
        if not 0 < self.views.local_scale_min <= self.views.local_scale_max <= 1:
            raise ConfigError("Local view scales must satisfy 0 < min <= max <= 1.")
        if not 0 < self.data.train_generator_percent < 100:
            raise ConfigError("data.train_generator_percent must be between 1 and 99.")
        if self.data.train_per_class <= 0 or self.data.val_per_class <= 0:
            raise ConfigError("Per-class sample targets must be positive.")
        if self.data.max_scanned <= 0 or self.data.checkpoint_every_scanned <= 0:
            raise ConfigError("Scan and checkpoint limits must be positive.")
        if self.data.network_max_retries < 0 or self.data.network_retry_base_seconds <= 0:
            raise ConfigError("Network retry settings must be non-negative and positive.")
        if self.data.hf_auth not in {"auto", "required", "disabled"}:
            raise ConfigError("data.hf_auth must be auto, required, or disabled.")
        if self.data.max_download_gb <= 0:
            raise ConfigError("data.max_download_gb must be positive.")
        if not self.data.shard_indices or any(index < 0 for index in self.data.shard_indices):
            raise ConfigError("data.shard_indices must contain non-negative shard indices.")
        if len(set(self.data.shard_indices)) != len(self.data.shard_indices):
            raise ConfigError("data.shard_indices must not contain duplicates.")
        if self.data.max_shard_cache_gb <= 0:
            raise ConfigError("data.max_shard_cache_gb must be positive.")
        if self.data.systematic_per_model_cap <= 0 or self.data.non_systematic_per_model_cap <= 0:
            raise ConfigError("Generator sample caps must be positive.")
        ratios = self.data.architecture_ratios
        if set(ratios) != {"LatDiff", "GAN", "PixDiff", "other"}:
            raise ConfigError("Architecture ratios must define LatDiff, GAN, PixDiff, and other.")
        if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
            raise ConfigError("Architecture ratios must be non-negative and sum to 1.")
        probabilities = [
            self.augmentations.transformed_clean_probability,
            self.augmentations.single_operation_probability,
            self.augmentations.double_operation_probability,
        ]
        if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
            raise ConfigError("Augmentation operation-count probabilities must sum to 1.")
        bounded_probabilities = [
            self.data.max_real_source_fraction,
            self.model.dropout,
            self.training.warmup_fraction,
            self.training.threshold,
        ]
        if any(not 0 <= value <= 1 for value in bounded_probabilities):
            raise ConfigError("Probability and fraction fields must be in [0, 1].")
        if self.training.batch_size <= 0 or self.training.gradient_accumulation_steps <= 0:
            raise ConfigError("Batch size and gradient accumulation must be positive.")
        if self.training.num_workers < 0 or self.training.prefetch_factor <= 0:
            raise ConfigError("Worker count must be non-negative and prefetch factor positive.")
        if self.training.amp_dtype not in {"fp16", "bf16"}:
            raise ConfigError("training.amp_dtype must be fp16 or bf16.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation."""
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type[Any]] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "views": ViewsConfig,
    "augmentations": AugmentationsConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "training": TrainingConfig,
    "output": OutputConfig,
}


def _make_section(section_name: str, section_type: type[Any], raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Configuration section '{section_name}' must be a mapping.")
    allowed = {item.name for item in dataclasses.fields(section_type)}
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown keys in '{section_name}': {names}")
    return section_type(**dict(raw))


def _apply_override(raw: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ConfigError(f"Override must use section.key=value syntax: {expression}")
    path, raw_value = expression.split("=", 1)
    parts = path.split(".")
    if len(parts) != 2 or parts[0] not in _SECTIONS:
        raise ConfigError(f"Override must target a known section and field: {path}")
    section, key = parts
    allowed = {item.name for item in dataclasses.fields(_SECTIONS[section])}
    if key not in allowed:
        raise ConfigError(f"Unknown override field: {path}")
    raw.setdefault(section, {})[key] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> AppConfig:
    """Load a YAML configuration with strict keys and optional CLI overrides."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("The root configuration value must be a mapping.")
    unknown_sections = set(loaded) - set(_SECTIONS)
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ConfigError(f"Unknown configuration sections: {names}")
    for expression in overrides or []:
        _apply_override(loaded, expression)
    sections = {
        name: _make_section(name, section_type, loaded.get(name, {}))
        for name, section_type in _SECTIONS.items()
    }
    config = AppConfig(**sections)
    config.validate()
    return config


def config_argument_parser(description: str) -> argparse.ArgumentParser:
    """Build the common CLI parser used by public entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override one centralized configuration value. May be repeated.",
    )
    return parser
