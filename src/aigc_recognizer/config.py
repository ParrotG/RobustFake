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
    repo_id: str = "Shanmuk4622/ai-image-detection-dataset"
    revision: str = "8f1f536676f96cbc58bffd520ed50d1e7b9e894a"
    metadata_file: str = "metadata/manifest.parquet"
    source_config_file: str = "metadata/config.json"
    shard_cache_dir: str = "data/cache/ai_image_detection_shards"
    output_dir: str = "data/processed/ai_image_detection_20k"
    manifest_path: str = "data/processed/ai_image_detection_20k/manifest.jsonl"
    audit_path: str = "data/processed/ai_image_detection_20k/audit.json"
    state_path: str = "data/processed/ai_image_detection_20k/preparation_state.json"
    nuisance_report_path: str = "data/processed/ai_image_detection_20k/nuisance_report.json"
    hf_auth: str = "required"
    generators: list[str] = field(
        default_factory=lambda: [
            "sd15",
            "sdxl",
            "flux_schnell",
            "kandinsky22",
            "pixart_sigma",
            "wuerstchen",
        ]
    )
    expected_parent_count: int = 10_000
    expected_pipeline_version: str = "1.2"
    expected_image_size: int = 512
    download_workers: int = 2
    checkpoint_every_shards: int = 1
    network_max_retries: int = 5
    network_retry_base_seconds: float = 5.0
    max_download_gb: float = 28.0
    max_shard_cache_gb: float = 3.0
    max_image_pixels: int = 50_000_000
    exact_deduplication: bool = True
    perceptual_deduplication: bool = True
    perceptual_hash_size: int = 16
    official_leakage_manifest: str = "data/evaluation/wildfake_official/manifest.jsonl"
    official_leakage_root: str = "data/evaluation/wildfake_official"
    leakage_phash_distance: int = 8
    leakage_dhash_distance: int = 8


@dataclass
class StandardizationConfig:
    enabled: bool = True
    application_probability: float = 0.75
    resize_weight: float = 0.30
    codec_weight: float = 0.30
    resize_codec_weight: float = 0.40
    resize_scale_min: float = 0.75
    resize_scale_max: float = 1.0
    jpeg_weight: float = 0.80
    webp_weight: float = 0.20
    quality_min: int = 85
    quality_max: int = 100


@dataclass
class NuisanceAuditConfig:
    enabled: bool = True
    random_state: int = 2026
    feature_size: int = 128
    max_iter: int = 150
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    permutation_repeats: int = 5


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
class OfficialEvaluationConfig:
    repo_id: str = "hy2628982280/WildFake"
    revision: str = "master"
    output_dir: str = "data/evaluation/wildfake_official"
    manifest_path: str = "data/evaluation/wildfake_official/manifest.jsonl"
    audit_path: str = "data/evaluation/wildfake_official/audit.json"
    metadata_dir: str = "data/cache/wildfake_official_metadata"
    checkpoint_path: str = "artifacts/runs/clip_b16_multiview/best.pt"
    results_path: str = "artifacts/evaluations/wildfake_official/results.json"
    predictions_path: str = "artifacts/evaluations/wildfake_official/predictions.jsonl"
    dalle_metadata_file: str = "label_csv_files/dalle3.csv"
    coco_metadata_file: str = "label_csv_files/real_coco.csv"
    dalle_archive_file: str = "Images/Diffusion_based/DALLE.zip"
    coco_archive_file: str = "Images/Real/coco.zip"
    dalle_archive_sha256: str = "5e4ebc56daa06ebeec99711b9cc204571558d3e17366f2df992a8cfd4f251d4c"
    coco_archive_sha256: str = "0b4dda0968e5f0d3cb60434c24204fcdac1cc0b40018093f15307edd545905b3"
    expected_real_count: int = 4_998
    expected_fake_count: int = 8_843
    max_download_gb: float = 4.0
    checkpoint_every: int = 100
    request_timeout_seconds: float = 60.0
    network_max_retries: int = 5
    network_retry_backoff: float = 1.0
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 2
    save_predictions: bool = True
    scenarios: list[str] = field(
        default_factory=lambda: [
            "clean",
            "jpeg_90",
            "jpeg_70",
            "jpeg_50",
            "jpeg_30",
            "blur_0.5",
            "blur_1.0",
            "blur_2.0",
            "resize_0.5",
            "resize_0.25",
            "noise_0.02",
            "noise_0.05",
            "noise_0.10",
            "color_jitter_0.20",
            "center_crop_0.80",
        ]
    )


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    nuisance_audit: NuisanceAuditConfig = field(default_factory=NuisanceAuditConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    official_evaluation: OfficialEvaluationConfig = field(
        default_factory=OfficialEvaluationConfig
    )

    def validate(self) -> None:
        """Validate cross-field constraints before any expensive work starts."""
        if self.data.repo_id != "Shanmuk4622/ai-image-detection-dataset":
            raise ConfigError("Only Shanmuk4622/ai-image-detection-dataset is supported.")
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
        if not self.data.revision:
            raise ConfigError("data.revision must pin a dataset commit.")
        if len(self.data.generators) != 6 or len(set(self.data.generators)) != 6:
            raise ConfigError("data.generators must contain six unique generator names.")
        if self.data.expected_parent_count <= 0 or self.data.expected_image_size <= 0:
            raise ConfigError("Expected dataset counts and dimensions must be positive.")
        if self.data.download_workers <= 0 or self.data.checkpoint_every_shards <= 0:
            raise ConfigError("Download workers and shard checkpoint interval must be positive.")
        if self.data.network_max_retries < 0 or self.data.network_retry_base_seconds <= 0:
            raise ConfigError("Network retry settings must be non-negative and positive.")
        if self.data.hf_auth not in {"auto", "required", "disabled"}:
            raise ConfigError("data.hf_auth must be auto, required, or disabled.")
        if self.data.max_download_gb <= 0:
            raise ConfigError("data.max_download_gb must be positive.")
        if self.data.max_shard_cache_gb <= 0:
            raise ConfigError("data.max_shard_cache_gb must be positive.")
        if self.data.leakage_phash_distance < 0 or self.data.leakage_dhash_distance < 0:
            raise ConfigError("Leakage hash distances must be non-negative.")
        probabilities = [
            self.augmentations.transformed_clean_probability,
            self.augmentations.single_operation_probability,
            self.augmentations.double_operation_probability,
        ]
        if any(value < 0 for value in probabilities) or abs(sum(probabilities) - 1.0) > 1e-6:
            raise ConfigError("Augmentation operation-count probabilities must sum to 1.")
        bounded_probabilities = [
            self.standardization.application_probability,
            self.model.dropout,
            self.training.warmup_fraction,
            self.training.threshold,
        ]
        if any(not 0 <= value <= 1 for value in bounded_probabilities):
            raise ConfigError("Probability and fraction fields must be in [0, 1].")
        standardization_weights = [
            self.standardization.resize_weight,
            self.standardization.codec_weight,
            self.standardization.resize_codec_weight,
        ]
        if any(value < 0 for value in standardization_weights) or abs(sum(standardization_weights) - 1.0) > 1e-6:
            raise ConfigError("Standardization operation weights must sum to 1.")
        codec_weights = [self.standardization.jpeg_weight, self.standardization.webp_weight]
        if any(value < 0 for value in codec_weights) or abs(sum(codec_weights) - 1.0) > 1e-6:
            raise ConfigError("Standardization codec weights must sum to 1.")
        if not 0 < self.standardization.resize_scale_min <= self.standardization.resize_scale_max <= 1:
            raise ConfigError("Standardization resize scales must satisfy 0 < min <= max <= 1.")
        if not 1 <= self.standardization.quality_min <= self.standardization.quality_max <= 100:
            raise ConfigError("Standardization codec quality must be in [1, 100].")
        if (
            self.nuisance_audit.feature_size <= 0
            or self.nuisance_audit.max_iter <= 0
            or self.nuisance_audit.max_leaf_nodes < 2
            or self.nuisance_audit.min_samples_leaf <= 0
        ):
            raise ConfigError("Nuisance classifier limits are invalid.")
        if self.nuisance_audit.learning_rate <= 0 or self.nuisance_audit.permutation_repeats <= 0:
            raise ConfigError("Nuisance classifier settings must be positive.")
        if self.training.batch_size <= 0 or self.training.gradient_accumulation_steps <= 0:
            raise ConfigError("Batch size and gradient accumulation must be positive.")
        if self.training.num_workers < 0 or self.training.prefetch_factor <= 0:
            raise ConfigError("Worker count must be non-negative and prefetch factor positive.")
        if self.training.amp_dtype not in {"fp16", "bf16"}:
            raise ConfigError("training.amp_dtype must be fp16 or bf16.")
        official = self.official_evaluation
        if official.repo_id != "hy2628982280/WildFake":
            raise ConfigError("Only hy2628982280/WildFake is supported for official evaluation.")
        if official.expected_real_count <= 0 or official.expected_fake_count <= 0:
            raise ConfigError("Official evaluation class counts must be positive.")
        if official.max_download_gb <= 0 or official.checkpoint_every <= 0:
            raise ConfigError("Official evaluation limits must be positive.")
        if official.request_timeout_seconds <= 0 or official.network_max_retries < 0:
            raise ConfigError("Official evaluation network settings are invalid.")
        if official.network_retry_backoff <= 0:
            raise ConfigError("Official evaluation retry backoff must be positive.")
        if official.batch_size <= 0 or official.num_workers < 0:
            raise ConfigError("Official evaluation loader settings are invalid.")
        if official.prefetch_factor <= 0:
            raise ConfigError("Official evaluation prefetch factor must be positive.")
        supported_scenarios = {
            "clean",
            "jpeg_90",
            "jpeg_70",
            "jpeg_50",
            "jpeg_30",
            "blur_0.5",
            "blur_1.0",
            "blur_2.0",
            "resize_0.5",
            "resize_0.25",
            "noise_0.02",
            "noise_0.05",
            "noise_0.10",
            "color_jitter_0.20",
            "center_crop_0.80",
        }
        if not official.scenarios or not set(official.scenarios) <= supported_scenarios:
            raise ConfigError("Official evaluation contains an unsupported scenario.")
        if "clean" not in official.scenarios:
            raise ConfigError("Official evaluation scenarios must include clean.")
        if len(set(official.scenarios)) != len(official.scenarios):
            raise ConfigError("Official evaluation scenarios must not contain duplicates.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation."""
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type[Any]] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "standardization": StandardizationConfig,
    "nuisance_audit": NuisanceAuditConfig,
    "views": ViewsConfig,
    "augmentations": AugmentationsConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "training": TrainingConfig,
    "output": OutputConfig,
    "official_evaluation": OfficialEvaluationConfig,
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
