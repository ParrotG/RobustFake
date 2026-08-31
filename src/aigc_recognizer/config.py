"""Strict, centralized configuration loading for the project."""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass
class ProjectConfig:
    seed: int = 2026
    run_name: str = "clip_b16_multilayer_v3"


@dataclass
class DataConfig:
    repo_id: str = "Shanmuk4622/ai-image-detection-dataset"
    revision: str = "8f1f536676f96cbc58bffd520ed50d1e7b9e894a"
    metadata_file: str = "metadata/manifest.parquet"
    source_config_file: str = "metadata/config.json"
    shard_cache_dir: str = "data/cache/ai_image_detection_shards"
    output_dir: str = "data/processed/mixed_aigc_80k"
    manifest_path: str = "data/processed/mixed_aigc_80k/manifest.jsonl"
    audit_path: str = "data/processed/mixed_aigc_80k/audit.json"
    state_path: str = "data/processed/mixed_aigc_80k/selection_state.json"
    nuisance_report_path: str = "data/processed/mixed_aigc_80k/nuisance_report.json"
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
    max_train_samples: int = 12_000
    max_evaluation_samples: int = 4_000
    feature_workers: int = 4


@dataclass
class MixedDataConfig:
    """Configuration for the reproducible multi-source training pool."""

    enabled: bool = True
    output_dir: str = "data/processed/mixed_aigc_80k"
    manifest_path: str = "data/processed/mixed_aigc_80k/manifest.jsonl"
    audit_path: str = "data/processed/mixed_aigc_80k/audit.json"
    state_path: str = "data/processed/mixed_aigc_80k/selection_state.json"
    dedup_report_path: str = "data/processed/mixed_aigc_80k/dedup_report.json"
    distribution_report_path: str = "data/processed/mixed_aigc_80k/distribution_report.json"
    cache_dir: str = "data/cache/mixed_aigc_80k"
    shanmuk_root: str = "data/processed/ai_image_detection_20k"
    community_root: str = "data/processed/community_forensics_20k"
    shanmuk_repo_id: str = "Shanmuk4622/ai-image-detection-dataset"
    shanmuk_revision: str = "8f1f536676f96cbc58bffd520ed50d1e7b9e894a"
    community_repo_id: str = "OwensLab/CommunityForensics-Small"
    community_revision: str = "6c539a534c07917307c381f5af4053c6091b5278"
    tiny_genimage_repo_id: str = "TheKernel01/Tiny-GenImage"
    tiny_genimage_revision: str = "89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69"
    wildfake_repo_id: str = "hy2628982280/WildFake"
    wildfake_revision: str = "18f53ff36ad9da60644039f0452b0e7b3907af6f"
    wildfake_train_metadata_file: str = (
        "split_train_test/csv_file/total_split/train_metadata.csv"
    )
    wildfake_train_metadata_sha256: str = (
        "26c6eacf6a34b7c61e7a1a3230c97624f58f857553cb525fe7f95dfbd5858be5"
    )
    target_total: int = 80_000
    target_real: int = 40_000
    target_fake: int = 40_000
    reserve_multiplier: float = 1.25
    max_network_gb: float = 80.0
    storage_warning_min_gb: float = 25.0
    storage_warning_max_gb: float = 45.0
    download_workers: int = 4
    hash_workers: int = 4
    hash_checkpoint_every: int = 5000
    checkpoint_every: int = 100
    network_max_retries: int = 5
    network_retry_base_seconds: float = 2.0
    request_timeout_seconds: float = 60.0
    max_image_pixels: int = 50_000_000
    phash_size: int = 16
    phash_distance: int = 8
    dhash_distance: int = 8
    crop_phash_distance: int = 8
    source_quotas: dict[str, dict[str, dict[str, int]]] = field(
        default_factory=lambda: {
            "shanmuk": {
                "train": {"0": 4000, "1": 4000},
                "val_id": {"0": 1000, "1": 1000},
                "val_dg": {"0": 0, "1": 0},
            },
            "wildfake": {
                "train": {"0": 9000, "1": 9000},
                "val_id": {"0": 0, "1": 0},
                "val_dg": {"0": 4000, "1": 2000},
            },
            "community_forensics": {
                "train": {"0": 13000, "1": 13000},
                "val_id": {"0": 1000, "1": 2000},
                "val_dg": {"0": 0, "1": 1000},
            },
            "tiny_genimage": {
                "train": {"0": 6000, "1": 6000},
                "val_id": {"0": 2000, "1": 1000},
                "val_dg": {"0": 0, "1": 1000},
            },
        }
    )
    external_deny_manifests: list[str] = field(
        default_factory=lambda: [
            "data/evaluation/wildfake_official/manifest.jsonl",
            "data/evaluation/wildfake_broad_6k/manifest.jsonl",
            "data/evaluation/sid_set_4k/manifest.jsonl",
        ]
    )
    wildfake_train_real_sources: list[str] = field(
        default_factory=lambda: ["laion5b", "imagenet", "ffhq", "celebahq"]
    )
    wildfake_dg_real_sources: list[str] = field(
        default_factory=lambda: ["afhq", "church"]
    )
    wildfake_dg_fake_architectures: list[str] = field(
        default_factory=lambda: ["DDPM", "GALIP", "MAGE"]
    )
    global_heldout_generator_aliases: list[str] = field(
        default_factory=lambda: ["ddpm", "galip", "mage", "wukong"]
    )
    community_subset_weights: dict[str, float] = field(
        default_factory=lambda: {"Systematic": 0.50, "Manual": 0.40, "Commercial": 0.10}
    )
    community_architecture_weights: dict[str, float] = field(
        default_factory=lambda: {"LatDiff": 0.50, "GAN": 0.20, "PixDiff": 0.15, "other": 0.15}
    )
    community_systematic_model_cap: int = 8
    community_other_model_cap: int = 200
    tiny_expected_rows: int = 35_000
    tiny_expected_license: str = "cc-by-nc-sa-4.0"
    tiny_generators: list[str] = field(
        default_factory=lambda: [
            "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"
        ]
    )
    tiny_allowed_empty_generators: list[str] = field(default_factory=lambda: ["SD14"])


@dataclass
class ViewsConfig:
    input_size: int = 224
    global_crop_scale_min: float = 0.90
    global_crop_scale_max: float = 1.0
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
    backbone_name: str = "ViT-B-16-quickgelu"
    pretrained: str = "openai"
    embedding_dim: int = 512
    intermediate_layers: list[int] = field(default_factory=list)
    intermediate_dim: int = 768
    head_dim: int = 256
    projection_dim: int = 128
    dropout: float = 0.10
    residual_statistics_enabled: bool = False
    residual_statistics_dim: int = 24
    residual_hidden_dim: int = 64
    # Legacy CNN residual settings retained for compatibility with the xyl API.
    residual_enabled: bool = False
    residual_channels: int = 16
    residual_embedding_dim: int = 64
    residual_head_dim: int = 64


@dataclass
class LossConfig:
    classification_weight: float = 1.0
    consistency_weight: float = 0.1
    consistency_rampup_epochs: int = 3
    contrastive_weight: float = 0.05
    contrastive_temperature: float = 0.10


@dataclass
class TrainingConfig:
    epochs: int = 8
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    warmup_fraction: float = 0.05
    num_workers: int = 12
    prefetch_factor: int = 1
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
class ProvenanceConfig:
    max_files: int = 10_000
    c2pa_tool_path: str | None = None
    c2pa_tool_timeout_seconds: float = 30.0
    c2pa_remote_manifest_fetch: bool = False
    c2pa_ocsp_fetch: bool = False
    c2pa_verify_trust: bool = True


@dataclass
class WatermarkConfig:
    enabled: bool = True
    ocr_enabled: bool = True
    tesseract_path: str | None = None
    ocr_timeout_seconds: float = 12.0
    max_dimension: int = 1600
    corner_fraction: float = 0.38
    upscale_factor: int = 3
    languages: str = "eng+chi_sim"


@dataclass
class PerspectiveConfig:
    max_dimension: int = 1600
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    hough_threshold: int = 50
    hough_min_line_length_ratio: float = 0.12
    hough_max_line_gap: int = 20
    min_long_line_length_ratio: float = 0.15
    similar_length_tolerance: float = 0.25
    angle_deduplication_degrees: float = 8.0
    max_lines: int = 24
    intersection_cluster_ratio: float = 0.035
    min_intersection_angle_degrees: float = 8.0
    min_vanishing_point_support: int = 2
    max_vanishing_distance_ratio: float = 4.0
    strict_color_length_selection: bool = True
    strict_color_distance: float = 25.0
    strict_length_tolerance: float = 0.35
    strict_max_length_cv: float = 0.20
    strict_anchor_x_ratio: float = 0.18
    strict_anchor_y_ratio: float = 0.62
    strict_angle_band_degrees: float = 35.0
    parallel_angle_tolerance_degrees: float = 10.0
    fisheye_min_contour_length_ratio: float = 0.20
    fisheye_min_contour_elongation: float = 3.0
    fisheye_min_span_ratio: float = 0.30
    fisheye_max_curved_contour_count: int = 16
    fisheye_min_support: int = 2
    fisheye_min_angle_change_degrees: float = 18.0
    fisheye_min_chord_excess: float = 0.035
    fisheye_min_line_residual_ratio: float = 0.025
    fisheye_score_threshold: float = 0.50
    fisheye_curvature_threshold: float = 0.045


@dataclass
class FeatureCacheConfig:
    """Settings for deterministic frozen-backbone feature precomputation."""

    root_dir: str = "data/processed/feature_cache"
    use_for_training: bool = False
    train_variants: int = 2
    shard_size: int = 2048
    batch_size: int = 64
    num_workers: int = 12
    prefetch_factor: int = 1
    pin_memory: bool = True
    dtype: str = "float16"


@dataclass
class OfficialEvaluationConfig:
    repo_id: str = "hy2628982280/WildFake"
    revision: str = "master"
    output_dir: str = "data/evaluation/wildfake_official"
    manifest_path: str = "data/evaluation/wildfake_official/manifest.jsonl"
    audit_path: str = "data/evaluation/wildfake_official/audit.json"
    metadata_dir: str = "data/cache/wildfake_official_metadata"
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


@dataclass
class EvaluationConfig:
    """Settings shared by every manifest-backed external evaluation."""

    checkpoint_path: str = "artifacts/runs/clip_b16_multilayer_v3/best.pt"
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 2
    save_predictions: bool = True
    enable_composed_scenarios: bool = True
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
    composed_scenarios: list[str] = field(
        default_factory=lambda: [
            "combo_social_resize_0.5_jpeg_70",
            "combo_repost_jpeg_90_resize_0.5_jpeg_70",
            "combo_crop_0.80_resize_0.5_jpeg_70",
            "combo_blur_1.0_resize_0.5_jpeg_50",
            "combo_edit_color_0.20_noise_0.02_jpeg_70",
            "combo_stress_crop_0.80_blur_1.0_resize_0.25_jpeg_30",
        ]
    )


@dataclass
class WildFakeEvaluationConfig:
    """Preparation and artifact locations for the broad WildFake sample."""

    repo_id: str = "hy2628982280/WildFake"
    revision: str = "master"
    metadata_file: str = "split_train_test/csv_file/total_split/test_metadata.csv"
    metadata_sha256: str = "0cec85fcec02e6e262f5b9726560b0355ab1293a30d29f063bf10a3c9d1b16c3"
    metadata_dir: str = "data/cache/wildfake_broad_metadata"
    integrity_cache_path: str = "data/cache/wildfake_broad_metadata/archive_integrity.json"
    output_dir: str = "data/evaluation/wildfake_broad_6k"
    manifest_path: str = "data/evaluation/wildfake_broad_6k/manifest.jsonl"
    audit_path: str = "data/evaluation/wildfake_broad_6k/audit.json"
    state_path: str = "data/evaluation/wildfake_broad_6k/preparation_state.json"
    results_path: str = "artifacts/evaluations/wildfake_broad_6k/results.json"
    predictions_path: str = "artifacts/evaluations/wildfake_broad_6k/predictions.jsonl"
    target_real: int = 3_000
    target_fake: int = 3_000
    fake_families: list[str] = field(
        default_factory=lambda: ["GAN_based", "Diffusion_based", "Other_based"]
    )
    fake_architectures: list[str] = field(
        default_factory=lambda: [
            "BigGAN",
            "DF-GAN",
            "GALIP",
            "GigaGAN",
            "starGAN",
            "styleGAN",
            "ADM",
            "DDIM",
            "DDPM",
            "Imagen",
            "VQDM",
            "MAE",
            "MAGE",
            "VQGAN",
            "VQVAE",
        ]
    )
    real_sources: list[str] = field(
        default_factory=lambda: ["afhq", "celebahq", "church", "ffhq", "imagenet", "laion5b"]
    )
    excluded_source_paths: list[str] = field(
        default_factory=lambda: [
            "GAN_based/Advanced/GigaGAN/fake_images/18598.png",
        ]
    )
    detect_extreme_zip_compression: bool = True
    extreme_zip_compression_ratio: float = 0.02
    download_workers: int = 4
    checkpoint_every: int = 100
    request_timeout_seconds: float = 60.0
    network_max_retries: int = 5
    network_retry_backoff: float = 1.0
    max_download_gb: float = 12.0


@dataclass
class SidEvaluationConfig:
    """Preparation and artifact locations for the SID-Set validation sample."""

    repo_id: str = "saberzl/SID_Set"
    revision: str = "dc03ead57929879319ce30a82bfcfb8d317b10bd"
    split: str = "validation"
    shard_prefix: str = "data/validation-"
    shard_cache_dir: str = "data/cache/sid_set_validation_shards"
    output_dir: str = "data/evaluation/sid_set_4k"
    manifest_path: str = "data/evaluation/sid_set_4k/manifest.jsonl"
    audit_path: str = "data/evaluation/sid_set_4k/audit.json"
    state_path: str = "data/evaluation/sid_set_4k/preparation_state.json"
    results_path: str = "artifacts/evaluations/sid_set_4k/results.json"
    predictions_path: str = "artifacts/evaluations/sid_set_4k/predictions.jsonl"
    hf_auth: str = "auto"
    target_real: int = 2_000
    target_fake: int = 2_000
    download_workers: int = 3
    checkpoint_every_shards: int = 1
    network_max_retries: int = 5
    network_retry_base_seconds: float = 5.0
    max_download_gb: float = 18.0
    max_shard_cache_gb: float = 2.0


@dataclass
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    standardization: StandardizationConfig = field(default_factory=StandardizationConfig)
    nuisance_audit: NuisanceAuditConfig = field(default_factory=NuisanceAuditConfig)
    mixed_data: MixedDataConfig = field(default_factory=MixedDataConfig)
    views: ViewsConfig = field(default_factory=ViewsConfig)
    augmentations: AugmentationsConfig = field(default_factory=AugmentationsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    feature_cache: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
    official_evaluation: OfficialEvaluationConfig = field(
        default_factory=OfficialEvaluationConfig
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wildfake_evaluation: WildFakeEvaluationConfig = field(
        default_factory=WildFakeEvaluationConfig
    )
    sid_evaluation: SidEvaluationConfig = field(default_factory=SidEvaluationConfig)

    def validate(self) -> None:
        """Validate cross-field constraints before any expensive work starts."""
        if self.data.repo_id != "Shanmuk4622/ai-image-detection-dataset":
            raise ConfigError("Only Shanmuk4622/ai-image-detection-dataset is supported.")
        mixed = self.mixed_data
        if mixed.enabled:
            if (
                self.data.output_dir != mixed.output_dir
                or self.data.manifest_path != mixed.manifest_path
                or self.data.audit_path != mixed.audit_path
                or self.data.state_path != mixed.state_path
            ):
                raise ConfigError("data paths must match mixed_data paths when mixed preparation is enabled.")
            if any(
                revision in {"main", "master"} or len(revision) != 40
                for revision in (
                    mixed.shanmuk_revision,
                    mixed.community_revision,
                    mixed.tiny_genimage_revision,
                    mixed.wildfake_revision,
                )
            ):
                raise ConfigError("Every mixed-data source must use a 40-character commit SHA.")
            if mixed.target_total != mixed.target_real + mixed.target_fake:
                raise ConfigError("Mixed-data class targets must sum to target_total.")
            if mixed.target_real != mixed.target_fake:
                raise ConfigError("Mixed-data real and fake targets must be equal.")
            configured = Counter()
            for source, splits in mixed.source_quotas.items():
                if source not in {"shanmuk", "wildfake", "community_forensics", "tiny_genimage"}:
                    raise ConfigError(f"Unsupported mixed-data source quota: {source}")
                if set(splits) != {"train", "val_id", "val_dg"}:
                    raise ConfigError("Each mixed-data source must define all three splits.")
                for split, labels in splits.items():
                    if set(labels) != {"0", "1"} or any(int(value) < 0 for value in labels.values()):
                        raise ConfigError("Mixed-data quotas require non-negative string label keys 0 and 1.")
                    for label, value in labels.items():
                        configured[(split, label)] += int(value)
            expected = {
                ("train", "0"): 32_000, ("train", "1"): 32_000,
                ("val_id", "0"): 4_000, ("val_id", "1"): 4_000,
                ("val_dg", "0"): 4_000, ("val_dg", "1"): 4_000,
            }
            if dict(configured) != expected:
                raise ConfigError("Mixed-data quotas must produce the locked 64k/8k/8k split table.")
            locked_source_quotas = {
                "shanmuk": {
                    "train": {"0": 4000, "1": 4000},
                    "val_id": {"0": 1000, "1": 1000},
                    "val_dg": {"0": 0, "1": 0},
                },
                "wildfake": {
                    "train": {"0": 9000, "1": 9000},
                    "val_id": {"0": 0, "1": 0},
                    "val_dg": {"0": 4000, "1": 2000},
                },
                "community_forensics": {
                    "train": {"0": 13000, "1": 13000},
                    "val_id": {"0": 1000, "1": 2000},
                    "val_dg": {"0": 0, "1": 1000},
                },
                "tiny_genimage": {
                    "train": {"0": 6000, "1": 6000},
                    "val_id": {"0": 2000, "1": 1000},
                    "val_dg": {"0": 0, "1": 1000},
                },
            }
            if mixed.source_quotas != locked_source_quotas:
                raise ConfigError("Mixed-data per-source quotas must match the locked 80k table.")
            if mixed.reserve_multiplier < 1 or mixed.max_network_gb <= 0:
                raise ConfigError("Mixed-data reserve and network limits are invalid.")
            if (
                mixed.download_workers <= 0
                or mixed.hash_workers <= 0
                or mixed.hash_checkpoint_every <= 0
                or mixed.checkpoint_every <= 0
            ):
                raise ConfigError("Mixed-data concurrency and checkpoint settings must be positive.")
            if (
                mixed.network_max_retries < 0
                or mixed.network_retry_base_seconds <= 0
                or mixed.request_timeout_seconds <= 0
            ):
                raise ConfigError("Mixed-data network retry settings are invalid.")
            if (
                mixed.storage_warning_min_gb <= 0
                or mixed.storage_warning_max_gb < mixed.storage_warning_min_gb
                or mixed.max_image_pixels <= 0
                or mixed.phash_size <= 0
                or mixed.phash_distance < 0
                or mixed.dhash_distance < 0
                or mixed.crop_phash_distance < mixed.phash_distance
            ):
                raise ConfigError("Mixed-data storage and image safety settings are invalid.")
            if len(set(mixed.tiny_generators)) != 8 or "Wukong" not in mixed.tiny_generators:
                raise ConfigError("Tiny-GenImage must define eight unique generators including Wukong.")
            if not set(mixed.tiny_allowed_empty_generators) < set(mixed.tiny_generators):
                raise ConfigError("Tiny-GenImage empty-generator exceptions must be a strict subset.")
            if not mixed.tiny_expected_license:
                raise ConfigError("Tiny-GenImage expected license must be configured.")
        if self.model.backbone_name not in {"ViT-B-16", "ViT-B-16-quickgelu"}:
            raise ConfigError("Only ViT-B-16 variants are currently supported.")
        if self.model.pretrained != "openai":
            raise ConfigError("Only the OpenAI pretrained weights are supported in v1.")
        if self.model.embedding_dim != 512:
            raise ConfigError("ViT-B-16 requires model.embedding_dim=512.")
        if self.model.intermediate_dim != 768:
            raise ConfigError("ViT-B-16 intermediate tokens require model.intermediate_dim=768.")
        if self.model.residual_statistics_dim != 24:
            raise ConfigError(
                "The fixed residual-statistics extractor requires model.residual_statistics_dim=24."
            )
        if self.model.residual_hidden_dim <= 0:
            raise ConfigError("model.residual_hidden_dim must be positive.")
        if (
            len(set(self.model.intermediate_layers)) != len(self.model.intermediate_layers)
            or self.model.intermediate_layers != sorted(self.model.intermediate_layers)
            or any(index < 0 or index >= 12 for index in self.model.intermediate_layers)
        ):
            raise ConfigError(
                "model.intermediate_layers must contain unique ascending ViT-B/16 block indices in [0, 11]."
            )
        if self.views.input_size != 224:
            raise ConfigError("ViT-B-16 currently requires views.input_size=224.")
        if not (
            0
            < self.views.local_scale_min
            <= self.views.local_scale_max
            <= self.views.global_crop_scale_min
            <= self.views.global_crop_scale_max
            <= 1
        ):
            raise ConfigError(
                "View scales must satisfy 0 < local min <= local max <= "
                "global min <= global max <= 1."
            )
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
        if (
            self.nuisance_audit.max_train_samples <= 0
            or self.nuisance_audit.max_evaluation_samples <= 0
            or self.nuisance_audit.feature_workers <= 0
        ):
            raise ConfigError("Nuisance classifier sampling and worker limits must be positive.")
        if self.training.batch_size <= 0 or self.training.gradient_accumulation_steps <= 0:
            raise ConfigError("Batch size and gradient accumulation must be positive.")
        if self.training.epochs <= 0 or self.training.learning_rate <= 0:
            raise ConfigError("Epoch count and learning rate must be positive.")
        if self.training.weight_decay < 0 or not 0 <= self.training.warmup_fraction < 1:
            raise ConfigError("Weight decay must be non-negative and warmup fraction in [0, 1).")
        if self.training.num_workers < 0 or self.training.prefetch_factor <= 0:
            raise ConfigError("Worker count must be non-negative and prefetch factor positive.")
        if self.training.amp_dtype not in {"fp16", "bf16"}:
            raise ConfigError("training.amp_dtype must be fp16 or bf16.")
        if self.provenance.max_files <= 0 or self.provenance.c2pa_tool_timeout_seconds <= 0:
            raise ConfigError("Provenance limits must be positive.")
        watermark = self.watermark
        if watermark.ocr_timeout_seconds <= 0 or watermark.max_dimension <= 0:
            raise ConfigError("Watermark OCR timeout and max dimension must be positive.")
        if not 0 < watermark.corner_fraction <= 1 or watermark.upscale_factor <= 0:
            raise ConfigError("Watermark corner fraction and upscale factor are invalid.")
        perspective = self.perspective
        if perspective.max_dimension <= 0 or perspective.hough_threshold <= 0:
            raise ConfigError("Perspective image dimension and Hough threshold must be positive.")
        if not 0 < perspective.hough_min_line_length_ratio <= 1:
            raise ConfigError("perspective.hough_min_line_length_ratio must be in (0, 1].")
        if not 0 < perspective.min_long_line_length_ratio <= 1:
            raise ConfigError("perspective.min_long_line_length_ratio must be in (0, 1].")
        if not 0 <= perspective.similar_length_tolerance < 1:
            raise ConfigError("perspective.similar_length_tolerance must be in [0, 1).")
        if perspective.max_lines < 3:
            raise ConfigError("perspective.max_lines must be at least 3.")
        if not 0 < perspective.intersection_cluster_ratio <= 1:
            raise ConfigError("perspective.intersection_cluster_ratio must be in (0, 1].")
        if perspective.min_vanishing_point_support < 2:
            raise ConfigError("perspective.min_vanishing_point_support must be at least 2.")
        if perspective.max_vanishing_distance_ratio <= 0:
            raise ConfigError("perspective.max_vanishing_distance_ratio must be positive.")
        if perspective.strict_color_distance <= 0 or perspective.strict_length_tolerance <= 0 or perspective.strict_max_length_cv <= 0:
            raise ConfigError("Strict perspective color and length thresholds must be positive.")
        if not 0 < perspective.strict_anchor_x_ratio <= 0.5:
            raise ConfigError("perspective.strict_anchor_x_ratio must be in (0, 0.5].")
        if not 0 < perspective.strict_anchor_y_ratio <= 1:
            raise ConfigError("perspective.strict_anchor_y_ratio must be in (0, 1].")
        if not 0 < perspective.strict_angle_band_degrees < 90:
            raise ConfigError("perspective.strict_angle_band_degrees must be in (0, 90).")
        if not 0 < perspective.parallel_angle_tolerance_degrees < 45:
            raise ConfigError("perspective.parallel_angle_tolerance_degrees must be in (0, 45).")
        if not 0 < perspective.fisheye_min_contour_length_ratio <= 1:
            raise ConfigError("perspective.fisheye_min_contour_length_ratio must be in (0, 1].")
        if perspective.fisheye_min_contour_elongation < 1:
            raise ConfigError("perspective.fisheye_min_contour_elongation must be at least 1.")
        if not 0 < perspective.fisheye_min_span_ratio <= 1:
            raise ConfigError("perspective.fisheye_min_span_ratio must be in (0, 1].")
        if perspective.fisheye_max_curved_contour_count < perspective.fisheye_min_support:
            raise ConfigError("perspective.fisheye_max_curved_contour_count must cover min support.")
        if perspective.fisheye_min_support < 2:
            raise ConfigError("perspective.fisheye_min_support must be at least 2.")
        if perspective.fisheye_min_angle_change_degrees <= 0:
            raise ConfigError("perspective.fisheye_min_angle_change_degrees must be positive.")
        if perspective.fisheye_min_chord_excess <= 0 or perspective.fisheye_min_line_residual_ratio <= 0:
            raise ConfigError("Fisheye curvature thresholds must be positive.")
        if not 0 < perspective.fisheye_score_threshold <= 1:
            raise ConfigError("perspective.fisheye_score_threshold must be in (0, 1].")
        if not 0 < perspective.fisheye_curvature_threshold < 1:
            raise ConfigError("perspective.fisheye_curvature_threshold must be in (0, 1).")
        cache = self.feature_cache
        if cache.train_variants <= 0 or cache.shard_size <= 0 or cache.batch_size <= 0:
            raise ConfigError("Feature-cache variants, shard size, and batch size must be positive.")
        if cache.num_workers < 0 or cache.prefetch_factor <= 0:
            raise ConfigError("Feature-cache worker count must be non-negative and prefetch factor positive.")
        if cache.dtype not in {"float16", "float32"}:
            raise ConfigError("feature_cache.dtype must be float16 or float32.")
        if self.loss.consistency_rampup_epochs < 0:
            raise ConfigError("loss.consistency_rampup_epochs must be non-negative.")
        if any(
            value < 0
            for value in (
                self.loss.classification_weight,
                self.loss.consistency_weight,
                self.loss.contrastive_weight,
            )
        ) or self.loss.contrastive_temperature <= 0:
            raise ConfigError("Loss weights must be non-negative and temperature positive.")
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
        evaluation = self.evaluation
        if evaluation.batch_size <= 0 or evaluation.num_workers < 0:
            raise ConfigError("External evaluation loader settings are invalid.")
        if evaluation.prefetch_factor <= 0:
            raise ConfigError("External evaluation prefetch factor must be positive.")
        all_supported_scenarios = supported_scenarios | {
            "combo_social_resize_0.5_jpeg_70",
            "combo_repost_jpeg_90_resize_0.5_jpeg_70",
            "combo_crop_0.80_resize_0.5_jpeg_70",
            "combo_blur_1.0_resize_0.5_jpeg_50",
            "combo_edit_color_0.20_noise_0.02_jpeg_70",
            "combo_stress_crop_0.80_blur_1.0_resize_0.25_jpeg_30",
        }
        configured_scenarios = evaluation.scenarios + evaluation.composed_scenarios
        if not evaluation.scenarios or not set(configured_scenarios) <= all_supported_scenarios:
            raise ConfigError("External evaluation contains an unsupported scenario.")
        if "clean" not in evaluation.scenarios:
            raise ConfigError("External evaluation scenarios must include clean.")
        if len(set(configured_scenarios)) != len(configured_scenarios):
            raise ConfigError("External evaluation scenarios must not contain duplicates.")
        broad = self.wildfake_evaluation
        if broad.repo_id != "hy2628982280/WildFake":
            raise ConfigError("The broad evaluation supports only hy2628982280/WildFake.")
        if broad.target_real <= 0 or broad.target_fake <= 0:
            raise ConfigError("Broad WildFake class targets must be positive.")
        if broad.download_workers <= 0 or broad.checkpoint_every <= 0:
            raise ConfigError("Broad WildFake concurrency and checkpoint settings must be positive.")
        if broad.max_download_gb <= 0 or broad.request_timeout_seconds <= 0:
            raise ConfigError("Broad WildFake download limits must be positive.")
        if broad.network_max_retries < 0 or broad.network_retry_backoff <= 0:
            raise ConfigError("Broad WildFake retry settings are invalid.")
        if not broad.fake_families or not broad.fake_architectures or not broad.real_sources:
            raise ConfigError("Broad WildFake strata must not be empty.")
        if len(set(broad.excluded_source_paths)) != len(broad.excluded_source_paths):
            raise ConfigError("Broad WildFake excluded source paths must be unique.")
        if any(
            not path or path.startswith("/") or ".." in Path(path).parts
            for path in broad.excluded_source_paths
        ):
            raise ConfigError("Broad WildFake excluded source paths must be safe relative paths.")
        if not 0 < broad.extreme_zip_compression_ratio <= 0.10:
            raise ConfigError(
                "Broad WildFake extreme ZIP compression ratio must be in (0, 0.10]."
            )
        sid = self.sid_evaluation
        if sid.repo_id != "saberzl/SID_Set" or not sid.revision:
            raise ConfigError("SID evaluation requires a pinned saberzl/SID_Set revision.")
        if sid.split != "validation":
            raise ConfigError("SID evaluation currently supports only the validation split.")
        if sid.hf_auth not in {"auto", "required", "disabled"}:
            raise ConfigError("sid_evaluation.hf_auth must be auto, required, or disabled.")
        if sid.target_real <= 0 or sid.target_fake <= 0:
            raise ConfigError("SID evaluation class targets must be positive.")
        if sid.download_workers <= 0 or sid.checkpoint_every_shards <= 0:
            raise ConfigError("SID evaluation concurrency and checkpoint settings must be positive.")
        if sid.max_download_gb <= 0 or sid.max_shard_cache_gb <= 0:
            raise ConfigError("SID evaluation download limits must be positive.")
        if sid.network_max_retries < 0 or sid.network_retry_base_seconds <= 0:
            raise ConfigError("SID evaluation retry settings are invalid.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-safe representation."""
        return dataclasses.asdict(self)


_SECTIONS: dict[str, type[Any]] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "standardization": StandardizationConfig,
    "nuisance_audit": NuisanceAuditConfig,
    "mixed_data": MixedDataConfig,
    "views": ViewsConfig,
    "augmentations": AugmentationsConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "training": TrainingConfig,
    "output": OutputConfig,
    "provenance": ProvenanceConfig,
    "watermark": WatermarkConfig,
    "perspective": PerspectiveConfig,
    "feature_cache": FeatureCacheConfig,
    "official_evaluation": OfficialEvaluationConfig,
    "evaluation": EvaluationConfig,
    "wildfake_evaluation": WildFakeEvaluationConfig,
    "sid_evaluation": SidEvaluationConfig,
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
