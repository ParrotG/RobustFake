"""Frozen CLIP multi-view detector with a lightweight forensic branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from aigc_recognizer.config import ModelConfig


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class DetectorOutput:
    """Outputs required for classification and contrastive training."""

    logits: torch.Tensor
    features: torch.Tensor
    projections: torch.Tensor


def _make_high_pass_kernels() -> torch.Tensor:
    """Create fixed depthwise edge and Laplacian filters for RGB residuals."""
    laplacian = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ) / 4.0
    horizontal = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    vertical = horizontal.transpose(0, 1)
    kernels = torch.stack([laplacian, horizontal, vertical]).unsqueeze(1)
    return kernels.repeat(3, 1, 1, 1)


class HighFrequencyResidualBranch(nn.Module):
    """Encode fixed high-frequency RGB residuals with a small trainable CNN."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.output_dim = config.residual_embedding_dim
        self.register_buffer("high_pass_kernels", _make_high_pass_kernels())
        channels = config.residual_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(9, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=4 if channels % 4 == 0 else 1, num_channels=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(
                num_groups=4 if (channels * 2) % 4 == 0 else 1,
                num_channels=channels * 2,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels * 2,
                config.residual_embedding_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one L2-normalized residual embedding per image."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Residual branch input must have shape [batch, 3, height, width].")
        mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        rgb = (images.float() * std + mean).clamp(0.0, 1.0)
        residuals = F.conv2d(rgb, self.high_pass_kernels.float(), groups=3)
        features = self.encoder(residuals).flatten(1)
        return F.normalize(features.float(), dim=-1)


class FrozenClipDetector(nn.Module):
    """Aggregate frozen CLIP and trainable residual features across views."""

    def __init__(self, visual_encoder: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.config = config
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()

        clip_aggregate_dim = config.embedding_dim * 2
        self.clip_feature_head = nn.Sequential(
            nn.LayerNorm(clip_aggregate_dim),
            nn.Linear(clip_aggregate_dim, config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.residual_branch = (
            HighFrequencyResidualBranch(config) if config.residual_enabled else None
        )
        residual_head_dim = config.residual_head_dim if config.residual_enabled else 0
        if self.residual_branch is not None:
            residual_aggregate_dim = config.residual_embedding_dim * 2
            self.residual_feature_head = nn.Sequential(
                nn.LayerNorm(residual_aggregate_dim),
                nn.Linear(residual_aggregate_dim, residual_head_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        else:
            self.residual_feature_head = None
        fused_dim = config.head_dim + residual_head_dim
        self.feature_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, config.head_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.head_dim, 1)
        self.projection_head = nn.Sequential(
            nn.Linear(config.head_dim, config.head_dim),
            nn.GELU(),
            nn.Linear(config.head_dim, config.projection_dim),
        )

    def train(self, mode: bool = True) -> "FrozenClipDetector":
        """Keep the frozen encoder in evaluation mode while training the heads."""
        super().train(mode)
        self.visual_encoder.eval()
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.visual_encoder(images)
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise RuntimeError("The visual encoder must return a [batch, embedding] tensor.")
        if embeddings.shape[-1] != self.config.embedding_dim:
            raise RuntimeError(
                f"Expected embedding dimension {self.config.embedding_dim}, "
                f"received {embeddings.shape[-1]}."
            )
        return F.normalize(embeddings.float(), dim=-1)

    @staticmethod
    def _aggregate_views(embeddings: torch.Tensor) -> torch.Tensor:
        mean = embeddings.mean(dim=1)
        standard_deviation = embeddings.std(dim=1, unbiased=False)
        return torch.cat([mean, standard_deviation], dim=-1)

    def forward(self, views: torch.Tensor) -> DetectorOutput:
        """Classify a [batch, views, channels, height, width] tensor."""
        if views.ndim != 5 or views.shape[1] < 2:
            raise ValueError("Detector input must have shape [batch, at least 2 views, C, H, W].")
        batch_size, view_count = views.shape[:2]
        flattened_views = views.flatten(0, 1)
        clip_embeddings = self._encode(flattened_views).reshape(batch_size, view_count, -1)
        clip_features = self.clip_feature_head(self._aggregate_views(clip_embeddings))
        if self.residual_branch is not None and self.residual_feature_head is not None:
            residual_embeddings = self.residual_branch(flattened_views).reshape(
                batch_size, view_count, -1
            )
            residual_features = self.residual_feature_head(
                self._aggregate_views(residual_embeddings)
            )
            fused = torch.cat([clip_features, residual_features], dim=-1)
        else:
            fused = clip_features
        features = self.feature_head(fused)
        logits = self.classifier(features).squeeze(-1)
        projections = F.normalize(self.projection_head(features), dim=-1)
        return DetectorOutput(logits=logits, features=features, projections=projections)

    def forward_pair(
        self, clean_views: torch.Tensor, transformed_views: torch.Tensor
    ) -> tuple[DetectorOutput, DetectorOutput]:
        """Process clean and transformed views in one larger encoder invocation."""
        if clean_views.shape != transformed_views.shape:
            raise ValueError("Clean and transformed view tensors must have matching shapes.")
        batch_size = clean_views.shape[0]
        combined = self(torch.cat([clean_views, transformed_views], dim=0))
        clean = DetectorOutput(
            logits=combined.logits[:batch_size],
            features=combined.features[:batch_size],
            projections=combined.projections[:batch_size],
        )
        transformed = DetectorOutput(
            logits=combined.logits[batch_size:],
            features=combined.features[batch_size:],
            projections=combined.projections[batch_size:],
        )
        return clean, transformed

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only small trainable state, excluding frozen CLIP weights."""
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith("visual_encoder.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore only trainable components and validate all supplied keys."""
        result = self.load_state_dict(state, strict=False)
        unexpected = result.unexpected_keys
        missing_trainable = [
            name for name in result.missing_keys if not name.startswith("visual_encoder.")
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                f"Invalid trainable checkpoint. Missing={missing_trainable}, unexpected={unexpected}"
            )

    def parameter_counts(self) -> dict[str, int]:
        """Report total and trainable parameters for constraint auditing."""
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def load_open_clip_visual(config: ModelConfig) -> nn.Module:
    """Load the configured OpenCLIP model and retain only its visual encoder."""
    import open_clip

    clip_model = open_clip.create_model(config.backbone_name, pretrained=config.pretrained)
    visual = clip_model.visual
    actual_dimension = getattr(visual, "output_dim", config.embedding_dim)
    if actual_dimension != config.embedding_dim:
        raise RuntimeError(
            f"Configured embedding dimension {config.embedding_dim} does not match "
            f"the backbone output dimension {actual_dimension}."
        )
    return visual


def create_detector(config: ModelConfig, visual_encoder: nn.Module | None = None) -> FrozenClipDetector:
    """Create the detector, allowing a dependency-injected encoder for tests."""
    encoder = visual_encoder if visual_encoder is not None else load_open_clip_visual(config)
    return FrozenClipDetector(encoder, config)
