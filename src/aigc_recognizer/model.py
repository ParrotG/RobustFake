"""Frozen CLIP multi-view detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from aigc_recognizer.config import ModelConfig


@dataclass
class DetectorOutput:
    """Outputs required for classification and contrastive training."""

    logits: torch.Tensor
    features: torch.Tensor
    projections: torch.Tensor


class FrozenClipDetector(nn.Module):
    """Aggregate frozen CLIP embeddings with a trainable invariant head."""

    def __init__(self, visual_encoder: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.config = config
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad_(False)
        self.visual_encoder.eval()

        aggregate_dim = config.embedding_dim * 2
        self.feature_head = nn.Sequential(
            nn.LayerNorm(aggregate_dim),
            nn.Linear(aggregate_dim, config.head_dim),
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

    def forward(self, views: torch.Tensor) -> DetectorOutput:
        """Classify a [batch, views, channels, height, width] tensor."""
        if views.ndim != 5 or views.shape[1] < 2:
            raise ValueError("Detector input must have shape [batch, at least 2 views, C, H, W].")
        batch_size, view_count = views.shape[:2]
        embeddings = self._encode(views.flatten(0, 1)).reshape(batch_size, view_count, -1)
        mean = embeddings.mean(dim=1)
        standard_deviation = embeddings.std(dim=1, unbiased=False)
        aggregate = torch.cat([mean, standard_deviation], dim=-1)
        features = self.feature_head(aggregate)
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
