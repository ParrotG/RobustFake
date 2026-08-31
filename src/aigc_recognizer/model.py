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


@dataclass
class EncodedViews:
    """Frozen per-view features that can be cached before head training."""

    final: torch.Tensor
    intermediate: torch.Tensor | None = None


class FrozenClipDetector(nn.Module):
    """Aggregate frozen CLIP embeddings with a trainable invariant head."""

    def __init__(self, visual_encoder: nn.Module | None, config: ModelConfig) -> None:
        super().__init__()
        self.visual_encoder = visual_encoder
        self.config = config
        if self.visual_encoder is not None:
            for parameter in self.visual_encoder.parameters():
                parameter.requires_grad_(False)
            self.visual_encoder.eval()

        self.intermediate_projections = nn.ModuleList(
            nn.Linear(config.intermediate_dim, config.embedding_dim, bias=False)
            for _ in config.intermediate_layers
        )
        if config.intermediate_layers:
            self.layer_gate = nn.Linear(config.embedding_dim, 1)
            self.layer_bias = nn.Parameter(
                torch.zeros(len(config.intermediate_layers) + 1)
            )
        else:
            self.layer_gate = None
            self.register_parameter("layer_bias", None)

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
        if self.visual_encoder is not None:
            self.visual_encoder.eval()
        return self

    def _encode_images(self, images: torch.Tensor) -> EncodedViews:
        if self.visual_encoder is None:
            raise RuntimeError("The visual encoder is not loaded for cached-feature training.")
        with torch.inference_mode():
            if self.config.intermediate_layers:
                forward_intermediates = getattr(
                    self.visual_encoder, "forward_intermediates", None
                )
                if not callable(forward_intermediates):
                    raise RuntimeError(
                        "The visual encoder does not expose forward_intermediates."
                    )
                result = forward_intermediates(
                    images,
                    indices=self.config.intermediate_layers,
                    normalize_intermediates=True,
                    output_fmt="NLC",
                    output_extra_tokens=True,
                )
                embeddings = result["image_features"]
                prefixes = result.get("image_intermediates_prefix")
                if not isinstance(prefixes, list) or len(prefixes) != len(
                    self.config.intermediate_layers
                ):
                    raise RuntimeError("The visual encoder returned invalid intermediate tokens.")
                intermediate = torch.stack(
                    [tokens[:, 0] for tokens in prefixes], dim=1
                )
            else:
                embeddings = self.visual_encoder(images)
                intermediate = None
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
            raise RuntimeError("The visual encoder must return a [batch, embedding] tensor.")
        if embeddings.shape[-1] != self.config.embedding_dim:
            raise RuntimeError(
                f"Expected embedding dimension {self.config.embedding_dim}, "
                f"received {embeddings.shape[-1]}."
            )
        if intermediate is not None and (
            intermediate.ndim != 3
            or intermediate.shape[1] != len(self.config.intermediate_layers)
            or intermediate.shape[2] != self.config.intermediate_dim
        ):
            raise RuntimeError(
                "Intermediate tokens must have shape [batch, configured layers, intermediate dim]."
            )
        return EncodedViews(
            final=F.normalize(embeddings.float(), dim=-1),
            intermediate=(
                intermediate.float().clone() if intermediate is not None else None
            ),
        )

    def encode_views(self, views: torch.Tensor) -> EncodedViews:
        """Encode image views without evaluating the trainable detector heads."""
        if views.ndim != 5 or views.shape[1] < 2:
            raise ValueError("Detector input must have shape [batch, at least 2 views, C, H, W].")
        batch_size, view_count = views.shape[:2]
        encoded = self._encode_images(views.flatten(0, 1))
        intermediate = None
        if encoded.intermediate is not None:
            intermediate = encoded.intermediate.reshape(
                batch_size, view_count, len(self.config.intermediate_layers), -1
            )
        return EncodedViews(
            final=encoded.final.reshape(batch_size, view_count, -1),
            intermediate=intermediate,
        )

    def encode_pair(
        self, clean_views: torch.Tensor, transformed_views: torch.Tensor
    ) -> tuple[EncodedViews, EncodedViews]:
        """Encode clean and transformed views in one frozen-backbone invocation."""
        if clean_views.shape != transformed_views.shape:
            raise ValueError("Clean and transformed view tensors must have matching shapes.")
        batch_size = clean_views.shape[0]
        encoded = self.encode_views(torch.cat([clean_views, transformed_views], dim=0))

        def section(start: int, end: int) -> EncodedViews:
            intermediate = (
                encoded.intermediate[start:end]
                if encoded.intermediate is not None
                else None
            )
            return EncodedViews(encoded.final[start:end], intermediate)

        return section(0, batch_size), section(batch_size, batch_size * 2)

    def _fuse_layers(self, encoded: EncodedViews) -> torch.Tensor:
        final = F.normalize(encoded.final.float(), dim=-1)
        if not self.config.intermediate_layers:
            if encoded.intermediate is not None and encoded.intermediate.shape[-2] != 0:
                raise ValueError("Cached intermediate features do not match the model configuration.")
            return final
        if encoded.intermediate is None:
            raise ValueError("Configured intermediate layers require cached intermediate features.")
        expected = (
            *final.shape[:-1],
            len(self.config.intermediate_layers),
            self.config.intermediate_dim,
        )
        if encoded.intermediate.shape != expected:
            raise ValueError(
                f"Expected intermediate feature shape {expected}, received "
                f"{tuple(encoded.intermediate.shape)}."
            )
        projected = [
            F.normalize(projection(encoded.intermediate[..., index, :].float()), dim=-1)
            for index, projection in enumerate(self.intermediate_projections)
        ]
        layers = torch.stack([final, *projected], dim=-2)
        if self.layer_gate is None or self.layer_bias is None:
            raise RuntimeError("Layer fusion modules were not initialized.")
        scores = self.layer_gate(layers).squeeze(-1) + self.layer_bias
        weights = torch.softmax(scores, dim=-1)
        return F.normalize((layers * weights.unsqueeze(-1)).sum(dim=-2), dim=-1)

    def forward_encoded(self, encoded: EncodedViews) -> DetectorOutput:
        """Run trainable heads on online or precomputed per-view features."""
        if encoded.final.ndim != 3 or encoded.final.shape[1] < 2:
            raise ValueError("Encoded final features must have shape [batch, views, embedding].")
        if encoded.final.shape[-1] != self.config.embedding_dim:
            raise ValueError("Encoded final feature dimension does not match the model configuration.")
        embeddings = self._fuse_layers(encoded)
        mean = embeddings.mean(dim=1)
        standard_deviation = embeddings.std(dim=1, unbiased=False)
        aggregate = torch.cat([mean, standard_deviation], dim=-1)
        features = self.feature_head(aggregate)
        logits = self.classifier(features).squeeze(-1)
        projections = F.normalize(self.projection_head(features), dim=-1)
        return DetectorOutput(logits=logits, features=features, projections=projections)

    def forward(self, views: torch.Tensor) -> DetectorOutput:
        """Classify a [batch, views, channels, height, width] tensor."""
        return self.forward_encoded(self.encode_views(views))

    def forward_pair(
        self, clean_views: torch.Tensor, transformed_views: torch.Tensor
    ) -> tuple[DetectorOutput, DetectorOutput]:
        """Process clean and transformed views in one larger encoder invocation."""
        clean, transformed = self.encode_pair(clean_views, transformed_views)
        return self.forward_encoded(clean), self.forward_encoded(transformed)

    def forward_pair_encoded(
        self, clean: EncodedViews, transformed: EncodedViews
    ) -> tuple[DetectorOutput, DetectorOutput]:
        """Classify a cached clean/transformed feature pair."""
        return self.forward_encoded(clean), self.forward_encoded(transformed)

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


def create_cached_detector(config: ModelConfig) -> FrozenClipDetector:
    """Create only trainable heads for a precomputed-feature training run."""
    return FrozenClipDetector(None, config)
