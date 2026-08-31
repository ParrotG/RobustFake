import pytest
import torch
from torch import nn

from aigc_recognizer.config import LossConfig, ModelConfig
from aigc_recognizer.losses import robust_detection_loss
from aigc_recognizer.model import (
    DetectorOutput,
    EncodedViews,
    FrozenClipDetector,
    ResidualStatisticsExtractor,
)


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


class DummyIntermediateEncoder(DummyVisualEncoder):
    def __init__(self, output_dim: int, intermediate_dim: int) -> None:
        super().__init__(output_dim)
        self.intermediate_projection = nn.Linear(3, intermediate_dim)

    def forward_intermediates(
        self,
        images: torch.Tensor,
        *,
        indices: list[int],
        **_kwargs: object,
    ) -> dict[str, object]:
        pooled = images.mean(dim=(-1, -2))
        prefix = self.intermediate_projection(pooled).unsqueeze(1)
        return {
            "image_features": self.projection(pooled),
            "image_intermediates_prefix": [prefix + index for index in indices],
        }


def test_model_is_view_permutation_invariant_and_backbone_is_frozen() -> None:
    config = ModelConfig(
        embedding_dim=8,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=False,
    )
    encoder = DummyVisualEncoder(8)
    model = FrozenClipDetector(encoder, config).eval()
    views = torch.randn(3, 2, 3, 8, 8)
    forward = model(views)
    reversed_views = model(views.flip(1))
    assert forward.logits.shape == (3,)
    assert forward.projections.shape == (3, 4)
    assert torch.allclose(forward.logits, reversed_views.logits, atol=1e-6)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert model.parameter_counts()["trainable"] < 5_000_000


def test_robust_loss_is_finite_and_only_heads_receive_gradients() -> None:
    config = ModelConfig(
        embedding_dim=8,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=False,
    )
    encoder = DummyVisualEncoder(8)
    model = FrozenClipDetector(encoder, config)
    clean = model(torch.randn(4, 2, 3, 8, 8))
    transformed = model(torch.randn(4, 2, 3, 8, 8))
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    losses = robust_detection_loss(clean, transformed, labels, LossConfig())
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert model.classifier.weight.grad is not None


def test_paired_forward_matches_separate_forward_calls() -> None:
    config = ModelConfig(
        embedding_dim=8,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=False,
    )
    model = FrozenClipDetector(DummyVisualEncoder(8), config).eval()
    clean_views = torch.randn(4, 2, 3, 8, 8)
    transformed_views = torch.randn(4, 2, 3, 8, 8)

    expected_clean = model(clean_views)
    expected_transformed = model(transformed_views)
    clean, transformed = model.forward_pair(clean_views, transformed_views)

    assert torch.allclose(clean.logits, expected_clean.logits, atol=1e-6)
    assert torch.allclose(clean.projections, expected_clean.projections, atol=1e-6)
    assert torch.allclose(transformed.logits, expected_transformed.logits, atol=1e-6)
    assert torch.allclose(
        transformed.projections, expected_transformed.projections, atol=1e-6
    )


def test_multilayer_online_and_cached_forward_match() -> None:
    config = ModelConfig(
        embedding_dim=8,
        intermediate_layers=[0, 2],
        intermediate_dim=5,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=False,
    )
    model = FrozenClipDetector(DummyIntermediateEncoder(8, 5), config).eval()
    views = torch.randn(3, 2, 3, 8, 8)

    encoded = model.encode_views(views)
    online = model(views)
    cached = model.forward_encoded(encoded)

    assert encoded.final.shape == (3, 2, 8)
    assert encoded.intermediate is not None
    assert encoded.intermediate.shape == (3, 2, 2, 5)
    assert torch.allclose(online.logits, cached.logits, atol=1e-6)
    cached.logits.sum().backward()
    assert model.intermediate_projections[0].weight.grad is not None
    assert model.layer_gate is not None
    assert model.layer_gate.weight.grad is not None


def test_multilayer_ablation_reuses_encoded_intermediate_features() -> None:
    config = ModelConfig(
        embedding_dim=8,
        intermediate_layers=[0, 2],
        multilayer_fusion_enabled=False,
        intermediate_dim=5,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=False,
    )
    model = FrozenClipDetector(DummyIntermediateEncoder(8, 5), config).eval()
    views = torch.randn(3, 2, 3, 8, 8)

    encoded = model.encode_views(views)
    output = model.forward_encoded(encoded)

    assert encoded.intermediate is not None
    assert output.logits.shape == (3,)
    assert len(model.intermediate_projections) == 0
    assert model.layer_gate is None


def test_probability_consistency_is_bounded_and_respects_ramp_scale() -> None:
    projections = torch.nn.functional.normalize(torch.randn(2, 4), dim=-1)
    clean = DetectorOutput(
        logits=torch.tensor([100.0, -100.0]),
        features=torch.zeros(2, 4),
        projections=projections,
    )
    transformed = DetectorOutput(
        logits=torch.tensor([-100.0, 100.0]),
        features=torch.zeros(2, 4),
        projections=projections,
    )
    labels = torch.tensor([1.0, 0.0])
    config = LossConfig(contrastive_weight=0.0)

    disabled = robust_detection_loss(
        clean, transformed, labels, config, consistency_scale=0.0
    )
    enabled = robust_detection_loss(
        clean, transformed, labels, config, consistency_scale=1.0
    )

    assert enabled["consistency"] <= 0.5
    assert torch.allclose(
        enabled["total"] - disabled["total"],
        config.consistency_weight * enabled["consistency"],
        atol=1e-5,
    )


def test_residual_statistics_branch_is_view_invariant_and_cache_compatible() -> None:
    config = ModelConfig(
        embedding_dim=8,
        head_dim=6,
        projection_dim=4,
        dropout=0.0,
        residual_statistics_enabled=True,
        residual_hidden_dim=5,
    )
    model = FrozenClipDetector(DummyVisualEncoder(8), config).eval()
    views = torch.randn(3, 2, 3, 16, 16)

    encoded = model.encode_views(views)
    online = model(views)
    cached = model.forward_encoded(encoded)
    reversed_views = model(views.flip(1))

    assert encoded.residual_statistics is not None
    assert encoded.residual_statistics.shape == (3, 2, 24)
    assert torch.isfinite(encoded.residual_statistics).all()
    assert torch.allclose(online.logits, cached.logits, atol=1e-6)
    assert torch.allclose(online.logits, reversed_views.logits, atol=1e-6)
    online.logits.sum().backward()
    assert model.residual_branch is not None
    assert model.residual_branch[1].weight.grad is not None

    without_statistics = EncodedViews(final=encoded.final)
    with torch.no_grad(), pytest.raises(ValueError, match="requires residual statistics"):
        model.forward_encoded(without_statistics)


def test_residual_statistics_extractor_rejects_invalid_channels() -> None:
    extractor = ResidualStatisticsExtractor()
    with pytest.raises(ValueError, match="Residual statistics require"):
        extractor(torch.randn(2, 2, 1, 16, 16))
