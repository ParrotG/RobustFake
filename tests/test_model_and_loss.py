import torch
from torch import nn

from aigc_recognizer.config import LossConfig, ModelConfig
from aigc_recognizer.losses import robust_detection_loss
from aigc_recognizer.model import FrozenClipDetector, HighFrequencyResidualBranch


class DummyVisualEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-1, -2)))


def test_model_is_view_permutation_invariant_and_backbone_is_frozen() -> None:
    config = ModelConfig(embedding_dim=8, head_dim=6, projection_dim=4, dropout=0.0)
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


def test_high_frequency_residual_branch_uses_fixed_filters_and_is_finite() -> None:
    config = ModelConfig(
        embedding_dim=8,
        head_dim=6,
        projection_dim=4,
        residual_channels=4,
        residual_embedding_dim=5,
        residual_head_dim=3,
    )
    branch = HighFrequencyResidualBranch(config)
    images = torch.rand(3, 3, 16, 16)
    features = branch(images)

    assert features.shape == (3, 5)
    assert torch.isfinite(features).all()
    assert branch.high_pass_kernels.requires_grad is False
    assert all(parameter.requires_grad for parameter in branch.encoder.parameters())

    constant_features = branch(torch.full_like(images, 0.5))
    assert torch.isfinite(constant_features).all()


def test_robust_loss_is_finite_and_only_heads_receive_gradients() -> None:
    config = ModelConfig(embedding_dim=8, head_dim=6, projection_dim=4, dropout=0.0)
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
    config = ModelConfig(embedding_dim=8, head_dim=6, projection_dim=4, dropout=0.0)
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
