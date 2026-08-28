import torch
from torch import nn

from aigc_recognizer.config import LossConfig, ModelConfig
from aigc_recognizer.losses import robust_detection_loss
from aigc_recognizer.model import FrozenClipDetector


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
