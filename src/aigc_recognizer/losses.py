"""Losses for paired robust detector training."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from aigc_recognizer.config import LossConfig
from aigc_recognizer.model import DetectorOutput


def supervised_contrastive_loss(
    clean: torch.Tensor,
    transformed: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Pull same-class clean/transformed projections together within a batch."""
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive.")
    features = F.normalize(torch.cat([clean, transformed], dim=0), dim=-1)
    repeated_labels = torch.cat([labels, labels], dim=0).reshape(-1)
    sample_count = features.shape[0]
    logits = features @ features.T / temperature
    identity = torch.eye(sample_count, dtype=torch.bool, device=features.device)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    valid = ~identity
    positive = repeated_labels[:, None].eq(repeated_labels[None, :]) & valid
    exp_logits = torch.exp(logits) * valid
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    usable = positive_count > 0
    if not usable.any():
        return features.sum() * 0.0
    mean_log_probability = (positive * log_probability).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_probability[usable].mean()


def robust_detection_loss(
    clean: DetectorOutput,
    transformed: DetectorOutput,
    labels: torch.Tensor,
    config: LossConfig,
) -> dict[str, torch.Tensor]:
    """Combine classification, transformation consistency, and contrastive terms."""
    clean_bce = F.binary_cross_entropy_with_logits(clean.logits, labels)
    transformed_bce = F.binary_cross_entropy_with_logits(transformed.logits, labels)
    classification = 0.5 * (clean_bce + transformed_bce)
    consistency = F.smooth_l1_loss(transformed.logits, clean.logits.detach())
    contrastive = supervised_contrastive_loss(
        clean.projections,
        transformed.projections,
        labels,
        config.contrastive_temperature,
    )
    total = (
        config.classification_weight * classification
        + config.consistency_weight * consistency
        + config.contrastive_weight * contrastive
    )
    return {
        "total": total,
        "classification": classification.detach(),
        "clean_bce": clean_bce.detach(),
        "transformed_bce": transformed_bce.detach(),
        "consistency": consistency.detach(),
        "contrastive": contrastive.detach(),
    }
