"""Binary detector metrics with safe handling for incomplete smoke datasets."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def binary_metrics(
    labels: Sequence[float], probabilities: Sequence[float], threshold: float
) -> dict[str, float]:
    """Compute threshold-free and thresholded binary metrics."""
    targets = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    two_classes = np.unique(targets).size == 2
    return {
        "auroc": float(roc_auc_score(targets, scores)) if two_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(targets, scores)) if two_classes else float("nan")
        ),
        "balanced_accuracy": (
            float(balanced_accuracy_score(targets, predictions))
            if two_classes
            else float("nan")
        ),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
    }
