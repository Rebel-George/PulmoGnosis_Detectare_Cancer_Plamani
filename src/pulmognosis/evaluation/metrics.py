"""Binary classification metrics with bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float | np.ndarray = 0.5) -> dict[str, float]:
    """ROC-AUC (threshold-free) plus F1, sensitivity and specificity at `threshold`.

    `threshold` may be one value or one per patient (e.g. each fold's own cut-off).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= np.asarray(threshold)).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over patients. Resamples with a single class are skipped."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
