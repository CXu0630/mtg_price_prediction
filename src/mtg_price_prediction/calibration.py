"""Group-conditional (Mondrian) split-conformal calibration for the deep GP's
80% intervals.

An interval correction is fit per *predicted*-median price bucket (true
price isn't known at inference time) on a held-out calibration split, then
applied by widening (or, for a negative correction, narrowing) both sides
of the raw 80% interval. All values are on the log(price) scale.
"""
from __future__ import annotations

import numpy as np

ALPHA = 0.2  # 1 - 0.8, for an 80% interval

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]


def bucket_of(log_prices: np.ndarray) -> np.ndarray:
    bucket_ids = np.zeros(len(log_prices), dtype=int)
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        bucket_ids[(log_prices >= lo) & (log_prices < hi)] = i
    return bucket_ids


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Standard finite-sample-correct split-conformal quantile level."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def fit_corrections(pred_median: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray, y: np.ndarray,
                    alpha: float = ALPHA) -> dict[int, float]:
    """Per-bucket additive correction, keyed by bucket index."""
    scores = np.maximum(p_lo - y, y - p_hi)
    bucket = bucket_of(pred_median)
    return {b: conformal_quantile(scores[bucket == b], alpha)
            for b in range(len(PRICE_BUCKETS)) if (bucket == b).any()}


def apply_corrections(pred_median: np.ndarray, p_lo: np.ndarray, p_hi: np.ndarray,
                      corrections: dict[int, float]) -> tuple[np.ndarray, np.ndarray]:
    correction = np.array([corrections.get(b, 0.0) for b in bucket_of(pred_median)])
    return p_lo - correction, p_hi + correction
