"""Group-conditional (Mondrian) conformal calibration for a Model 2
checkpoint, same method as scripts/calibrate_model1.py (see its docstring
for the full rationale) -- a per-predicted-price-bucket interval
correction, since a single global correction would be dominated by the
huge cheap-card population and barely touch the $10-100/$100+ tail.

Model 2's SVGP posterior gives a Gaussian (mean, variance) directly, so the
"raw" [p10, p90] interval here is mean +/- 1.2816*std rather than CatBoost's
native MultiQuantile output, but the calibration step itself is identical:
a nonconformity score `max(q_lo - y, y - q_hi)`, conformal-quantile
correction per bucket, applied by widening both sides.

Uses the val split (from load_data(), same split the checkpoint was
trained/early-stopped with) as the calibration set -- it wasn't used for
gradient updates, only early-stopping model selection, so it's a
legitimate held-out set for computing the correction (same reasoning as
Model 1's calib split).

**Two correction modes**, both group-conditional (same buckets, same
conformal-quantile machinery), differing only in what gets corrected:
- "flat" (the original method above): the same *additive* correction for
  every card in a bucket, sized to fix the worst-calibrated card in it --
  valid, but blunt: a card the model was actually confident about gets
  punished with the same wide interval as one it wasn't.
- "normalized": scales the correction by each card's *own* posterior std
  (a real per-card signal the GP gives us, unlike Model 1's quantile
  regression) -- the nonconformity score is divided by std before taking
  the per-bucket conformal quantile, then multiplied back by the query
  card's own std when building its interval. Confident predictions get a
  smaller absolute widening than uncertain ones, even within the same
  bucket, while keeping the same coverage guarantee (Lei et al.'s
  "normalized/studentized" conformal prediction).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from train_model2 import Config, MODEL_OUT_PATH, build_model, load_data, _predict

ALPHA = 0.2
PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]


def _bucket_of(values: np.ndarray) -> np.ndarray:
    bucket_ids = np.zeros(len(values), dtype=int)
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        bucket_ids[(values >= lo) & (values < hi)] = i
    return bucket_ids


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def _fit_corrections(pred_mean: np.ndarray, pred_std: np.ndarray, y: np.ndarray, normalized: bool) -> dict:
    """Per-bucket correction factors, fit on a calibration set (val).
    `normalized=False`: correction is a log-price amount, applied by
    addition. `normalized=True`: correction is a dimensionless multiple of
    std, applied by multiplying the query's own std (see module docstring).
    """
    p10, p90 = pred_mean - 1.2816 * pred_std, pred_mean + 1.2816 * pred_std
    raw_scores = np.maximum(p10 - y, y - p90)
    scores = raw_scores / pred_std if normalized else raw_scores

    bucket = _bucket_of(pred_mean)
    corrections = {}
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = bucket == b
        if mask.sum() == 0:
            continue
        if mask.sum() < 20:
            print(f"  WARNING: only {mask.sum()} calibration points in {label}, correction may be unstable")
        corrections[b] = _conformal_quantile(scores[mask], ALPHA)
        kind = "x std" if normalized else "(log-price)"
        print(f"  calibration bucket {label:10s} n={mask.sum():5d}  correction={corrections[b]:+.4f} {kind}")
    return corrections


def _apply_corrections(pred_mean: np.ndarray, pred_std: np.ndarray, corrections: dict, normalized: bool) -> tuple[np.ndarray, np.ndarray]:
    p10, p90 = pred_mean - 1.2816 * pred_std, pred_mean + 1.2816 * pred_std
    bucket = _bucket_of(pred_mean)
    correction_per_row = np.array([corrections.get(b, 0.0) for b in bucket])
    if normalized:
        correction_per_row = correction_per_row * pred_std
    return p10 - correction_per_row, p90 + correction_per_row


def _report(test_mean, test_std, test_y, p10_raw, p90_raw, p10_cal, p90_cal, method_name: str) -> None:
    true_bucket = _bucket_of(test_y)
    print(f"\n=== [{method_name}] Test-set coverage by TRUE price bucket ===")
    print(f"{'bucket':10s} {'n':>6s} {'mae':>8s} {'cov_raw':>9s} {'cov_calibrated':>15s} "
          f"{'width_raw':>10s} {'width_cal':>10s}")
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = true_bucket == b
        if mask.sum() == 0:
            continue
        mae = np.mean(np.abs(test_mean[mask] - test_y[mask]))
        cov_raw = np.mean((test_y[mask] >= p10_raw[mask]) & (test_y[mask] <= p90_raw[mask]))
        cov_cal = np.mean((test_y[mask] >= p10_cal[mask]) & (test_y[mask] <= p90_cal[mask]))
        width_raw = np.mean(p90_raw[mask] - p10_raw[mask])
        width_cal = np.mean(p90_cal[mask] - p10_cal[mask])
        print(f"{label:10s} {mask.sum():6d} {mae:8.4f} {cov_raw:9.3f} {cov_cal:15.3f} "
              f"{width_raw:10.3f} {width_cal:10.3f}")
    cov_raw_all = np.mean((test_y >= p10_raw) & (test_y <= p90_raw))
    cov_cal_all = np.mean((test_y >= p10_cal) & (test_y <= p90_cal))
    width_cal_all = np.mean(p90_cal - p10_cal)
    print(f"\n{'OVERALL':10s} {len(test_y):6d} {'':8s} {cov_raw_all:9.3f} {cov_cal_all:15.3f} "
          f"{'':10s} {width_cal_all:10.3f}")


def main(checkpoint_path: Path = MODEL_OUT_PATH) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config: Config = checkpoint.get("config", Config())

    data = load_data(seed=config.seed, features_path=config.features_path)
    model, likelihood = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    likelihood.load_state_dict(checkpoint["likelihood"])

    print(f"checkpoint: {checkpoint_path}  config: {config.label}")

    val_mean, val_std = _predict(model, likelihood, data["val"], config.batch_size)
    val_y = data["val"][4].numpy()
    test_mean, test_std = _predict(model, likelihood, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    p10_raw, p90_raw = test_mean - 1.2816 * test_std, test_mean + 1.2816 * test_std

    for normalized, method_name in [(False, "flat"), (True, "normalized")]:
        print(f"\n--- fitting {method_name} correction ---")
        corrections = _fit_corrections(val_mean, val_std, val_y, normalized)
        p10_cal, p90_cal = _apply_corrections(test_mean, test_std, corrections, normalized)
        _report(test_mean, test_std, test_y, p10_raw, p90_raw, p10_cal, p90_cal, method_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_OUT_PATH))
