"""Group-conditional conformal calibration for the quantile-head Model 2
variant (scripts/train_model2_quantile.py) -- same method as
scripts/calibrate_model2.py, simpler here since q10/q90 are the model's
own direct quantile outputs (no mean+z*std approximation needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import sys
from pathlib import Path

import numpy as np
import torch

from train_model2_quantile import QuantileConfig, MODEL_OUT_PATH, build_model, load_data, _predict
from calibrate_model2 import PRICE_BUCKET_LABELS, _bucket_of, _conformal_quantile

ALPHA = 0.2


def main(checkpoint_path: Path = MODEL_OUT_PATH) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config: QuantileConfig = checkpoint.get("config", QuantileConfig())

    data = load_data(seed=config.seed, features_path=config.features_path)
    model = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    print(f"checkpoint: {checkpoint_path}  config: {config.label}")

    val_preds = _predict(model, data["val"], config.batch_size)
    val_y = data["val"][4].numpy()
    val_p10, val_p50, val_p90 = val_preds[:, 0], val_preds[:, 1], val_preds[:, 2]
    val_bucket = _bucket_of(val_p50)

    corrections = {}
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = val_bucket == b
        if mask.sum() == 0:
            continue
        if mask.sum() < 20:
            print(f"  WARNING: only {mask.sum()} calibration points in {label}, correction may be unstable")
        scores = np.maximum(val_p10[mask] - val_y[mask], val_y[mask] - val_p90[mask])
        corrections[b] = _conformal_quantile(scores, ALPHA)
        print(f"  calibration bucket {label:10s} n={mask.sum():5d}  correction={corrections[b]:+.4f}")

    test_preds = _predict(model, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    p10_raw, p50, p90_raw = test_preds[:, 0], test_preds[:, 1], test_preds[:, 2]

    test_bucket_by_pred = _bucket_of(p50)
    correction_per_row = np.array([corrections.get(b, 0.0) for b in test_bucket_by_pred])
    p10_cal = p10_raw - correction_per_row
    p90_cal = p90_raw + correction_per_row

    true_bucket = _bucket_of(test_y)
    print("\n=== Test-set coverage by TRUE price bucket ===")
    print(f"{'bucket':10s} {'n':>6s} {'mae':>8s} {'cov_raw':>9s} {'cov_calibrated':>15s} "
          f"{'width_raw':>10s} {'width_cal':>10s} {'width_cv':>9s}")
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = true_bucket == b
        if mask.sum() == 0:
            continue
        mae = np.mean(np.abs(p50[mask] - test_y[mask]))
        cov_raw = np.mean((test_y[mask] >= p10_raw[mask]) & (test_y[mask] <= p90_raw[mask]))
        cov_cal = np.mean((test_y[mask] >= p10_cal[mask]) & (test_y[mask] <= p90_cal[mask]))
        width_raw = p90_raw[mask] - p10_raw[mask]
        width_cal = p90_cal[mask] - p10_cal[mask]
        cv = width_cal.std() / width_cal.mean()
        print(f"{label:10s} {mask.sum():6d} {mae:8.4f} {cov_raw:9.3f} {cov_cal:15.3f} "
              f"{width_raw.mean():10.3f} {width_cal.mean():10.3f} {cv:9.3f}")

    cov_raw_all = np.mean((test_y >= p10_raw) & (test_y <= p90_raw))
    cov_cal_all = np.mean((test_y >= p10_cal) & (test_y <= p90_cal))
    width_cal_all = (p90_cal - p10_cal)
    print(f"\n{'OVERALL':10s} {len(test_y):6d} {'':8s} {cov_raw_all:9.3f} {cov_cal_all:15.3f} "
          f"{'':10s} {width_cal_all.mean():10.3f} {width_cal_all.std()/width_cal_all.mean():9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_OUT_PATH))
