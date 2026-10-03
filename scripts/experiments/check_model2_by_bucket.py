"""Bucketed evaluation of a Model 2 checkpoint's test predictions, same
convention as scripts/check_rebalance_effect.py -- aggregate metrics can
hide bucket-specific problems (this is exactly how Model 1's $10-100/$100+
ceiling was found, not from the aggregate number).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import sys

import numpy as np
import pandas as pd
import torch

from train_model2 import Config, MODEL_OUT_PATH, build_model, load_data, _predict

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def main(checkpoint_path=MODEL_OUT_PATH) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config: Config = checkpoint.get("config", Config())

    data = load_data(seed=config.seed, features_path=config.features_path)
    model, likelihood = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    likelihood.load_state_dict(checkpoint["likelihood"])

    test_mean, test_std = _predict(model, likelihood, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    q10 = test_mean - 1.2816 * test_std
    q50 = test_mean
    q90 = test_mean + 1.2816 * test_std

    rows = []
    for (lo, hi), label in zip(PRICE_BUCKETS, PRICE_BUCKET_LABELS):
        mask = (test_y >= lo) & (test_y < hi)
        n = mask.sum()
        if n == 0:
            continue
        coverage = np.mean((test_y[mask] >= q10[mask]) & (test_y[mask] <= q90[mask]))
        pinball = np.mean([_pinball_loss(test_y[mask], q, ql) for q, ql in
                            [(q10[mask], 0.1), (q50[mask], 0.5), (q90[mask], 0.9)]])
        mae = np.mean(np.abs(q50[mask] - test_y[mask]))
        rows.append({"bucket": label, "n": n, "coverage_pct": coverage * 100, "mae_log": mae, "avg_pinball": pinball})

    result = pd.DataFrame(rows)
    print(f"checkpoint: {checkpoint_path}  config: {config.label}")
    print(result.to_string(index=False))
    overall_cov = np.mean((test_y >= q10) & (test_y <= q90))
    print(f"\noverall: n={len(test_y)}, coverage={overall_cov*100:.1f}%")
    return 0


if __name__ == "__main__":
    from pathlib import Path
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_OUT_PATH))
