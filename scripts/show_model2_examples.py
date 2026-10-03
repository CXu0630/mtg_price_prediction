"""Prints Model 2 predictions for a handful of real (held-out test-split)
cards spanning the price range, so the numbers in SPECS.md are checkable
against actual named cards, not just aggregate metrics. Uses the current
best checkpoint (SPECS.md S4.2: full architecture + graduated high-price
oversampling, no low-price cap) plus its group-conditional calibration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_model2 import MODEL_OUT_PATH, Config, build_model, load_data, _predict, _split
from calibrate_model2 import PRICE_BUCKET_LABELS, _bucket_of, _conformal_quantile

DEFAULT_CHECKPOINT = MODEL_OUT_PATH  # training/model2.pt, the best config (train_model2.BEST_CONFIG)
DATASET_PATH = Path(__file__).resolve().parent.parent / "mtg-card-market-dataset" / "dataset.parquet"
ALPHA = 0.2
N_PER_BUCKET = 3


def main(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config: Config = checkpoint.get("config", Config())

    data = load_data(seed=config.seed, features_path=config.features_path)
    model, likelihood = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    likelihood.load_state_dict(checkpoint["likelihood"])
    print(f"checkpoint: {checkpoint_path.name}  config: {config.label}\n")

    # Recompute the same split to recover which real cards ended up in
    # val/test (features_model2.parquet's row order, same seed as training).
    features_df = pd.read_parquet(config.features_path, columns=["__group_oracle_id", "__printing_key"])
    group_key = features_df["__group_oracle_id"].to_numpy()
    _, val_idx, test_idx = _split(group_key, config.seed)

    # --- calibration (same as calibrate_model2.py) ---
    val_mean, val_std = _predict(model, likelihood, data["val"], config.batch_size)
    val_y = data["val"][4].numpy()
    val_p10, val_p90 = val_mean - 1.2816 * val_std, val_mean + 1.2816 * val_std
    val_bucket = _bucket_of(val_mean)
    corrections = {}
    for b in range(len(PRICE_BUCKET_LABELS)):
        mask = val_bucket == b
        if mask.sum() == 0:
            continue
        scores = np.maximum(val_p10[mask] - val_y[mask], val_y[mask] - val_p90[mask])
        corrections[b] = _conformal_quantile(scores, ALPHA)

    # --- test predictions ---
    test_mean, test_std = _predict(model, likelihood, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    p10_raw, p90_raw = test_mean - 1.2816 * test_std, test_mean + 1.2816 * test_std
    correction_per_row = np.array([corrections.get(b, 0.0) for b in _bucket_of(test_mean)])
    p10_cal = p10_raw - correction_per_row
    p90_cal = p90_raw + correction_per_row

    cards = pd.read_parquet(DATASET_PATH, columns=["printing_key", "name", "set", "rarity", "finish", "price_avg"])
    cards = cards.set_index("printing_key")
    printing_keys = features_df["__printing_key"].to_numpy()[test_idx]

    true_bucket = _bucket_of(test_y)
    rng = np.random.default_rng(0)
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        idx_in_bucket = np.flatnonzero(true_bucket == b)
        if len(idx_in_bucket) == 0:
            continue
        pick = rng.choice(idx_in_bucket, size=min(N_PER_BUCKET, len(idx_in_bucket)), replace=False)
        print(f"=== {label} (n={len(idx_in_bucket)} in test) ===")
        for i in pick:
            key = printing_keys[i]
            row = cards.loc[key]
            true_price = np.exp(test_y[i])
            lo, mid, hi = np.exp(p10_cal[i]), np.exp(test_mean[i]), np.exp(p90_cal[i])
            print(f"  {row['name']:35s} ({row['set']}, {row['rarity']}, {row['finish']:8s})  "
                  f"true=${true_price:9,.2f}   predicted=[${lo:8,.2f} - ${hi:8,.2f}], median=${mid:8,.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT))
