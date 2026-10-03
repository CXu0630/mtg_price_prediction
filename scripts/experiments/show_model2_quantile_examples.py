"""Prints quantile-head Model 2 predictions for a handful of real
(held-out test-split) cards spanning the price range -- mirrors
scripts/show_model2_examples.py (the GP version), so the two can be
compared card-by-card, not just via aggregate metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_model2_quantile import QuantileConfig, MODEL_OUT_PATH, build_model, load_data, _predict
from calibrate_model2 import PRICE_BUCKET_LABELS, _bucket_of, _conformal_quantile

DATASET_PATH = Path(__file__).resolve().parents[2] / "mtg-card-market-dataset" / "dataset.parquet"
ALPHA = 0.2
N_PER_BUCKET = 3


def main(checkpoint_path: Path = MODEL_OUT_PATH) -> int:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config: QuantileConfig = checkpoint.get("config", QuantileConfig())

    data = load_data(seed=config.seed, features_path=config.features_path)
    model = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    print(f"checkpoint: {checkpoint_path.name}  config: {config.label}\n")

    features_df = pd.read_parquet(config.features_path, columns=["__group_oracle_id", "__printing_key"])
    group_key = features_df["__group_oracle_id"].to_numpy()
    from train_model2 import _split
    _, val_idx, test_idx = _split(group_key, config.seed)

    val_preds = _predict(model, data["val"], config.batch_size)
    val_y = data["val"][4].numpy()
    val_p10, val_p90 = val_preds[:, 0], val_preds[:, 2]
    val_bucket = _bucket_of(val_preds[:, 1])
    corrections = {}
    for b in range(len(PRICE_BUCKET_LABELS)):
        mask = val_bucket == b
        if mask.sum() == 0:
            continue
        scores = np.maximum(val_p10[mask] - val_y[mask], val_y[mask] - val_p90[mask])
        corrections[b] = _conformal_quantile(scores, ALPHA)

    test_preds = _predict(model, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    p10_raw, p50, p90_raw = test_preds[:, 0], test_preds[:, 1], test_preds[:, 2]
    correction_per_row = np.array([corrections.get(b, 0.0) for b in _bucket_of(p50)])
    p10_cal, p90_cal = p10_raw - correction_per_row, p90_raw + correction_per_row

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
            lo, mid, hi = np.exp(p10_cal[i]), np.exp(p50[i]), np.exp(min(p90_cal[i], 20.0))
            print(f"  {row['name']:35s} ({row['set']}, {row['rarity']}, {row['finish']:8s})  "
                  f"true=${true_price:9,.2f}   predicted=[${lo:8,.2f} - ${hi:8,.2f}], median=${mid:8,.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_OUT_PATH))
