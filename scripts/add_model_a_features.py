"""Adds Model A's inputs to the deep GP's training table, so the GP sees the same card attributes and
supply as the boosted-tree fair-value model (plus its own text and art embeddings).

Added columns (gp_features.MODEL_A_COLS): mana-cost detail, rules-text length, set and rarity-slot size,
products carrying the printing, max/summed pull rate, collector-only, reprint/age history as of the end
of the price window, and the set's price level / chase level. The set price columns use TRAIN-split
prices only: a training card sees its set's other training cards, a val/test card sees the set's
training cards, so no held-out price enters a feature. Serving builds the same columns with
gp_features.model_a_features and the exported set_prices.parquet.

    python scripts/add_model_a_features.py   # training/features_model2.parquet -> training/features_model2_plus_a.parquet

Run after scripts/prepare_training_data_model2.py and before scripts/train_model2.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mtg_price_prediction.gp_features import model_a_features
from train_model2 import SEED, _split

ROOT = Path(__file__).resolve().parent.parent
BASE_FEATURES = ROOT / "training" / "features_model2.parquet"
OUT = ROOT / "training" / "features_model2_plus_a.parquet"
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
AS_OF = pd.Timestamp("2026-09-20")  # end of the price_avg window; must match export_models.MODEL_A_AS_OF


def main() -> int:
    feats = pd.read_parquet(BASE_FEATURES)
    dataset = pd.read_parquet(DATASET)
    rows = dataset.set_index("printing_key").loc[feats["__printing_key"].to_numpy()].reset_index()

    train_idx, _, _ = _split(feats["__group_oracle_id"].to_numpy(), SEED)
    tr = rows.iloc[train_idx].assign(lp=feats["__target_log_price_avg"].to_numpy()[train_idx])
    set_prices = tr.groupby(["set", "oracle_id"], as_index=False)["lp"].max()

    extra = model_a_features(rows, dataset, set_prices, AS_OF)
    out = pd.concat([feats, extra.set_index(feats.index)], axis=1)
    out.to_parquet(OUT, index=False)
    print(f"added {extra.shape[1]} columns ({np.mean(extra['set_price_missing']):.1%} rows without a set price) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
