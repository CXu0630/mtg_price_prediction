"""Does the full tag feature set (tier C) help specifically in the buckets
where tier A struggles (SPECS.md/conversation: aggregate ablation numbers
were nearly flat, but that's exactly the trap the rebalancing check caught
-- an aggregate metric dominated by cheap cards can hide a real effect in
one bucket)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupShuffleSplit

from train_model1 import (
    FEATURES_PATH, TARGET_COL, GROUP_COL, QUANTILES,
    ABLATIONS, _feature_cols_for, _cap_bins, RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS,
)
from check_rebalance_effect import PRICE_BUCKETS, PRICE_BUCKET_LABELS, _bucketed_metrics


def _train_and_eval(name, tag_groups, df, all_feature_cols, train_idx, test_idx, y):
    cols = _feature_cols_for(all_feature_cols, tag_groups)
    X = df[cols].copy()
    cat_features = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype) == "str"]
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)

    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
        iterations=4000, learning_rate=0.05, depth=6, random_seed=0, verbose=False,
        cat_features=cat_features,
    )
    model.fit(Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_features))
    preds = model.predict(X.iloc[test_idx])
    _bucketed_metrics(y[test_idx], preds, f"{name} ({len(cols)} features)")


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    y = df[TARGET_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    all_feature_cols = [c for c in df.columns if c not in (TARGET_COL, GROUP_COL)]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    keep = _cap_bins(y[train_idx], RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS)
    train_idx = train_idx[keep]

    for name in ["A_fundamentals", "C_full"]:
        _train_and_eval(name, ABLATIONS[name], df, all_feature_cols, train_idx, test_idx, y)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
