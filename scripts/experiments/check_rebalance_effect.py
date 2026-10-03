"""One-off diagnostic: does training-set rebalancing (train_model1.py's
_cap_bins) actually help where it's meant to -- the higher-value tail --
or does the worse aggregate test pinball/coverage mean it's just worse?

Trains tier A (fundamentals) twice, identical in every way except one has
the rebalanced train set and one doesn't, and reports pinball/coverage
broken out by price bucket on the *same* untouched test set for both, so
the comparison isolates the effect of rebalancing alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupShuffleSplit

from train_model1 import (
    FEATURES_PATH, TARGET_COL, GROUP_COL, QUANTILES,
    ABLATIONS, _feature_cols_for, _cap_bins, RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS,
)
from catboost import CatBoostRegressor, Pool

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]


def _train(X_train, y_train, X_test, cat_features):
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
        iterations=4000, learning_rate=0.05, depth=6, random_seed=0, verbose=False,
        cat_features=cat_features,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    model.fit(train_pool)
    return model.predict(X_test)


def _bucketed_metrics(y_test, preds, label):
    print(f"\n--- {label} ---")
    for (lo, hi), blabel in zip(PRICE_BUCKETS, PRICE_BUCKET_LABELS):
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() == 0:
            continue
        yt = y_test[mask]
        p10, p50, p90 = preds[mask, 0], preds[mask, 1], preds[mask, 2]
        pinball50 = mean_pinball_loss(yt, p50, alpha=0.5)
        coverage = np.mean((yt >= p10) & (yt <= p90))
        print(f"  {blabel:10s} n={mask.sum():6d}  pinball_p50={pinball50:.4f}  coverage_80={coverage:.3f}")


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    y = df[TARGET_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    all_feature_cols = [c for c in df.columns if c not in (TARGET_COL, GROUP_COL)]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    cols = _feature_cols_for(all_feature_cols, ABLATIONS["A_fundamentals"])
    X = df[cols].copy()
    cat_features = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype) == "str"]
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)

    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # Variant 1: no rebalancing
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    preds_unbalanced = _train(X_train, y_train, X_test, cat_features)
    _bucketed_metrics(y_test, preds_unbalanced, "unbalanced training set")

    # Variant 2: rebalanced (same cap as train_model1.py)
    keep = _cap_bins(y[train_idx], RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS)
    train_idx_r = train_idx[keep]
    X_train_r, y_train_r = X.iloc[train_idx_r], y[train_idx_r]
    preds_rebalanced = _train(X_train_r, y_train_r, X_test, cat_features)
    _bucketed_metrics(y_test, preds_rebalanced, "rebalanced training set")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
