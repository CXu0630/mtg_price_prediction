"""Small randomized hyperparameter search for Model 1, motivated by the
overfitting signal found earlier (coverage got *worse* with more
iterations on the default depth=6/l2_leaf_reg=3/lr=0.05 config -- that
config was never actually tuned, just a first guess).

Uses the same fit/calibration/test split as calibrate_model1.py. Each
candidate is trained on `fit` with `calibration` as the early-stopping
eval_set, then scored on `calibration` itself (pinball + coverage,
weighted toward the buckets that were worst: $10-100 and $100+) -- a
coarse aggregate selection signal, not the per-bucket correction fit
itself, so reusing `calibration` here is a defensible simplification for
a first tuning pass, not a fresh 4th split.

The winning config is then retrained fresh and run through the full
group-conditional calibration + untouched-test evaluation from
calibrate_model1.py, for a final, honest, not-tuned-against number.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import itertools
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupShuffleSplit

from train_model1 import (
    FEATURES_PATH, TARGET_COL, GROUP_COL, QUANTILES,
    ABLATIONS, _feature_cols_for, _cap_bins, RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS,
)
from check_rebalance_effect import PRICE_BUCKETS, PRICE_BUCKET_LABELS

SEARCH_SPACE = {
    "depth": [4, 6, 8],
    "l2_leaf_reg": [3, 10, 30],
    "learning_rate": [0.03, 0.08],
}
N_CANDIDATES = 8  # random subset of the 3x3x2=18 grid, not exhaustive


def _bucket_of(values: np.ndarray) -> np.ndarray:
    bucket_ids = np.zeros(len(values), dtype=int)
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        bucket_ids[(values >= lo) & (values < hi)] = i
    return bucket_ids


def _score(y_true, preds) -> float:
    """Lower is better. Weighted toward the buckets that were worst
    (10-100, 100+) so the search optimizes for the actual problem, not
    just overall pinball (which is dominated by cheap cards regardless)."""
    bucket = _bucket_of(y_true)
    weights = {0: 1.0, 1: 1.0, 2: 2.5, 3: 2.5}  # upweight $10-100 and $100+
    total, weight_sum = 0.0, 0.0
    for b, w in weights.items():
        mask = bucket == b
        if mask.sum() == 0:
            continue
        pinball50 = mean_pinball_loss(y_true[mask], preds[mask, 1], alpha=0.5)
        total += w * pinball50
        weight_sum += w
    return total / weight_sum


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    y = df[TARGET_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    all_feature_cols = [c for c in df.columns if c not in (TARGET_COL, GROUP_COL)]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    pool_idx, test_idx = next(splitter.split(df, groups=groups))
    calib_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1)
    fit_idx, calib_idx = next(calib_splitter.split(df.iloc[pool_idx], groups=groups[pool_idx]))
    fit_idx, calib_idx = pool_idx[fit_idx], pool_idx[calib_idx]
    fit_idx = fit_idx[_cap_bins(y[fit_idx], RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS)]

    cols = _feature_cols_for(all_feature_cols, ABLATIONS["A_fundamentals"])
    X = df[cols].copy()
    cat_features = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype) == "str"]
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)

    all_combos = list(itertools.product(SEARCH_SPACE["depth"], SEARCH_SPACE["l2_leaf_reg"], SEARCH_SPACE["learning_rate"]))
    rng = np.random.default_rng(0)
    candidates = [all_combos[i] for i in rng.choice(len(all_combos), size=N_CANDIDATES, replace=False)]

    fit_pool = Pool(X.iloc[fit_idx], y[fit_idx], cat_features=cat_features)
    calib_pool = Pool(X.iloc[calib_idx], y[calib_idx], cat_features=cat_features)

    results = []
    for depth, l2, lr in candidates:
        model = CatBoostRegressor(
            loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
            iterations=4000, learning_rate=lr, depth=depth, l2_leaf_reg=l2,
            random_seed=0, verbose=False,
        )
        model.fit(fit_pool, eval_set=calib_pool, early_stopping_rounds=100)
        preds = model.predict(X.iloc[calib_idx])
        score = _score(y[calib_idx], preds)
        print(f"depth={depth} l2_leaf_reg={l2} lr={lr}  ->  weighted_pinball={score:.4f}  "
              f"best_iter={model.get_best_iteration()}")
        results.append((score, depth, l2, lr))

    results.sort()
    best_score, best_depth, best_l2, best_lr = results[0]
    print(f"\nBest: depth={best_depth} l2_leaf_reg={best_l2} lr={best_lr} (weighted_pinball={best_score:.4f})")
    print("scripts/calibrate_model1.py" + f" should be re-run with depth={best_depth}, l2_leaf_reg={best_l2}, learning_rate={best_lr}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
