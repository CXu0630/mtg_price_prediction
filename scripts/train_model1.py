"""Model 1 (GBDT baseline) -- CatBoost quantile regression on
training/features.parquet, per SPECS.md §4.1.

Target: log(price_avg) (SPECS.md §3's cohort-relative idea was walked back
-- see conversation -- Model 1 is a direct regressor with no comps
reweighting step, so a cohort baseline would just add complexity without
the reconstruction benefit it has for Model 2's kernel approach).

Quantile levels: p10/p50/p90 (an 80% interval + median) -- not yet
formally decided in SPECS.md §8, this is a placeholder default, reasonable
and standard, not a considered final choice.

Split: GroupShuffleSplit on oracle_id (SPECS.md §6 -- different printings
of the same card are too similar to let leak across train/test).

Ablation ladder (SPECS.md §4.1's "fundamentals -> + art tags -> full"):
  A. fundamentals only -- mechanics/rarity/legality/demand/scarcity, no
     crowdsourced tag columns at all
  B. A + art_tags
  C. full -- A + art_tags + keywords + oracle_tags + type_tags +
     promo_types + frame_effects

Training-set rebalancing: the raw price distribution is dominated by bulk
cards -- two histogram bins alone ($0.19-$0.31 and $0.31-$0.50) hold 35%
of all 142,012 rows -- which both skews the loss toward getting bulk
commons exactly right (not the interesting part of "price range
prediction") and gives the model little reason to work on the sparser,
harder, higher-value tail. Fixed via a per-bin row cap on log(price_avg)
(RESAMPLE_CAP_PER_BIN), **applied to the training split only** -- the test
set stays at its natural distribution so evaluation numbers still mean
"how good on the real population of cards," not the rebalanced one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "training" / "features.parquet"

QUANTILES = [0.1, 0.5, 0.9]
TARGET_COL = "__target_log_price_avg"
GROUP_COL = "__group_oracle_id"

TAG_PREFIXES = {
    "art_tags": "art_tags__",
    "keywords": "keywords__",
    "oracle_tags": "oracle_tags__",
    "type_tags": "type_tags__",
    "promo_types": "promo_types__",
    "frame_effects": "frame_effects__",
}

ABLATIONS = {
    "A_fundamentals": [],
    "B_fundamentals_plus_art": ["art_tags"],
    "C_full": list(TAG_PREFIXES),
}

# Chosen from the actual per-bin counts (checked this session): the two
# dominant bulk-card bins hold ~25-26k rows each, while the whole $1-$15
# range (the more practically-relevant part of the distribution) already
# sits at roughly 5k-8.5k rows/bin -- capping at 8,000 flattens the two
# oversized cheap bins down to about that same level without artificially
# touching bins that are already sparse.
RESAMPLE_CAP_PER_BIN = 8000
RESAMPLE_N_BINS = 30


def _cap_bins(y_train: np.ndarray, cap: int, n_bins: int, seed: int = 0) -> np.ndarray:
    """Row positions (into y_train) to keep after capping each log-price
    bin at `cap` rows, randomly subsampling bins that exceed it. Bin edges
    come from the training split only, not the full dataset."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(y_train.min(), y_train.max(), n_bins + 1)
    bin_ids = np.digitize(y_train, edges[1:-1])

    keep = []
    for b in range(n_bins):
        in_bin = np.flatnonzero(bin_ids == b)
        if len(in_bin) > cap:
            in_bin = rng.choice(in_bin, size=cap, replace=False)
        keep.append(in_bin)
    return np.sort(np.concatenate(keep))


def _feature_cols_for(all_cols: list[str], tag_groups: list[str]) -> list[str]:
    tag_cols = {c for group in TAG_PREFIXES for c in all_cols if c.startswith(TAG_PREFIXES[group])}
    included_tag_cols = {c for group in tag_groups for c in all_cols if c.startswith(TAG_PREFIXES[group])}
    return [c for c in all_cols if c not in tag_cols or c in included_tag_cols]


def _train_one(X_train, y_train, X_test, y_test, cat_features) -> dict:
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
        iterations=4000,
        learning_rate=0.05,
        depth=6,
        random_seed=0,
        verbose=False,
        cat_features=cat_features,
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

    preds = model.predict(X_test)  # (n, len(QUANTILES))
    pinball = {
        f"pinball_p{int(q*100)}": mean_pinball_loss(y_test, preds[:, i], alpha=q)
        for i, q in enumerate(QUANTILES)
    }
    p10, p90 = preds[:, 0], preds[:, -1]
    coverage_80 = float(np.mean((y_test >= p10) & (y_test <= p90)))
    mean_width = float(np.mean(p90 - p10))
    return {**pinball, "coverage_80pct_interval": coverage_80, "mean_log_interval_width": mean_width,
            "n_test": len(y_test), "best_iteration": model.get_best_iteration()}


def main() -> int:
    df = pd.read_parquet(FEATURES_PATH)
    y = df[TARGET_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    all_feature_cols = [c for c in df.columns if c not in (TARGET_COL, GROUP_COL)]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    print(f"train: {len(train_idx)} rows, test: {len(test_idx)} rows "
          f"({df[GROUP_COL].iloc[train_idx].nunique()} / {df[GROUP_COL].iloc[test_idx].nunique()} distinct card concepts)")

    n_before = len(train_idx)
    keep = _cap_bins(y[train_idx], RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS)
    train_idx = train_idx[keep]
    print(f"rebalanced train: {n_before} -> {len(train_idx)} rows "
          f"(bulk-bin cap = {RESAMPLE_CAP_PER_BIN}/bin over {RESAMPLE_N_BINS} bins; test set untouched)")

    results = {}
    for name, tag_groups in ABLATIONS.items():
        cols = _feature_cols_for(all_feature_cols, tag_groups)
        X = df[cols]
        cat_features = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype) == "str"]

        X = X.copy()
        for c in cat_features:
            X[c] = X[c].fillna("__missing__").astype(str)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n=== {name} ({len(cols)} features) ===")
        metrics = _train_one(X_train, y_train, X_test, y_test, cat_features)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        results[name] = metrics

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
