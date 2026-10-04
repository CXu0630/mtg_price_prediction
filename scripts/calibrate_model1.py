"""Magnitude-aware conformal calibration for Model 1's quantile intervals.

Finding from the rebalance check (scripts/experiments/check_rebalance_effect.py):
coverage of the nominal 80% [p10, p90] interval is badly uneven across the
price range -- ~70-76% for cheap/mid cards but only 59-61% at $10-$100 and
42-43% at $100+. A single global correction (standard split conformal
prediction / CQR, Romano-Patterson-Candes 2019) only guarantees coverage
*on average* over the whole population; with miscalibration this uneven,
a global correction would be dominated by the huge cheap-card population
and barely touch the high end where it's needed most.

So this does **group-conditional (Mondrian) conformal calibration**:
compute a separate interval-widening correction per price bucket, bucketed
by the model's *own predicted* median (not true price, which isn't known
at inference time) -- deployable, and directly targets the heteroscedastic
miscalibration found.

Three-way split, all by oracle_id group (no printing of the same card
concept crossing splits):
  - fit (~68% of all rows): trained on, after the same bulk-bin cap as
    train_model1.py (only the fit set is rebalanced)
  - calibration (~12%): NOT rebalanced -- stays at the natural distribution,
    so it's exchangeable with the untouched test set, which conformal
    validity depends on. Used only to compute the per-bucket correction.
  - test (20%): completely untouched, used only for final evaluation.

This is also the script that produces the *best* Model 1: it trains the
tuned tier-A (fundamentals-only) config and saves it to
training/model1.cbm, with its per-bucket corrections, feature list and
test metrics in training/model1_calibration.json (consumed by
scripts/export_models.py).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupShuffleSplit

from train_model1 import (
    ROOT, FEATURES_PATH, TARGET_COL, GROUP_COL, QUANTILES,
    ABLATIONS, _feature_cols_for, _cap_bins, RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS,
)

ALPHA = 0.2  # 1 - 0.8, for an 80% interval
MODEL_OUT_PATH = ROOT / "training" / "model1.cbm"
CALIBRATION_OUT_PATH = ROOT / "training" / "model1_calibration.json"
HYPERPARAMS = {"iterations": 4000, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3, "random_seed": 0}

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]


def _bucket_of(values: np.ndarray) -> np.ndarray:
    bucket_ids = np.zeros(len(values), dtype=int)
    for i, (lo, hi) in enumerate(PRICE_BUCKETS):
        bucket_ids[(values >= lo) & (values < hi)] = i
    return bucket_ids


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Standard finite-sample-correct split-conformal quantile level."""
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def _local_corrections(calib_median: np.ndarray, calib_scores: np.ndarray,
                        query_median: np.ndarray, alpha: float, k: int) -> np.ndarray:
    """Per-query-point correction from the k calibration points nearest in
    *predicted* median (not a hard bucket) -- window width adapts to local
    density automatically (wide where calibration data is sparse, e.g. the
    $100+ tail; narrow where it's dense), so sparse regions borrow
    statistical strength from their neighbors instead of being calibrated
    in total isolation on a handful of points."""
    order = np.argsort(calib_median)
    sorted_median = calib_median[order]
    sorted_scores = calib_scores[order]
    n = len(sorted_median)

    positions = np.searchsorted(sorted_median, query_median)
    half = k // 2
    lo = np.clip(positions - half, 0, max(n - k, 0))
    lo = np.minimum(lo, n - k) if n > k else np.zeros_like(lo)

    corrections = np.empty(len(query_median))
    for i, start in enumerate(lo):
        window = sorted_scores[start:start + k]
        corrections[i] = _conformal_quantile(window, alpha)
    return corrections


def main(drop_features: tuple[str, ...] = (), save: bool = True, extra_features=None,
         random_seed: int = HYPERPARAMS["random_seed"]) -> dict:
    """Trains + calibrates the best Model 1 config. `drop_features` removes
    columns (ablations); `extra_features(fit_mask) -> DataFrame` adds
    row-aligned columns, given a boolean mask of the fit rows so anything
    price-derived can be computed from those only; `save=False` leaves
    training/model1.* untouched. Returns the test metrics."""
    df = pd.read_parquet(FEATURES_PATH)
    y = df[TARGET_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()
    all_feature_cols = [c for c in df.columns if c not in (TARGET_COL, GROUP_COL)]

    # train_model1.py's split: 80% train+calib pool, 20% test (untouched).
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    pool_idx, test_idx = next(splitter.split(df, groups=groups))

    # Carve a calibration set out of the pool: 15% of the pool, by group.
    calib_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=1)
    fit_idx, calib_idx = next(calib_splitter.split(df.iloc[pool_idx], groups=groups[pool_idx]))
    fit_idx, calib_idx = pool_idx[fit_idx], pool_idx[calib_idx]

    n_before = len(fit_idx)
    fit_mask = np.zeros(len(df), bool)  # all fit rows, before rebalancing (for extra_features)
    fit_mask[fit_idx] = True
    keep = _cap_bins(y[fit_idx], RESAMPLE_CAP_PER_BIN, RESAMPLE_N_BINS)
    fit_idx = fit_idx[keep]
    print(f"fit: {n_before} -> {len(fit_idx)} rows (rebalanced) | "
          f"calibration: {len(calib_idx)} rows (natural dist.) | test: {len(test_idx)} rows (natural dist.)")

    cols = [c for c in _feature_cols_for(all_feature_cols, ABLATIONS["A_fundamentals"]) if c not in drop_features]
    X = df[cols].copy()
    if extra_features is not None:
        extra = extra_features(fit_mask).set_index(X.index)
        X = pd.concat([X, extra], axis=1)
        cols = cols + list(extra.columns)
    cat_features = [c for c in cols if X[c].dtype == "object" or str(X[c].dtype) == "str"]
    for c in cat_features:
        X[c] = X[c].fillna("__missing__").astype(str)

    # depth/l2_leaf_reg/learning_rate chosen by scripts/experiments/tune_model1.py's
    # search (see SPECS.md §4.1) -- early stopping against the calibration
    # split is what actually regularizes here, not depth/l2_leaf_reg
    # (those landed close to CatBoost's own defaults).
    model = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={','.join(str(q) for q in QUANTILES)}",
        **{**HYPERPARAMS, "random_seed": random_seed}, verbose=False, allow_writing_files=False,
        cat_features=cat_features,
    )
    calib_pool_for_stopping = Pool(X.iloc[calib_idx], y[calib_idx], cat_features=cat_features)
    model.fit(Pool(X.iloc[fit_idx], y[fit_idx], cat_features=cat_features),
              eval_set=calib_pool_for_stopping, early_stopping_rounds=100)
    print(f"  trained to iteration {model.get_best_iteration()}")

    # --- Calibration: per predicted-bucket correction ---
    calib_preds = model.predict(X.iloc[calib_idx])
    calib_y = y[calib_idx]
    calib_bucket = _bucket_of(calib_preds[:, 1])  # bucket by predicted median

    corrections = {}
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = calib_bucket == b
        if mask.sum() < 20:
            print(f"  WARNING: only {mask.sum()} calibration points in {label}, correction may be unstable")
        scores = np.maximum(calib_preds[mask, 0] - calib_y[mask], calib_y[mask] - calib_preds[mask, 2])
        corrections[b] = _conformal_quantile(scores, ALPHA)
        print(f"  calibration bucket {label:10s} n={mask.sum():5d}  correction={corrections[b]:+.4f}")

    # --- Evaluate on the untouched test set, both uncalibrated and calibrated ---
    test_preds = model.predict(X.iloc[test_idx])
    test_y = y[test_idx]
    test_bucket_by_pred = _bucket_of(test_preds[:, 1])
    correction_per_row = np.array([corrections[b] for b in test_bucket_by_pred])

    p10_raw, p50, p90_raw = test_preds[:, 0], test_preds[:, 1], test_preds[:, 2]
    p10_cal = p10_raw - correction_per_row
    p90_cal = p90_raw + correction_per_row

    true_bucket = _bucket_of(test_y)
    print("\n=== Test-set coverage by TRUE price bucket ===")
    print(f"{'bucket':10s} {'n':>6s} {'pinball_p50':>12s} {'cov_raw':>9s} {'cov_calibrated':>15s} "
          f"{'width_raw':>10s} {'width_cal':>10s}")
    test_metrics = {}
    for b, label in enumerate(PRICE_BUCKET_LABELS):
        mask = true_bucket == b
        if mask.sum() == 0:
            continue
        pinball50 = mean_pinball_loss(test_y[mask], p50[mask], alpha=0.5)
        cov_raw = np.mean((test_y[mask] >= p10_raw[mask]) & (test_y[mask] <= p90_raw[mask]))
        cov_cal = np.mean((test_y[mask] >= p10_cal[mask]) & (test_y[mask] <= p90_cal[mask]))
        width_raw = np.mean(p90_raw[mask] - p10_raw[mask])
        width_cal = np.mean(p90_cal[mask] - p10_cal[mask])
        print(f"{label:10s} {mask.sum():6d} {pinball50:12.4f} {cov_raw:9.3f} {cov_cal:15.3f} "
              f"{width_raw:10.3f} {width_cal:10.3f}")
        test_metrics[label] = {"n": int(mask.sum()), "pinball_p50": float(pinball50),
                               "mae_p50": float(np.mean(np.abs(p50[mask] - test_y[mask]))),
                               "coverage_raw": float(cov_raw), "coverage_calibrated": float(cov_cal),
                               "width_calibrated": float(width_cal)}

    cov_raw_all = np.mean((test_y >= p10_raw) & (test_y <= p90_raw))
    cov_cal_all = np.mean((test_y >= p10_cal) & (test_y <= p90_cal))
    print(f"\n{'OVERALL':10s} {len(test_y):6d} {'':12s} {cov_raw_all:9.3f} {cov_cal_all:15.3f}")
    test_metrics["overall"] = {"n": len(test_y), "mae_p50": float(np.mean(np.abs(p50 - test_y))),
                               "coverage_raw": float(cov_raw_all), "coverage_calibrated": float(cov_cal_all),
                               "width_calibrated": float(np.mean(p90_cal - p10_cal))}

    if not save:
        return test_metrics
    model.save_model(str(MODEL_OUT_PATH))
    CALIBRATION_OUT_PATH.write_text(json.dumps({
        "quantiles": QUANTILES,
        "hyperparameters": {**HYPERPARAMS, "best_iteration": model.get_best_iteration()},
        "feature_cols": cols,
        "cat_features": cat_features,
        "calibration_corrections": {str(b): c for b, c in corrections.items()},
        "n_fit": len(fit_idx), "n_calibration": len(calib_idx),
        "test_metrics": test_metrics,
    }, indent=2))
    print(f"\n-> {MODEL_OUT_PATH}\n-> {CALIBRATION_OUT_PATH}")

    # --- Local (k-nearest-by-predicted-median) calibration, for comparison ---
    calib_scores = np.maximum(calib_preds[:, 0] - calib_y, calib_y - calib_preds[:, 2])
    for k in (1500, 4000):
        local_corr = _local_corrections(calib_preds[:, 1], calib_scores, test_preds[:, 1], ALPHA, k)
        p10_local = p10_raw - local_corr
        p90_local = p90_raw + local_corr

        print(f"\n=== Local calibration (k={k}), by TRUE price bucket ===")
        print(f"{'bucket':10s} {'n':>6s} {'coverage':>9s} {'width':>8s}")
        for b, label in enumerate(PRICE_BUCKET_LABELS):
            mask = true_bucket == b
            if mask.sum() == 0:
                continue
            cov = np.mean((test_y[mask] >= p10_local[mask]) & (test_y[mask] <= p90_local[mask]))
            width = np.mean(p90_local[mask] - p10_local[mask])
            print(f"{label:10s} {mask.sum():6d} {cov:9.3f} {width:8.3f}")
        cov_all = np.mean((test_y >= p10_local) & (test_y <= p90_local))
        print(f"{'OVERALL':10s} {len(test_y):6d} {cov_all:9.3f}")

    return test_metrics


if __name__ == "__main__":
    main()
