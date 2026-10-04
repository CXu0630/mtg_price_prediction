"""Experiment 4: CatBoost (Model 1) plus the teammate's Model A features -- the
same columns as exp_gp_model_a_features.py, with set price/chase level
computed from CatBoost's own *fit* split only (calibration and test rows never
contribute a price). 3 CatBoost seeds per variant, same split:

  cb_baseline_3s            the published config
  cb_plus_a                 + every Model A feature
  cb_plus_a_no_setprice     minus set price/chase level (the brand-new-set case)

    python scripts/experiments/exp_catboost_model_a_features.py   # ~9 runs x ~5 min
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _gp_common import ROOT, print_comparison, record
from exp_gp_model_a_features import SET_PRICE_COLS, model_a_features
import calibrate_model1

SEEDS = (0, 1, 2)


def main() -> int:
    ds = pd.read_parquet(ROOT / "mtg-card-market-dataset" / "dataset.parquet", columns=["printing_key", "price_avg"])
    priced = ds[ds["price_avg"].notna()].reset_index(drop=True)  # features.parquet's row order
    target = pd.read_parquet(ROOT / "training" / "features.parquet", columns=["__target_log_price_avg"])
    assert np.allclose(np.log(priced["price_avg"]), target["__target_log_price_avg"]), "row order mismatch"
    keys, log_price = priced["printing_key"], target["__target_log_price_avg"].to_numpy()

    def extra(fit_mask, drop=()):
        cols = model_a_features(keys, log_price, fit_mask).drop(columns=list(drop))
        return cols.astype(float)

    variants = {
        "cb_baseline_3s": None,
        "cb_plus_a": lambda m: extra(m),
        "cb_plus_a_no_setprice": lambda m: extra(m, SET_PRICE_COLS),
    }
    for label, fn in variants.items():
        runs = []
        for seed in SEEDS:
            m = calibrate_model1.main(save=False, extra_features=fn, random_seed=seed)
            print(f"  [{label} seed {seed}] MAE {m['overall']['mae_p50']:.3f} | cov $10-100 "
                  f"{m['$10-$100']['coverage_calibrated']:.3f} $100+ {m['$100+']['coverage_calibrated']:.3f}", flush=True)
            runs.append(m)
        rename = {"mae_p50": "mae", "coverage_calibrated": "coverage", "width_calibrated": "width", "n": "n"}
        record(label, "catboost", [{b: {rename[k]: v for k, v in r.items() if k in rename} for b, r in run_.items()}
                                   for run_ in runs])
    print_comparison(["cb_baseline_3s", "cb_plus_a_no_setprice", "cb_plus_a"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
