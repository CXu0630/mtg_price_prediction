"""Walk-forward backtest of the Model B rebuild: does adding comps' moves help?

Variants (same rows, same splits, same model):
  base        teammate's Model B inputs (Manapool level left out, see build_daily.py)
  comps_gp    + comps_chg / comps_rel_level / comps_n from the deep GP's embedding
  comps_raw   + the same from plain standardized features (control)

Per test week s (7 snapshot days from s): train on t <= s-22, conformal-
calibrate on t in [s-21, s-15] -- every training/calibration target (t+14) is
known before the test week starts. Model as in mtgpricepredictor/price_move.py:
three quantile HistGradientBoosting models (10/50/90%; lr 0.05, 400 iter, min
leaf 100), training subsampled to 600k rows, crossing quantiles clipped around
the median, then CQR. Signals: BUY if the calibrated 80% range is above 0,
SELL if below; no signal when |jump_7d| > log 2 (unconfirmed jump).

    python scripts/model_b/backtest.py   # -> training/model_b/backtest_results.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from build_daily import OUT_DIR

TEST_WEEKS = ["2026-08-17", "2026-08-24", "2026-08-31"]
TRAIN_ROWS = 600_000
ALPHA = 0.2
SEEDS = (0,)
NUMS = ["log_price", "change_21d_pct", "volatility_21d", "jump_7d", "releases_before_target", "printing_age_days",
        "card_age_days", "days_since_last_printing", "printings_to_date", "log_edhrec", "reprint", "promo",
        "reserved", "gap", "fair_log", "sib_chg", "sib_div", "sib_rel_level", "set_peer",
        "lvl_ck_retail", "lvl_ck_buylist", "lvl_cardmarket"]
CATS = ["release_status", "rarity", "finish_group", "set_type"]
COMPS = ["comps_chg", "comps_rel_level", "comps_n"]
VARIANTS = {"base": [], "comps_gp": [f"{c}_gp" for c in COMPS], "comps_raw": [f"{c}_raw" for c in COMPS],
            "no_model_a": None}  # None: base minus gap / fair_log (the app's B, docs/APP_DESIGN.md)


def load() -> pd.DataFrame:
    rows = pd.read_parquet(OUT_DIR / "daily.parquet")
    for space in ("gp", "raw"):
        comps = pd.read_parquet(OUT_DIR / f"comps_{space}.parquet").rename(columns={c: f"{c}_{space}" for c in COMPS})
        rows = rows.merge(comps, on=["printing_key", "snapshot_date"], how="left")
    for c in CATS:  # integer codes, fixed over the whole table so every split agrees
        rows[c] = rows[c].astype("category").cat.codes
    return rows


def fit_quantiles(X, y, cat_mask, seed):
    models = []
    for q in (0.1, 0.5, 0.9):
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                          min_samples_leaf=100, categorical_features=cat_mask, random_state=seed)
        models.append(m.fit(X, y))
    return models


def predict(models, X):
    lo, med, hi = (m.predict(X) for m in models)
    return np.minimum(lo, med), med, np.maximum(hi, med)


def conformal_q(scores, alpha=ALPHA):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, len(scores)) - 1])


def run_week(rows, week, cols, seed) -> dict:
    s = pd.Timestamp(week)
    t = rows["snapshot_date"]
    train = rows[t <= s - pd.Timedelta(days=22)]
    cal = rows[(t >= s - pd.Timedelta(days=21)) & (t <= s - pd.Timedelta(days=15))]
    test = rows[(t >= s) & (t <= s + pd.Timedelta(days=6))]
    train = train.sample(min(TRAIN_ROWS, len(train)), random_state=seed)
    cat_mask = [c in CATS for c in cols]
    models = fit_quantiles(train[cols], train["ret"], cat_mask, seed)

    lo, _, hi = predict(models, cal[cols])
    q = conformal_q(np.maximum(lo - cal["ret"], cal["ret"] - hi))
    lo, med, hi = predict(models, test[cols])
    lo, hi = lo - q, hi + q
    y = test["ret"].to_numpy()
    ok = np.abs(test["jump_7d"].fillna(0).to_numpy()) <= np.log(2)
    buy, sell = (lo > 0) & ok, (hi < 0) & ok
    return {"week": week, "n_test": len(test), "spearman": spearmanr(med, y)[0],
            "coverage80": float(np.mean((y >= lo) & (y <= hi))), "width80": float(np.mean(hi - lo)),
            "buy_n": int(buy.sum()), "buy_hit": float(np.mean(y[buy] > 0)) if buy.any() else np.nan,
            "buy_mean": float(np.mean(y[buy])) if buy.any() else np.nan,
            "sell_n": int(sell.sum()), "sell_hit": float(np.mean(y[sell] < 0)) if sell.any() else np.nan,
            "sell_mean": float(np.mean(y[sell])) if sell.any() else np.nan}


def main(only=tuple(VARIANTS)) -> int:
    rows = load()
    print(f"{len(rows):,} rows; comps coverage gp {rows['comps_chg_gp'].notna().mean():.1%}, "
          f"raw {rows['comps_chg_raw'].notna().mean():.1%}", flush=True)
    results = []
    for seed in SEEDS:
        for week in TEST_WEEKS:
            for variant, extra in VARIANTS.items():
                if variant not in only:
                    continue
                cols = ([c for c in NUMS if c not in ("gap", "fair_log")] + CATS) if extra is None else NUMS + CATS + extra
                r = run_week(rows, week, cols, seed)
                r.update(variant=variant, seed=seed)
                results.append(r)
                print(f"  {week} {variant:9s} spearman {r['spearman']:.4f} cov {r['coverage80']:.3f} "
                      f"BUY {r['buy_n']} ({r['buy_hit']:.1%}) SELL {r['sell_n']} ({r['sell_hit']:.1%})", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / f"backtest_results_{'_'.join(only)}.csv", index=False)
    print("\n=== mean over test weeks ===")
    print(res.groupby("variant")[["spearman", "coverage80", "width80", "buy_n", "buy_hit", "buy_mean",
                                  "sell_n", "sell_hit", "sell_mean"]].mean().reindex(list(only)).round(4).to_string())
    piv = res.pivot_table(index="week", columns="variant", values="spearman")
    for v in piv.columns:
        if v != "base" and "base" in piv:
            piv[f"d_{v}"] = piv[v] - piv["base"]
    print("\nSpearman by week (and delta vs base):")
    print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(tuple(sys.argv[1:]) or tuple(VARIANTS)))
