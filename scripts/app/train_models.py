"""Retrains the app's models from the refreshed prices (scripts/app/refresh_prices.py):

  Model A  fair value on the newest day's TCGplayer prices (OOF-calibrated 80% ranges)
  Model B  14-day TCGplayer move, every snapshot day with a known outcome
  Model C  14-day CK buylist move, rows with a trusted CK buylist
  (the deep GP backend is the published CXu0630/mtg-price-deep-gp; not retrained here)

Writes models/app/{model_a,model_b,model_c}.joblib, data/app/live_rows.parquet (the
newest day's market rows, for scoring and custom-card peers) and models/app/manifest.json.

    python scripts/app/train_models.py            # ~15-25 min on a laptop
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mtg_price_prediction import market
from mtg_price_prediction.fair_value import FairValueModel, with_history

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
PRICES = ROOT / "data" / "prices" / "price_history.parquet"
MODELS = ROOT / "models" / "app"
DATA = ROOT / "data" / "app"


def timed(label, fn):
    t = time.time()
    print(f"{label}...", flush=True)
    out = fn()
    print(f"  [{label}: {time.time() - t:.0f}s]", flush=True)
    return out


def priced_today(dataset: pd.DataFrame, px: market.Prices, uni: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """Newest-day (causal-cleaned) TCGplayer price for every dataset printing; specialty
    foils read their scryfall_id's foil series."""
    key = uni["scryfall_id"] + "|" + uni["ph_finish"]
    by_key = pd.Series(px.tcg_feat[as_of].to_numpy(), index=key.to_numpy())
    ph = dataset["finish"].map(market.PH_FINISH).fillna("foil")
    return (dataset["scryfall_id"] + "|" + ph).map(by_key)


def main() -> int:
    MODELS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(DATASET)
    uni = market.universe(dataset)
    px = timed("load prices", lambda: market.Prices.load(PRICES, uni))
    as_of = px.tcg_feat.columns[-1]
    events = dataset[["oracle_id", "released_at"]]

    # --- Model A ---
    rows_a = dataset.assign(price=priced_today(dataset, px, uni, as_of).to_numpy())
    rows_a = rows_a[(rows_a["price"] > 0) & ~rows_a["type_line"].fillna("").str.contains("Basic Land")]
    rows_a = with_history(rows_a.reset_index(drop=True), events, as_of)
    model_a = timed(f"Model A ({len(rows_a):,} priced printings, as of {as_of.date()})",
                    lambda: FairValueModel.train(rows_a, dataset, str(as_of.date())))
    print(f"  OOF log-MAE {model_a.oof_mae:.3f}, OOF 80% coverage {model_a.oof_coverage:.3f}, q {model_a.q:.3f}")
    joblib.dump(model_a, MODELS / "model_a.joblib")

    # --- Models B and C ---
    rows = timed("daily market rows", lambda: market.build_rows(uni, px))
    live = rows[rows["snapshot_date"] == as_of]
    live.to_parquet(DATA / "live_rows.parquet", index=False)
    print(f"  {len(rows):,} rows over {rows['snapshot_date'].nunique()} days; {len(live):,} live rows")
    model_b = timed("Model B", lambda: market.model_b(rows).fit(rows))
    rows_c = rows[rows["ck_buylist"].notna()]
    model_c = timed(f"Model C ({len(rows_c):,} rows with a trusted CK buylist)", lambda: market.model_c(rows_c).fit(rows_c))
    joblib.dump(model_b, MODELS / "model_b.joblib")
    joblib.dump(model_c, MODELS / "model_c.joblib")

    manifest = {
        "price_date": str(as_of.date()),
        "model_a": {"as_of": model_a.as_of, "train_rows": len(rows_a), "oof_log_mae": round(model_a.oof_mae, 4),
                    "oof_coverage80": round(model_a.oof_coverage, 4), "q": round(model_a.q, 4),
                    "n_features": len(model_a.columns)},
        "model_b": {"cutoff": model_b.cutoff, "cqr_margin": round(model_b.q, 4), "features": model_b.columns},
        "model_c": {"cutoff": model_c.cutoff, "cqr_margin": round(model_c.q, 4), "features": model_c.columns},
    }
    (MODELS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"-> {MODELS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
