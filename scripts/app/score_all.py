"""Scores every existing printing with the app's models and writes the static data
the frontend reads (docs/APP_DESIGN.md §6):

  web/public/data/meta.json        price date, model versions + quality, counts
  web/public/data/index.json       column-wise search index of every printing
  web/public/data/sets/<set>.json  full prediction rows per set (loaded on demand),
                                   each with its top GP comps inlined

Needs scripts/app/refresh_prices.py, train_models.py and build_gp_static.py first.

    python scripts/app/score_all.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mtg_price_prediction import market
from mtg_price_prediction.scoring import AppModels, score_existing

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
PRICES = ROOT / "data" / "prices" / "price_history.parquet"
LIVE = ROOT / "data" / "app" / "live_rows.parquet"
MODELS = ROOT / "models" / "app"
GP_DIR = ROOT / "models" / "deep_gp"
OUT = ROOT / "web" / "public" / "data"
N_COMPS_SHOWN = 5

MONEY = ["price_today", "ck_retail", "ck_buylist", "fair", "fair_lo", "fair_hi", "fair_gp", "fair_gp_lo",
         "fair_gp_hi", "pred_price", "price_lo", "price_hi", "buylist_pred", "buylist_lo", "buylist_hi"]
FIELDS = ["printing_key", "scryfall_id", "name", "set", "set_name", "collector_number", "finish", "rarity",
          "status", "price_date", *MONEY, "cardmarket_eur", "signal", "unconfirmed_jump", "comps"]


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 2)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _treatment(row) -> str:
    """Human-readable visual treatments for one printing's search entry."""
    effects = row["frame_effects"]
    tags = list(effects) if isinstance(effects, (list, tuple, np.ndarray)) else []
    labels = {"extendedart": "extended art", "fullart": "full art", "colorshifted": "color shifted"}
    tags = [labels.get(tag, tag) for tag in tags]
    if row["borderless"]:
        tags.insert(0, "borderless")
    elif row["border_color"] not in (None, "black"):
        tags.insert(0, f'{row["border_color"]} border')
    if row["full_art"]:
        tags.append("full art")
    return ", ".join(dict.fromkeys(tags))


def main() -> int:
    dataset = pd.read_parquet(DATASET)
    models = AppModels.load(MODELS, GP_DIR)
    live = pd.read_parquet(LIVE)
    as_of = pd.Timestamp(models.manifest["price_date"])
    uni = market.universe(dataset)
    px = market.Prices.load(PRICES, uni)
    series = uni["scryfall_id"] + "|" + uni["ph_finish"]
    today = pd.Series(px.tcg_feat[as_of].to_numpy(), index=series.to_numpy())
    tcg_today = (dataset["scryfall_id"] + "|" + dataset["finish"].map(market.PH_FINISH).fillna("foil")).map(today)
    gp_static = pd.read_parquet(MODELS / "gp_static.parquet")

    scored = score_existing(dataset, models, live, gp_static, tcg_today, as_of)
    by_key = scored.set_index("printing_key")
    comp_info = by_key[["name", "set", "finish", "price_today", "oracle_id"]]

    def comps_summary(keys):
        """Top comps that are distinct cards with a price today ("priced like these")."""
        if keys is None or len(keys) == 0:
            return []
        rows = comp_info.reindex(list(keys))
        rows = rows[rows["price_today"].notna()].drop_duplicates("oracle_id").head(N_COMPS_SHOWN)
        return [{"printing_key": k, "name": r["name"], "set": r["set"], "finish": r["finish"],
                 "price_today": _clean(r["price_today"])} for k, r in rows.iterrows()]

    scored["comps"] = scored["comps"].map(comps_summary)

    # Clear only what this script writes (models/ is export_web_models.py's).
    shutil.rmtree(OUT / "sets", ignore_errors=True)
    (OUT / "sets").mkdir(parents=True)
    for set_code, g in scored.groupby("set"):
        records = [{f: _clean(r[f]) for f in FIELDS if f in r} for r in g.to_dict("records")]
        (OUT / "sets" / f"{set_code}.json").write_text(json.dumps(records, separators=(",", ":")))
    # Column-wise search index; set names live once in `sets`, rarity is its first letter.
    index = {c: scored[c].tolist() for c in ("printing_key", "name", "set", "collector_number", "finish")}
    index["rarity"] = scored["rarity"].str[0].tolist()
    treatment_source = dataset.set_index("printing_key").reindex(scored["printing_key"])
    index["treatment"] = [_treatment(row) for _, row in treatment_source.iterrows()]
    index["sets"] = scored.drop_duplicates("set").set_index("set")["set_name"].to_dict()
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    meta = {**models.manifest, "n_printings": len(scored), "n_scored": int((scored["status"] == "scored").sum()),
            "n_sets": int(scored["set"].nunique()),
            "status_counts": scored["status"].value_counts().to_dict()}
    for m in ("model_b", "model_c"):
        meta[m] = {k: v for k, v in meta[m].items() if k != "features"}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json")) / 1e6
    print(f"{len(scored):,} printings ({meta['status_counts']}) in {meta['n_sets']} set files, {size:.0f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
