"""One output row per card -- existing printings (nightly, all at once) and custom
cards (on demand, from the API) -- in the schema of docs/APP_DESIGN.md §3.

  market     price_today (TCGplayer), ck_retail, ck_buylist (trusted only), cardmarket_eur
  Model A    fair / fair_lo / fair_hi           (teammate's method, retrained here)
             fair_gp / fair_gp_lo / fair_gp_hi  (deep GP backend) + comps
  Model B    pred_price / price_lo / price_hi (14 days), signal BUY / SELL / HOLD
  Model C    buylist_pred / buylist_lo / buylist_hi (14 days)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mtg_price_prediction import market
from mtg_price_prediction.cards import CustomCard, custom_rows
from mtg_price_prediction.fair_value import FairValueModel, with_history

SIGNALS = ("BUY", "SELL", "HOLD")


@dataclass
class AppModels:
    a: FairValueModel
    b: market.MoveModel
    c: market.MoveModel
    manifest: dict
    gp_dir: Path

    @classmethod
    def load(cls, models_dir: Path, gp_dir: Path) -> "AppModels":
        return cls(joblib.load(models_dir / "model_a.joblib"), joblib.load(models_dir / "model_b.joblib"),
                   joblib.load(models_dir / "model_c.joblib"), json.loads((models_dir / "manifest.json").read_text()),
                   gp_dir)

    _gp = None

    @property
    def gp(self):
        if self._gp is None:
            from mtg_price_prediction.deep_gp import DeepGPPredictor
            self._gp = DeepGPPredictor.from_pretrained(str(self.gp_dir))
        return self._gp


def _move_outputs(model: market.MoveModel, rows: pd.DataFrame, base: pd.Series, prefix: str) -> pd.DataFrame:
    p = model.predict(rows)
    return pd.DataFrame({f"{prefix}_pred": base * np.exp(p["move_med"]), f"{prefix}_lo": base * np.exp(p["move_lo"]),
                         f"{prefix}_hi": base * np.exp(p["move_hi"]), f"{prefix}_move": p["move_med"]}, index=rows.index)


def _signal(move_lo: pd.Series, move_hi: pd.Series, jump: pd.Series) -> pd.Series:
    sig = np.where(move_lo > 0, "BUY", np.where(move_hi < 0, "SELL", "HOLD"))
    return pd.Series(np.where(jump.abs().fillna(0) > market.UNCONFIRMED_JUMP, "HOLD", sig), index=move_lo.index)


def market_outputs(models: AppModels, live: pd.DataFrame) -> pd.DataFrame:
    """B and C for every live (newest-day, tradeable) row, keyed by printing_key."""
    b = models.b.predict(live)
    out = pd.DataFrame({"printing_key": live["printing_key"], "price_today": live["price"],
                        "ck_retail": live["ck_retail"], "ck_buylist": live["ck_buylist"],
                        "cardmarket_eur": live["cardmarket_eur"], "jump_7d": live["jump_7d"]}, index=live.index)
    out["pred_price"] = live["price"] * np.exp(b["move_med"])
    out["price_lo"], out["price_hi"] = live["price"] * np.exp(b["move_lo"]), live["price"] * np.exp(b["move_hi"])
    out["signal"] = _signal(b["move_lo"], b["move_hi"], live["jump_7d"])
    out["unconfirmed_jump"] = live["jump_7d"].abs() > market.UNCONFIRMED_JUMP
    has_bl = live["ck_buylist"].notna()
    c = models.c.predict(live[has_bl])
    bl = live.loc[has_bl, "ck_buylist"]
    out.loc[has_bl, "buylist_pred"] = bl * np.exp(c["move_med"])
    out.loc[has_bl, "buylist_lo"], out.loc[has_bl, "buylist_hi"] = bl * np.exp(c["move_lo"]), bl * np.exp(c["move_hi"])
    return out


def score_existing(dataset: pd.DataFrame, models: AppModels, live: pd.DataFrame, gp_static: pd.DataFrame,
                   tcg_today: pd.Series, as_of: pd.Timestamp) -> pd.DataFrame:
    """Every dataset printing. `tcg_today`: newest-day TCGplayer price per dataset row (NaN = unpriced)."""
    rows = with_history(dataset.assign(price=tcg_today.to_numpy()), dataset[["oracle_id", "released_at"]], as_of)
    a = models.a.predict(rows)
    out = dataset[["printing_key", "scryfall_id", "name", "set", "set_name", "collector_number", "finish",
                   "rarity", "oracle_id"]].copy()
    out[["fair", "fair_lo", "fair_hi"]] = a[["fair", "fair_lo", "fair_hi"]].to_numpy()
    out = out.merge(gp_static, on="printing_key", how="left")
    # Market rows are one per (scryfall_id, MTGJSON finish series); specialty foils share the foil series.
    series = dataset["scryfall_id"] + "|" + dataset["finish"].map(market.PH_FINISH).fillna("foil")
    series_of = dict(zip(dataset["printing_key"], series))
    mk = market_outputs(models, live)
    mk.index = mk.pop("printing_key").map(series_of).to_numpy()
    out = out.join(mk.reindex(series.to_numpy()).reset_index(drop=True))
    out["price_today"] = out["price_today"].fillna(pd.Series(tcg_today.to_numpy()))
    out["status"] = np.where(out["price_today"].notna(), np.where(out["pred_price"].notna(), "scored", "valued only"),
                             "unpriced")
    out["price_date"] = str(as_of.date())
    return out


def score_custom(cards: list[CustomCard], dataset: pd.DataFrame, models: AppModels, live: pd.DataFrame,
                 as_of: pd.Timestamp, backend: str = "a", images: list | None = None) -> pd.DataFrame:
    """Custom cards: Model A (chosen backend) gives the one day of price data, B forecasts from it,
    C is not applicable (no buylist exists)."""
    rows = custom_rows(cards, dataset, as_of)
    hist = with_history(rows, dataset[["oracle_id", "released_at"]], as_of)
    a = models.a.predict(hist)
    out = rows[["printing_key", "name", "set", "finish", "rarity"]].copy()
    out[["fair", "fair_lo", "fair_hi"]] = a[["fair", "fair_lo", "fair_hi"]].to_numpy()
    if backend == "gp":
        from mtg_price_prediction.embed import embed_images, embed_texts
        from mtg_price_prediction.gp_features import gp_feature_frame

        gp = models.gp
        feats = gp_feature_frame(rows.assign(illustration_id=[("upload" if im is not None else None)
                                                                for im in (images or [None] * len(rows))]),
                                 gp.prep["wide_cols"])
        pred = gp.predict(feats, embed_texts(rows["oracle_text"].tolist()), embed_images(images or [None] * len(rows)))
        out["fair_gp"], out["fair_gp_lo"], out["fair_gp_hi"] = (pred[c].to_numpy() for c in ("price_mid", "price_low", "price_high"))
        day0 = out["fair_gp"].to_numpy()
    else:
        day0 = out["fair"].to_numpy()
    b_rows = market.custom_day0_rows(rows, day0, live, as_of, dataset[["oracle_id", "released_at"]])
    b = models.b.predict(b_rows)
    out["price_today"] = np.nan
    out["model_a_price"] = day0
    out["pred_price"] = day0 * np.exp(b["move_med"].to_numpy())
    out["price_lo"], out["price_hi"] = day0 * np.exp(b["move_lo"].to_numpy()), day0 * np.exp(b["move_hi"].to_numpy())
    out["signal"] = _signal(b["move_lo"], b["move_hi"], pd.Series(0.0, index=b.index)).to_numpy()
    out["status"] = "custom"
    out["price_date"] = str(as_of.date())
    out["quantity"] = [c.quantity for c in cards]
    return out
