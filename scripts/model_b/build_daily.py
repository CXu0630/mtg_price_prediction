"""Daily as-of snapshot table for a Model B rebuild (14-day TCGplayer price
move), modeled on the teammate's mtgpricepredictor (build_daily.py +
price_move.py), built from our own raw MTGJSON price history
(mtg-card-prices/raw/price_history.parquet, 2026-06-22 .. 2026-09-20).

One row per (printing_key, snapshot day t) for printings in
training/features_model2.parquet (the rows the deep GP can embed) priced >=
$0.50 on TCGplayer at t, basic lands dropped. t runs from the first day with a
full 21-day lookback to the last day with a 14-day outcome.

Prices: TCGplayer retail (USD, paper) is the price and the target. Spikes
(a day > SPIKE x off the median of its neighbourhood) are dropped and carried
forward -- causally for features (trailing 5-day median, so a feature never
looks past t), centered for the target (label cleaning, not look-ahead).

Inputs (teammate's Model B names where they exist):
  own price   log_price, change_21d_pct (clipped -90..300), volatility_21d
              (std of daily % changes, clipped 0..0.5), jump_7d (log price /
              median of the 7 days before)
  peers       sib_chg, sib_div, sib_rel_level (the card's other printings),
              set_peer (21-day move of the set's other cards)
  markets     lvl_ck_retail, lvl_ck_buylist, lvl_cardmarket = log(market / TCG),
              carried forward up to 7 days. Manapool left out: its prices in
              this window are corrupted (mtg-card-market-dataset/README.md).
  Model A     fair_log / gap from the teammate's OOF weekly anchors
              (mtgpricepredictor/models/fair_anchors.parquet), latest anchor <= t
  static      ages and printings to date as of t, releases_before_target,
              log_edhrec (missing = 40,000), reprint, promo, reserved;
              categoricals release_status, rarity, finish_group, set_type
Target: ret = log(price at t+14 / price at t), centered-cleaned series.

    python scripts/model_b/build_daily.py   # -> training/model_b/daily.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PRICE_HISTORY = ROOT / "mtg-card-prices" / "raw" / "price_history.parquet"
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
GP_FEATURES = ROOT / "training" / "features_model2.parquet"
FAIR_ANCHORS = ROOT.parent / "mtgpricepredictor" / "models" / "fair_anchors.parquet"
OUT_DIR = ROOT / "training" / "model_b"

HORIZON, LOOKBACK, BULK, SPIKE = 14, 21, 0.50, 2.0
LEVEL_CARRY_DAYS = 7
EDHREC_WORST = 40_000
NEW_RELEASE_DAYS = 30
MARKETS = {"ck_retail": ("cardkingdom", "retail"), "ck_buylist": ("cardkingdom", "buylist"),
           "cardmarket": ("cardmarket", "retail")}
PH_FINISH = {"nonfoil": "normal", "etched": "etched"}  # everything else (foil + specialty foils) -> "foil"


def printings() -> pd.DataFrame:
    """Static attributes for the universe: rows the deep GP can embed."""
    keys = pd.read_parquet(GP_FEATURES, columns=["__printing_key"])["__printing_key"]
    ds = pd.read_parquet(DATASET, columns=[
        "printing_key", "scryfall_id", "oracle_id", "set", "rarity", "finish", "set_type", "released_at",
        "edhrec_rank", "reprint", "promo", "reserved", "type_line"]).set_index("printing_key").loc[keys].reset_index()
    ds = ds[~ds["type_line"].fillna("").str.contains("Basic Land")]
    ds["ph_finish"] = ds["finish"].map(PH_FINISH).fillna("foil")
    # A specialty foil and the plain foil of the same scryfall_id share one MTGJSON series: keep one row.
    ds = ds.drop_duplicates(["scryfall_id", "ph_finish"]).reset_index(drop=True)
    ds["released_at"] = pd.to_datetime(ds["released_at"])
    return ds


def wide_prices(provider: str, list_type: str, ds: pd.DataFrame) -> pd.DataFrame:
    t = pq.read_table(PRICE_HISTORY, filters=[("provider", "=", provider), ("list_type", "=", list_type),
                                              ("medium", "=", "paper")],
                      columns=["scryfall_id", "finish", "date", "price"]).to_pandas()
    w = t.pivot_table(index=["scryfall_id", "finish"], columns="date", values="price", aggfunc="mean")
    # MTGJSON skipped a couple of days: put every series on the full calendar (gaps carry forward in despike).
    w = w.reindex(columns=pd.date_range(t["date"].min(), t["date"].max(), freq="D"))
    idx = pd.MultiIndex.from_frame(ds[["scryfall_id", "ph_finish"]], names=["scryfall_id", "finish"])
    w = w.reindex(idx)
    w.index = ds.index
    return w.sort_index(axis=1)


def despike(w: pd.DataFrame, causal: bool) -> pd.DataFrame:
    """Drop days more than SPIKE x off their 5-day neighbourhood median, carry forward."""
    med = (w.T.shift(1).rolling(5, min_periods=3).median() if causal
           else w.T.rolling(5, center=True, min_periods=3).median()).T
    ratio = w / med
    bad = (ratio > SPIKE) | (ratio < 1 / SPIKE)
    return w.mask(bad).T.ffill().T


def peer_features(day: pd.DataFrame) -> pd.DataFrame:
    """sib_chg / sib_div / sib_rel_level over the card's other printings, set_peer over the set's other cards."""
    chg, lp = day["change_21d_pct"], day["log_price"]
    peer_chg = chg.where(day["price"] >= BULK)
    g = day.groupby("oracle_id")
    s, n = g["_pchg"].transform("sum"), g["_pchg"].transform("count")
    own = peer_chg.notna()
    out = pd.DataFrame(index=day.index)
    out["sib_chg"] = (s - peer_chg.fillna(0)) / (n - own).replace(0, np.nan)
    out["sib_div"] = chg - out["sib_chg"]
    ls, ln = g["log_price"].transform("sum"), g["log_price"].transform("count")
    out["sib_rel_level"] = lp - (ls - lp) / (ln - 1).replace(0, np.nan)
    card = day[own].groupby(["set", "oracle_id"])["_pchg"].mean().rename("card_chg").reset_index()
    tot = card.groupby("set")["card_chg"].agg(["sum", "count"])
    card = card.join(tot, on="set")
    card["set_peer"] = (card["sum"] - card["card_chg"]) / (card["count"] - 1).replace(0, np.nan)
    out["set_peer"] = day[["set", "oracle_id"]].merge(card[["set", "oracle_id", "set_peer"]], how="left",
                                                      on=["set", "oracle_id"])["set_peer"].to_numpy()
    return out


def print_history(ds: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Ages and printings-to-date of each printing's card as of t."""
    ev = ds[["oracle_id", "released_at"]].drop_duplicates()
    ev = ev[ev["released_at"] <= t]
    agg = ev.groupby("oracle_id")["released_at"].agg(["min", "max", "size"]).reindex(ds["oracle_id"])
    return pd.DataFrame({
        "printing_age_days": (t - ds["released_at"]).dt.days.to_numpy(float),
        "card_age_days": (t - agg["min"]).dt.days.to_numpy(float),
        "days_since_last_printing": (t - agg["max"]).dt.days.to_numpy(float),
        "printings_to_date": agg["size"].fillna(0).to_numpy(float),
    }, index=ds.index)


def build() -> pd.DataFrame:
    ds = printings()
    print(f"{len(ds)} printings in universe", flush=True)
    tcg_raw = wide_prices("tcgplayer", "retail", ds)
    feat_px, target_px = despike(tcg_raw, causal=True), despike(tcg_raw, causal=False)
    levels = {m: despike(wide_prices(p, lt, ds), causal=True).T.ffill(limit=LEVEL_CARRY_DAYS).T
              for m, (p, lt) in MARKETS.items()}
    anchors = pd.read_parquet(FAIR_ANCHORS)
    anchors = anchors[anchors["printing_key"].isin(set(ds["printing_key"]))]

    dates = feat_px.columns
    days = [d for d in dates if d - pd.Timedelta(days=LOOKBACK) >= dates[0] and d + pd.Timedelta(days=HORIZON) <= dates[-1]]
    print(f"snapshot days {days[0].date()} .. {days[-1].date()} ({len(days)})", flush=True)

    static = pd.DataFrame({
        "printing_key": ds["printing_key"], "oracle_id": ds["oracle_id"], "set": ds["set"],
        "log_edhrec": np.log(ds["edhrec_rank"].fillna(EDHREC_WORST).astype(float)),
        "reprint": ds["reprint"].astype(float), "promo": ds["promo"].astype(float),
        "reserved": ds["reserved"].astype(float), "rarity": ds["rarity"], "set_type": ds["set_type"],
        "finish_group": ds["finish"].where(ds["finish"].isin(["nonfoil", "foil", "etched"]), "special"),
    })
    frames = []
    for t in days:
        p = feat_px[t]
        start = feat_px[t - pd.Timedelta(days=LOOKBACK)]
        window = feat_px.loc[:, t - pd.Timedelta(days=LOOKBACK):t]
        prior7 = feat_px.loc[:, t - pd.Timedelta(days=7):t - pd.Timedelta(days=1)]
        day = static.copy()
        day["snapshot_date"] = t
        day["price"] = p
        day["log_price"] = np.log(p)
        day["change_21d_pct"] = (100 * (p / start - 1)).clip(-90, 300)
        day["volatility_21d"] = (window / window.shift(axis=1) - 1).std(axis=1).clip(0, 0.5)
        day["jump_7d"] = np.log(p / prior7.median(axis=1))
        day["_pchg"] = day["change_21d_pct"].where(p >= BULK)
        day = day.join(peer_features(day)).drop(columns="_pchg")
        for m, w in levels.items():
            day[f"lvl_{m}"] = np.log(w[t] / p).clip(-3, 3)
        day = day.join(print_history(ds, t))
        rel = ds["released_at"]
        day["release_status"] = np.select([rel > t, rel > t - pd.Timedelta(days=NEW_RELEASE_DAYS)],
                                          ["preorder", "new"], "established")
        day["releases_before_target"] = ((rel > t) & (rel <= t + pd.Timedelta(days=HORIZON))).astype(float)
        day["ret"] = np.log(target_px[t + pd.Timedelta(days=HORIZON)] / target_px[t])
        frames.append(day[(day["price"] >= BULK) & day["ret"].notna()])
    rows = pd.concat(frames, ignore_index=True)

    # Model A: fair value from the latest anchor on or before each snapshot day (no look-ahead).
    a = anchors.sort_values("anchor_date")[["printing_key", "anchor_date", "fair_log"]]
    rows = pd.merge_asof(rows.sort_values("snapshot_date"), a, left_on="snapshot_date", right_on="anchor_date",
                         by="printing_key", direction="backward").drop(columns="anchor_date")
    rows["gap"] = rows["log_price"] - rows["fair_log"]
    return rows.reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build()
    path = OUT_DIR / "daily.parquet"
    rows.to_parquet(path, index=False)
    print(f"{len(rows):,} rows ({rows['printing_key'].nunique():,} printings x {rows['snapshot_date'].nunique()} days) -> {path}")
    print(rows.drop(columns=["printing_key", "oracle_id", "set"]).describe().T[["count", "mean", "min", "max"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
