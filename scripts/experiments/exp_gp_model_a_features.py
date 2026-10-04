"""Experiment 3: the deep GP plus the features the teammate's Model A has and we
don't (mtgpricepredictor/fair_value.py's static_features + set_context), rebuilt
from our own dataset. Two variants:

  gp_plus_a            everything below
  gp_plus_a_no_setprice  same minus set price/chase level -- the "brand-new set"
                       case, where no card of the set has a price yet

Added (numerics prefixed num__ so the GP preprocessor imputes + standardizes them):
  - supply: set size, rarity-slot size (distinct cards per set x rarity), number
    of products carrying the printing, max / summed pull rate, collector-only.
    From the dataset's RAW pull rates -- features_model2 fills unknown rates
    with (rarity, finish) defaults, which would fake n_products.
  - reprint/age history as of the price window's end: printing age, card age,
    days since the card's last printing, printings to date.
  - mana cost: colored pips, max pips of one color, X, hybrid, Phyrexian;
    oracle text length; has related parts; is a variation.
  - set price level (mean log price of the set's other cards) and chase level
    (mean of its 5 priciest other cards), from TRAIN-split cards only: a train
    card sees its set's other train cards, a val/test card sees the set's train
    cards -- no held-out price ever enters a feature (stricter than Model A,
    which uses every other card's price).

    python scripts/experiments/exp_gp_model_a_features.py
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from _gp_common import EXP_DIR, ROOT, SPLIT_SEED, print_comparison, train_variant
from train_model2 import _split

BASE_FEATURES = ROOT / "training" / "features_model2.parquet"
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
AS_OF = pd.Timestamp("2026-09-20")  # end of the price_avg window
MAIN_PRODUCTS = ("play", "draft", "set", "default")
SET_PRICE_COLS = ["num__set_price_level", "num__set_chase_level", "set_price_missing"]


def _mana(cost: str | None) -> tuple:
    syms = [s.split("/") for s in re.findall(r"\{([^}]*)\}", cost or "")]
    colored = [s for s in syms if set(s) & set("WUBRG")]
    per_color = [sum(ch in s for s in colored) for ch in "WUBRG"]
    return (len(colored), max(per_color), any(s == ["X"] for s in syms),
            any(len(set(s) - {"P"}) > 1 for s in colored), any("P" in s for s in syms))


def _set_price_features(card_lp: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """card_lp: (set, oracle_id, lp) for TRAIN cards (lp = max log price over the
    card's train printings in that set). rows: (set, oracle_id) to score. Each
    row gets stats over the set's train cards excluding its own card."""
    stats = {}
    for s, g in card_lp.groupby("set"):
        lp = g.set_index("oracle_id")["lp"]
        stats[s] = (lp.sum(), len(lp), lp.sort_values(ascending=False))
    level, chase = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    for i, (s, oid) in enumerate(zip(rows["set"], rows["oracle_id"])):
        if s not in stats:
            continue
        total, n, ranked = stats[s]
        own = ranked.get(oid)
        if own is not None:
            total, n, ranked = total - own, n - 1, ranked.drop(oid)
        if n > 0:
            level[i] = total / n
            chase[i] = ranked.head(5).mean()
    return pd.DataFrame({"num__set_price_level": level, "num__set_chase_level": chase,
                         "set_price_missing": np.isnan(level)}, index=rows.index)


def model_a_features(printing_keys: pd.Series, log_price: np.ndarray, train_mask: np.ndarray) -> pd.DataFrame:
    """The extra columns for rows `printing_keys` (dataset printing_key order).
    Set price/chase level use only rows where `train_mask` is True."""
    ds = pd.read_parquet(DATASET)
    rate_cols = [c for c in ds.columns if c.endswith("_rate_percent")]
    ds = ds.set_index("printing_key")
    d = ds.loc[np.asarray(printing_keys)].reset_index()

    extra = pd.DataFrame(index=d.index)
    mana = np.array([_mana(c) for c in d["mana_cost"]], dtype=float)
    extra["num__colored_pips"], extra["num__max_pips_one_color"] = mana[:, 0], mana[:, 1]
    extra["has_x"], extra["hybrid"], extra["phyrexian"] = mana[:, 2] > 0, mana[:, 3] > 0, mana[:, 4] > 0
    extra["num__oracle_text_len"] = d["oracle_text"].fillna("").str.len().astype(float)
    extra["has_all_parts"] = d["all_parts"].notna()
    extra["is_variation"] = d["variation_of"].notna()

    # Supply (attributes of the set/print run, not prices -- fine to use every row).
    all_ds = ds.reset_index()
    set_size = all_ds.groupby("set")["oracle_id"].nunique()
    slot_size = all_ds.groupby(["set", "rarity"])["oracle_id"].nunique()
    extra["num__set_size"] = d["set"].map(set_size).astype(float)
    extra["num__rarity_slot_size"] = pd.MultiIndex.from_frame(d[["set", "rarity"]]).map(slot_size).astype(float)
    R = d[rate_cols].fillna(0.0)
    extra["num__n_products"] = (R > 0).sum(axis=1).astype(float)
    extra["num__max_rate"], extra["num__sum_rate"] = R.max(axis=1), R.sum(axis=1)
    main = [c for c in rate_cols if c.split("_rate")[0] in MAIN_PRODUCTS]
    extra["collector_only"] = (R.get("collector_rate_percent", 0) > 0) & (R[main].sum(axis=1) == 0)

    # Reprint/age history from the card's paper printings released by AS_OF.
    released = pd.to_datetime(all_ds["released_at"])
    printings = all_ds.assign(released=released)[released <= AS_OF].drop_duplicates("scryfall_id")
    per_card = printings.groupby("oracle_id")["released"].agg(["min", "max", "size"])
    rel = pd.to_datetime(d["released_at"])
    extra["num__printing_age_days"] = (AS_OF - rel).dt.days.astype(float)
    pc = per_card.reindex(d["oracle_id"])
    extra["num__card_age_days"] = (AS_OF - pc["min"]).dt.days.to_numpy(dtype=float)
    extra["num__days_since_last_printing"] = (AS_OF - pc["max"]).dt.days.to_numpy(dtype=float)
    extra["num__printings_to_date"] = pc["size"].fillna(0).to_numpy(dtype=float)

    # Set price / chase level, train-split prices only.
    tr = d[train_mask].assign(lp=np.asarray(log_price)[train_mask])
    card_lp = tr.groupby(["set", "oracle_id"], as_index=False)["lp"].max()
    extra = extra.join(_set_price_features(card_lp, d[["set", "oracle_id"]]))
    return extra


def build() -> pd.DataFrame:
    feats = pd.read_parquet(BASE_FEATURES)
    train_idx, _, _ = _split(feats["__group_oracle_id"].to_numpy(), SPLIT_SEED)
    train_mask = np.zeros(len(feats), bool)
    train_mask[train_idx] = True
    extra = model_a_features(feats["__printing_key"], feats["__target_log_price_avg"].to_numpy(), train_mask)
    print(f"added {extra.shape[1]} columns; set price level missing for {extra['set_price_missing'].mean():.1%} of rows")
    return pd.concat([feats, extra.set_index(feats.index)], axis=1)


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    full = build()
    plus_a = EXP_DIR / "features_model2_plus_a.parquet"
    full.to_parquet(plus_a, index=False)
    no_setprice = EXP_DIR / "features_model2_plus_a_no_setprice.parquet"
    full.drop(columns=SET_PRICE_COLS).to_parquet(no_setprice, index=False)

    train_variant("gp_plus_a", plus_a)
    train_variant("gp_plus_a_no_setprice", no_setprice)
    print_comparison(["gp_baseline", "gp_plus_a_no_setprice", "gp_plus_a"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
