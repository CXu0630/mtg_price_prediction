"""Model A -- a printing's cross-sectional fair value from its attributes (no price
history). A reimplementation of the teammate's mtgpricepredictor/fair_value.py
on our data, so the app retrains it itself:

  log(price) ~ HistGradientBoostingRegressor(static_features)
  sigma      ~ HGB on |out-of-fold residual|        (GroupKFold(3) by oracle_id)
  80% range  = exp(fair_log -/+ q * sigma), q = split-conformal quantile of |resid| / sigma

Differences from theirs, both deliberate: sealed-product prices are left out
(they go stale and a custom card has none), and ages / printings-to-date are
computed as of the price date from our dataset, which also works for custom
cards (via their `reprint_of`).

Works on any dataset-shaped rows -- existing printings or custom cards
(mtg_price_prediction.cards) -- so both paths share one feature builder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from mtg_price_prediction.market import print_history

SEED = 0
ALPHA = 0.20
SIGMA_FLOOR = 1e-3
SIGMA_ITER = 200
LISTS = {"oracle_tags": 200, "keywords": 60, "type_tags": 60, "colors": None, "color_identity": None,
         "produced_mana": None, "promo_types": 40, "frame_effects": None}
CATS = ["rarity", "finish", "set_type", "layout", "border_color", "frame", "lang", "security_stamp",
        "watermark", "release_status"]
MIN_LEVEL = 20
UNKNOWN_SET_COLS = ["set_size", "rarity_slot_size", "set_price_level", "set_chase_level"]
MIN_NON_NULL = 50  # feature columns with fewer non-missing training values are dropped
BOOLS = ["reserved", "game_changer", "promo", "reprint", "full_art", "textless", "booster", "story_spotlight",
         "can_lead_commander", "can_lead_oathbreaker", "can_lead_brawl", "borderless"]
NUMS = ["conv_mana_cost", "edhrec_salt"]
HISTORY = ["printing_age_days", "card_age_days", "days_since_last_printing", "printings_to_date"]
NUM_TEXT = ["power", "toughness", "loyalty", "defense"]
RANKS = ["edhrec_rank", "penny_rank"]
MAIN_PRODUCTS = ("play", "draft", "set", "default")
NEW_RELEASE_DAYS = 30


def hgb(max_iter: int = 1000) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63,
                                         early_stopping=True, random_state=SEED)


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, len(scores)) - 1])


def _col(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c] if c in df.columns else pd.Series(None, index=df.index, dtype=object)


def _release_status(df: pd.DataFrame) -> pd.Series:
    age = df["printing_age_days"]
    return pd.Series(np.where(age < 0, "preorder", np.where(age <= NEW_RELEASE_DAYS, "new", "established")),
                     index=df.index)


def _cat_columns(df: pd.DataFrame) -> list[str]:
    return CATS + sorted(c for c in df.columns if c.startswith("legalities."))


def _cat_values(df: pd.DataFrame, c: str) -> pd.Series:
    return (_release_status(df) if c == "release_status" else _col(df, c)).fillna("NA").astype(str)


def _one_hot(values: pd.Series, levels: list, name: str) -> pd.DataFrame:
    e = pd.Series(values.to_numpy(), index=np.arange(len(values))).explode()
    e = e[e.isin(levels)]
    m = np.zeros((len(values), len(levels)), np.float32)
    m[e.index.to_numpy(dtype=int), pd.Categorical(e, categories=levels).codes] = 1
    return pd.DataFrame(m, index=values.index, columns=[f"{name}={v}" for v in levels])


def _mana(cost) -> tuple:
    syms = [s.split("/") for s in re.findall(r"\{([^}]*)\}", cost or "")]
    colored = [s for s in syms if set(s) & set("WUBRG")]
    per_color = [sum(ch in s for s in colored) for ch in "WUBRG"]
    return (len(colored), max(per_color), any(s == ["X"] for s in syms),
            any(len(set(s) - {"P"}) > 1 for s in colored), any("P" in s for s in syms))


def with_history(rows: pd.DataFrame, events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Adds printing/card ages and printings-to-date as of `as_of`. `events` holds every
    known printing's (oracle_id, released_at); custom cards add their own release."""
    rows = rows.copy()
    rows["released_at"] = pd.to_datetime(rows["released_at"])
    allev = pd.concat([rows[["oracle_id", "released_at"]], events[["oracle_id", "released_at"]]], ignore_index=True)
    allev["released_at"] = pd.to_datetime(allev["released_at"])
    h = print_history(allev, as_of).iloc[:len(rows)]
    for c in HISTORY:
        rows[c] = h[c].to_numpy()
    return rows


def set_price_levels(rows: pd.DataFrame) -> dict:
    """Per set: mean log price and mean of its 5 priciest cards (saved for rows that bring no prices)."""
    card = rows.assign(lp=np.log(rows["price"])).groupby(["set", "oracle_id"])["lp"].max()
    g = card.groupby(level="set")
    return {"mean": g.mean().to_dict(), "chase": g.apply(lambda s: s.nlargest(5).mean()).to_dict()}


def set_context(df: pd.DataFrame, vocab: dict) -> tuple[np.ndarray, np.ndarray]:
    """Set price / chase level from the OTHER cards of the same set. Rows without a
    price (custom cards) take the levels saved at training time for their set."""
    if "price" not in df or df["price"].isna().all():
        return (df["set"].map(vocab["set_price"]["mean"]).to_numpy(float),
                df["set"].map(vocab["set_price"]["chase"]).to_numpy(float))
    cards = pd.DataFrame({"set": df["set"].to_numpy(), "oid": df["oracle_id"].to_numpy(), "lp": np.log(df["price"].to_numpy())})
    per = cards.groupby(["set", "oid"]).lp.agg(s="sum", n="count", mx="max").reset_index()
    tot = per.groupby("set")[["s", "n"]].transform("sum")
    per["mean_others"] = (tot["s"] - per["s"]) / (tot["n"] - per["n"]).replace(0, np.nan)
    per["rank"] = per.groupby("set")["mx"].rank(ascending=False, method="first")
    top = per.sort_values("mx", ascending=False).groupby("set")["mx"]
    t5, t6 = per["set"].map(top.apply(lambda s: s.head(5).sum())), per["set"].map(top.apply(lambda s: s.head(6).sum()))
    n = per.groupby("set")["oid"].transform("size")
    per["chase_others"] = np.where(per["rank"] <= 5, (t6 - per["mx"]) / (n - 1).clip(upper=5).replace(0, np.nan), t5 / 5)
    m = cards[["set", "oid"]].merge(per[["set", "oid", "mean_others", "chase_others"]], on=["set", "oid"], how="left")
    return m["mean_others"].to_numpy(), m["chase_others"].to_numpy()


def learn_vocab(rows: pd.DataFrame, all_printings: pd.DataFrame) -> dict:
    vocab = {"lists": {}, "cats": {}, "rank_fill": {}}
    for c, top in LISTS.items():
        counts = _col(rows, c).explode().value_counts()
        vocab["lists"][c] = counts.index[:top].tolist() if top else counts.index.tolist()
    for c in _cat_columns(rows):
        counts = _cat_values(rows, c).value_counts()
        vocab["cats"][c] = counts.index[counts >= MIN_LEVEL].tolist()
    for c in RANKS:
        vocab["rank_fill"][c] = float(rows[c].max()) + 1
    vocab["set_size"] = all_printings.groupby("set")["oracle_id"].nunique().to_dict()
    vocab["slot_size"] = {f"{s}|{r}": n for (s, r), n in all_printings.groupby(["set", "rarity"])["oracle_id"].nunique().items()}
    vocab["set_price"] = set_price_levels(rows)
    vocab["rate_cols"] = sorted(c for c in all_printings.columns if c.endswith("_rate_percent"))
    return vocab


def static_features(df: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """Float matrix of printing attributes. Needs HISTORY columns (with_history) but no price
    beyond the optional `price` column used for set context."""
    parts = [df[NUMS + HISTORY].astype(float), df[BOOLS].astype(float).fillna(0.0)]
    num = pd.DataFrame(index=df.index)
    for c in NUM_TEXT:
        num[c] = pd.to_numeric(_col(df, c), errors="coerce")
    for c in RANKS:
        num[f"log_{c}"] = np.log(pd.to_numeric(_col(df, c), errors="coerce").fillna(vocab["rank_fill"][c]))
    mana = np.array([_mana(v) for v in _col(df, "mana_cost")], dtype=float).reshape(-1, 5)
    for i, c in enumerate(["colored_pips", "max_pips_one_color", "has_x", "hybrid", "phyrexian"]):
        num[c] = mana[:, i]
    num["oracle_text_len"] = _col(df, "oracle_text").fillna("").str.len()
    num["has_all_parts"] = _col(df, "all_parts").notna().astype(float)
    num["is_variation"] = _col(df, "variation_of").notna().astype(float)
    parts.append(num)
    R = df.reindex(columns=vocab["rate_cols"]).astype(float).fillna(0.0)
    parts.append(R)
    sup = pd.DataFrame(index=df.index)
    sup["set_size"] = df["set"].map(vocab["set_size"]).astype(float)
    sup["rarity_slot_size"] = (df["set"].astype(str) + "|" + df["rarity"].astype(str)).map(vocab["slot_size"]).astype(float)
    sup["n_products"], sup["max_rate"], sup["sum_rate"] = (R > 0).sum(axis=1), R.max(axis=1), R.sum(axis=1)
    main = [c for c in vocab["rate_cols"] if c.split("_rate")[0] in MAIN_PRODUCTS]
    collector = R["collector_rate_percent"] if "collector_rate_percent" in R else 0
    sup["collector_only"] = ((collector > 0) & (R[main].sum(axis=1) == 0)).astype(float)
    sup["set_price_level"], sup["set_chase_level"] = set_context(df, vocab)
    # A set the model has never seen (custom cards without a real set): typical context for its
    # set type -- left missing it would look like a one-card promo set, which prices high.
    unknown = ~df["set"].isin(vocab["set_size"].keys()).to_numpy()
    if unknown.any() and "unknown_set" in vocab:
        for i in np.flatnonzero(unknown):
            d = vocab["unknown_set"].get(str(df["set_type"].iloc[i]), vocab["unknown_set"]["_all"])
            for c in UNKNOWN_SET_COLS:
                sup.iloc[i, sup.columns.get_loc(c)] = d[c] if c != "rarity_slot_size" else d["slot"].get(str(df["rarity"].iloc[i]), d[c])
    parts.append(sup)
    for c, levels in vocab["lists"].items():
        parts.append(_one_hot(_col(df, c), levels, c))
    for c, levels in vocab["cats"].items():
        parts.append(_one_hot(_cat_values(df, c), levels, c))
    return pd.concat(parts, axis=1).astype(np.float32)


def _unknown_set_defaults(X: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """Median set-level context per set type (and rarity-slot size per rarity), over real sets."""
    per_set = X[UNKNOWN_SET_COLS].assign(set=rows["set"].to_numpy(), set_type=rows["set_type"].to_numpy(),
                                         rarity=rows["rarity"].to_numpy())
    out = {}
    for key, g in [("_all", per_set), *per_set.groupby("set_type")]:
        s = g.drop_duplicates("set")
        out[str(key)] = {**{c: float(s[c].median()) for c in UNKNOWN_SET_COLS},
                         "slot": g.drop_duplicates(["set", "rarity"]).groupby("rarity")["rarity_slot_size"].median().to_dict()}
    return out


@dataclass
class FairValueModel:
    model: HistGradientBoostingRegressor
    sigma_model: HistGradientBoostingRegressor
    q: float
    vocab: dict
    columns: list[str]
    as_of: str
    oof_mae: float
    oof_coverage: float

    @classmethod
    def train(cls, rows: pd.DataFrame, all_printings: pd.DataFrame, as_of: str) -> "FairValueModel":
        """rows: priced printings (a `price` column), history columns added."""
        vocab = learn_vocab(rows, all_printings)
        X, y, groups = static_features(rows, vocab), np.log(rows["price"].to_numpy()), rows["oracle_id"]
        vocab["unknown_set"] = _unknown_set_defaults(X, rows)
        # Near-empty columns (e.g. `defense`, Battles only) can be all-NaN inside a CV fold, which
        # HistGradientBoosting's binning rejects; they carry no learnable signal anyway.
        X = X.loc[:, X.notna().sum() >= MIN_NON_NULL]
        oof = cross_val_predict(hgb(), X, y, groups=groups, cv=GroupKFold(3))
        resid = np.abs(y - oof)
        sigma_oof = np.maximum(cross_val_predict(hgb(SIGMA_ITER), X, resid, groups=groups, cv=GroupKFold(3)), SIGMA_FLOOR)
        q = conformal_q(resid / sigma_oof, ALPHA)
        model, sigma_model = hgb().fit(X, y), hgb(SIGMA_ITER).fit(X, resid)
        return cls(model, sigma_model, q, vocab, list(X.columns), as_of,
                   float(np.mean(resid)), float(np.mean(resid <= q * sigma_oof)))

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """fair_log + fair / fair_lo / fair_hi in USD, indexed like rows."""
        X = static_features(rows, self.vocab).reindex(columns=self.columns, fill_value=0.0)
        fair_log = self.model.predict(X)
        half = self.q * np.maximum(self.sigma_model.predict(X), SIGMA_FLOOR)
        return pd.DataFrame({"fair_log": fair_log, "fair": np.exp(fair_log), "fair_lo": np.exp(fair_log - half),
                             "fair_hi": np.exp(fair_log + half)}, index=rows.index)
