"""Dataset-shaped rows -> the deep GP's input table (the columns of
training/features_model2.parquet), so custom cards and existing printings go
through the same transformations as scripts/prepare_training_data_model2.py.

Tag vocabularies are read back from the published model's own wide-column list
(preprocessor.json), never re-derived, so they can't drift from training.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from mtg_price_prediction.deep_gp import CATEGORICAL_COLS

ROOT = Path(__file__).resolve().parents[2]
# Built by scripts/build_pull_rate_defaults.py (training/, local); the published deep-GP repo ships a copy.
PULL_RATE_DEFAULTS = next((p for p in (ROOT / "training" / "pull_rate_defaults.parquet",
                                       ROOT / "models" / "deep_gp" / "pull_rate_defaults.parquet") if p.exists()),
                          ROOT / "training" / "pull_rate_defaults.parquet")
BOOL_COLS = ["reserved", "booster", "promo", "reprint", "borderless", "full_art", "textless", "story_spotlight",
             "game_changer", "can_lead_commander", "can_lead_oathbreaker", "can_lead_brawl"]
EXTERNAL = ["edhrec_rank", "edhrec_salt", "penny_rank", "num_paper_printings"]
VARIABLE = ["power", "toughness", "loyalty", "defense"]
COLOR_LISTS = [("colors", "color"), ("color_identity", "color_identity"), ("produced_mana", "produced_mana"),
               ("color_indicator", "color_indicator")]
TAG_COLS = ["keywords", "oracle_tags", "art_tags", "type_tags", "promo_types", "frame_effects"]


def _has(tags, tag) -> bool:
    return tags is not None and tag in tags


MAIN_PRODUCTS = ("play", "draft", "set", "default")
MODEL_A_COLS = ["num__colored_pips", "num__max_pips_one_color", "has_x", "hybrid", "phyrexian",
                "num__oracle_text_len", "has_all_parts", "is_variation", "num__set_size", "num__rarity_slot_size",
                "num__n_products", "num__max_rate", "num__sum_rate", "collector_only", "num__printing_age_days",
                "num__card_age_days", "num__days_since_last_printing", "num__printings_to_date",
                "num__set_price_level", "num__set_chase_level", "set_price_missing"]


def _mana(cost: str | None) -> tuple:
    syms = [s.split("/") for s in re.findall(r"\{([^}]*)\}", cost or "")]
    colored = [s for s in syms if set(s) & set("WUBRG")]
    per_color = [sum(ch in s for s in colored) for ch in "WUBRG"]
    return (len(colored), max(per_color), any(s == ["X"] for s in syms),
            any(len(set(s) - {"P"}) > 1 for s in colored), any("P" in s for s in syms))


def _set_price_features(card_lp: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """card_lp: (set, oracle_id, lp) for the GP's training cards. Each row gets the mean and top-5 mean of
    the set's training cards, excluding its own card -- a card never sees its own price."""
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


def model_a_features(rows: pd.DataFrame, dataset: pd.DataFrame, set_prices: pd.DataFrame,
                     as_of: pd.Timestamp) -> pd.DataFrame:
    """The inputs Model A has beyond the base GP table (mana detail, supply, reprint/age history as of
    `as_of`, set price level), for dataset-shaped `rows`. `dataset` gives set sizes and each card's other
    printings; `set_prices` is the exported model's (set, oracle_id, lp) table of training-card prices.
    scripts/add_model_a_features.py builds the training table with this same function."""
    d = rows.reset_index(drop=True)
    rate_cols = [c for c in dataset.columns if c.endswith("_rate_percent")]
    extra = pd.DataFrame(index=d.index)
    mana = np.array([_mana(c) for c in d["mana_cost"]], dtype=float).reshape(-1, 5)
    extra["num__colored_pips"], extra["num__max_pips_one_color"] = mana[:, 0], mana[:, 1]
    extra["has_x"], extra["hybrid"], extra["phyrexian"] = mana[:, 2] > 0, mana[:, 3] > 0, mana[:, 4] > 0
    extra["num__oracle_text_len"] = d["oracle_text"].fillna("").str.len().astype(float)
    extra["has_all_parts"] = d["all_parts"].notna() if "all_parts" in d else False
    extra["is_variation"] = d["variation_of"].notna() if "variation_of" in d else False

    set_size = dataset.groupby("set")["oracle_id"].nunique()
    slot_size = dataset.groupby(["set", "rarity"])["oracle_id"].nunique()
    extra["num__set_size"] = d["set"].map(set_size).astype(float)
    extra["num__rarity_slot_size"] = pd.MultiIndex.from_frame(d[["set", "rarity"]]).map(slot_size).astype(float)
    R = d.reindex(columns=rate_cols).astype(float).fillna(0.0)
    extra["num__n_products"] = (R > 0).sum(axis=1).astype(float)
    extra["num__max_rate"], extra["num__sum_rate"] = R.max(axis=1), R.sum(axis=1)
    main = [c for c in rate_cols if c.split("_rate")[0] in MAIN_PRODUCTS]
    extra["collector_only"] = (R.get("collector_rate_percent", 0) > 0) & (R[main].sum(axis=1) == 0)

    released = pd.to_datetime(dataset["released_at"])
    printings = dataset.assign(released=released)[released <= as_of].drop_duplicates("scryfall_id")
    per_card = printings.groupby("oracle_id")["released"].agg(["min", "max", "size"])
    rel = pd.to_datetime(d["released_at"])
    extra["num__printing_age_days"] = (as_of - rel).dt.days.astype(float)
    pc = per_card.reindex(d["oracle_id"])
    extra["num__card_age_days"] = (as_of - pc["min"]).dt.days.to_numpy(dtype=float)
    extra["num__days_since_last_printing"] = (as_of - pc["max"]).dt.days.to_numpy(dtype=float)
    extra["num__printings_to_date"] = pc["size"].fillna(0).to_numpy(dtype=float)
    return extra.join(_set_price_features(set_prices, d[["set", "oracle_id"]]))


def gp_feature_frame(rows: pd.DataFrame, wide_cols: list[str], context: dict | None = None) -> pd.DataFrame:
    """Raw GP feature table (categoricals as strings, booleans, numerics) for `rows`;
    the GP preprocessor (deep_gp.transform) does vocab encoding / scaling.

    A GP trained with Model A's inputs (MODEL_A_COLS in wide_cols) also needs `context`:
    {"dataset", "set_prices", "as_of"} -- see model_a_features and DeepGPPredictor.feature_context."""
    rows = rows.reset_index(drop=True)
    out: dict[str, pd.Series] = {c: rows[c].fillna(False).astype(bool) for c in BOOL_COLS}
    out["conv_mana_cost"] = pd.to_numeric(rows["conv_mana_cost"], errors="coerce")
    for c in CATEGORICAL_COLS:
        out[c] = rows[c] if c in rows else pd.Series(None, index=rows.index, dtype=object)
    for c in VARIABLE:
        raw = rows[c] if c in rows else pd.Series(None, index=rows.index, dtype=object)
        num = pd.to_numeric(raw, errors="coerce")
        out[c], out[f"{c}_is_variable"] = num, raw.notna() & num.isna()
    for col, prefix in COLOR_LISTS:
        vals = rows[col] if col in rows else pd.Series([None] * len(rows))
        for letter in "WUBRG":
            out[f"{prefix}_{letter}"] = vals.apply(lambda t, l=letter: _has(t, l))
    for col in TAG_COLS:
        keep = [c[len(col) + 2:] for c in wide_cols if c.startswith(f"{col}__") and c != f"{col}__other"]
        vals = rows[col] if col in rows else pd.Series([None] * len(rows))
        for tag in keep:
            out[f"{col}__{tag}"] = vals.apply(lambda t, g=tag: _has(t, g))
        if f"{col}__other" in wide_cols:
            keep_set = set(keep)
            out[f"{col}__other"] = vals.apply(lambda t: t is not None and len(t) > 0 and any(x not in keep_set for x in t))
    for c in rows.columns:
        if c.startswith("legalities."):
            out[c.replace("legalities.", "legal_")] = rows[c] == "legal"
    rate_cols = [c for c in rows.columns if c.endswith("_rate_percent")]
    defaults = pd.read_parquet(PULL_RATE_DEFAULTS).set_index(["rarity", "finish"])
    for c in rate_cols:
        default = [defaults.loc[k, c] if k in defaults.index else np.nan for k in zip(rows["rarity"], rows["finish"])]
        is_default = rows[c].isna()
        out[c] = rows[c].where(~is_default, pd.Series(default, index=rows.index))
        out[f"{c}__is_default"] = is_default
    for c in EXTERNAL:
        out[c] = pd.to_numeric(rows[c], errors="coerce")
        out[f"{c}__isna"] = out[c].isna()
    out["has_image"] = rows["illustration_id"].notna() if "illustration_id" in rows else pd.Series(False, index=rows.index)
    frame = pd.DataFrame(out)
    if any(c in wide_cols for c in MODEL_A_COLS):
        if context is None:
            raise ValueError("this GP was trained with Model A's inputs: pass context=gp.feature_context(dataset)")
        frame = frame.join(model_a_features(rows, context["dataset"], context["set_prices"], context["as_of"]))
    # The table the GP was trained on: its exact columns, in order (power__isna etc. are added by the preprocessor).
    base_cols = [c for c in wide_cols if not (c.endswith("__isna") and c[:-6] in ("power", "toughness", "loyalty", "defense"))]
    missing = [c for c in base_cols if c not in frame]
    for c in missing:
        frame[c] = False
    return frame[CATEGORICAL_COLS + [c for c in base_cols if c not in CATEGORICAL_COLS]]
