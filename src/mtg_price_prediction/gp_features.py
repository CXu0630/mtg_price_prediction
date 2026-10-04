"""Dataset-shaped rows -> the deep GP's input table (the columns of
training/features_model2.parquet), so custom cards and existing printings go
through the same transformations as scripts/prepare_training_data_model2.py.

Tag vocabularies are read back from the published model's own wide-column list
(preprocessor.json), never re-derived, so they can't drift from training.
"""
from __future__ import annotations

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


def gp_feature_frame(rows: pd.DataFrame, wide_cols: list[str]) -> pd.DataFrame:
    """Raw GP feature table (categoricals as strings, booleans, numerics) for `rows`;
    the GP preprocessor (deep_gp.transform) does vocab encoding / scaling."""
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
    # The table the GP was trained on: its exact columns, in order (power__isna etc. are added by the preprocessor).
    base_cols = [c for c in wide_cols if not (c.endswith("__isna") and c[:-6] in ("power", "toughness", "loyalty", "defense"))]
    missing = [c for c in base_cols if c not in frame]
    for c in missing:
        frame[c] = False
    return frame[CATEGORICAL_COLS + [c for c in base_cols if c not in CATEGORICAL_COLS]]
