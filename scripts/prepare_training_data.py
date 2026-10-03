"""Builds training/features.parquet, the Model 1 (GBDT baseline) training
table -- local only (gitignored), not published to HF (see training/README.md).

Feature-engineering decisions (all made this session, see conversation/
SPECS.md for the reasoning):

- **No price as a feature, anywhere.** price_min/max/avg/early/mid/late
  are the target's source, not features -- including them as X would let
  the model just copy the target. {subtype}_price_usd (sealed product
  price) isn't direct leakage but is still a raw dollar value with the
  same staleness problem one hop removed, and its information content
  (premium vs. budget product) is already redundant with subtype/rarity/
  frame_effects/borderless, which don't carry that problem -- dropped too.
- **No `set`/`set_name`/`set_id`.** Two reasons: (1) already categorized
  as identity/keys, not features, since COLUMNS.md's first pass; (2) more
  importantly, this project's stated goal includes pricing *unreleased*
  cards -- a raw set-code categorical has zero learnable signal for a set
  the model has never seen, i.e. it actively fails on the hardest and most
  interesting inference case. `released_at` + `set_type` + `promo_types`
  (which already includes `universesbeyond` for crossover-hype sets like
  LTR) capture the generalizable part of "which set" without needing the
  exact code -- a novel future set still has well-defined values for all
  three.
- **Abundant multi-hot tag columns (keywords, oracle_tags, art_tags,
  type_tags, promo_types) are trimmed to their top-N most frequent values
  + a catch-all "_other" flag**, rather than one column per distinct value
  (some of these have 4,000-11,000+ distinct values, almost all
  long-tail). N chosen per column from its actual frequency concentration
  (checked empirically, not guessed) -- see TAG_TOP_N below. frame_effects
  (only 17 distinct total) is kept in full, no trimming needed.
- Raw text/string columns redundant with an already-derived numeric/tag
  form are dropped (`mana_cost` given `conv_mana_cost`+`colors`,
  `type_line` given `type_tags`). `oracle_text` itself is dropped too --
  Model 1 deliberately uses no dense text embeddings (SPECS.md §4.1).
- power/toughness/loyalty/defense have non-numeric markers (`*`, `+2`,
  `X`, `1+*`, dice notation) for variable-value cards -- parsed to float
  where possible, NaN otherwise, with a companion `_is_variable` boolean
  flag so that information isn't silently lost.
- `oracle_id` is kept, but only as a group key for train/test splitting
  (SPECS.md §6: split by printing, not card-concept -- different
  printings of the same card are too similar to be in both sets), not as
  a model feature.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
OUT_PATH = ROOT / "training" / "features.parquet"

# Chosen from each column's actual frequency concentration (checked this
# session): frame_effects has only 17 distinct values total (kept in
# full); promo_types' top 20 already covers the overwhelming majority of
# non-null occurrences; keywords/type_tags have a moderate long tail;
# oracle_tags/art_tags are the messiest (4,332 / 11,220 distinct values,
# mixing real signal with naming trivia like "alliteration" or universal
# non-discriminative tags like "artist-signature") -- top 100 is a
# pragmatic cutoff, not a curated one; revisit if these turn out to matter
# a lot in the ablation.
TAG_TOP_N = {
    "keywords": 50,
    "oracle_tags": 100,
    "art_tags": 100,
    "type_tags": 60,
    "promo_types": 20,
    "frame_effects": None,  # None = keep all (only 17 distinct)
}

COLOR_LETTERS = ["W", "U", "B", "R", "G"]

# Not used as features -- pure identifiers/join keys, or redundant with a
# derived form kept elsewhere. oracle_id is kept separately as the
# train/test split group key, not dropped outright.
DROP_COLS = [
    "scryfall_id", "printing_key", "uuid", "name",
    "set_id", "set", "set_name", "collector_number", "variation_of", "tcgplayer_id",
    "artist_ids", "illustration_id",
    "all_parts", "attraction_lights",
    "mana_cost", "type_line", "oracle_text",
]

PRICE_COLS = ["price_min", "price_max", "price_avg", "price_early", "price_mid", "price_late"]

BOOL_COLS = [
    "reserved", "booster", "promo", "reprint", "borderless", "full_art", "textless",
    "story_spotlight", "game_changer", "can_lead_commander", "can_lead_oathbreaker",
    "can_lead_brawl",
]

NUMERIC_COLS = [
    "conv_mana_cost", "edhrec_rank", "edhrec_salt", "penny_rank", "num_paper_printings",
]

CATEGORICAL_COLS = [
    "lang", "set_type", "rarity", "finish", "layout", "border_color", "frame",
    "watermark", "security_stamp", "artist",
]

VARIABLE_VALUE_COLS = ["power", "toughness", "loyalty", "defense"]


def _subtype_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    price_usd = [c for c in df.columns if c.endswith("_price_usd")]
    product_id = [c for c in df.columns if c.endswith("_product_id")]
    return price_usd, product_id


def _parse_variable_value(series: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    is_variable = series.notna() & numeric.isna()
    return pd.DataFrame({series.name: numeric, f"{series.name}_is_variable": is_variable})


def _explode_fixed_vocab(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """colors/color_identity/produced_mana/color_indicator: fixed 5-letter
    WUBRG vocabulary, no top-N needed -- one boolean column per color."""
    out = {}
    for letter in COLOR_LETTERS:
        out[f"{prefix}_{letter}"] = df[col].apply(lambda tags: letter in tags if tags is not None else False)
    return pd.DataFrame(out)


def _explode_top_n(df: pd.DataFrame, col: str, top_n: int | None) -> pd.DataFrame:
    from collections import Counter
    counts = Counter()
    for tags in df[col]:
        if tags is not None:
            counts.update(tags)

    keep = [tag for tag, _ in counts.most_common(top_n)] if top_n else list(counts)
    keep_set = set(keep)

    out = {f"{col}__{tag}": df[col].apply(lambda tags: tag in tags if tags is not None else False) for tag in keep}
    if top_n:
        out[f"{col}__other"] = df[col].apply(
            lambda tags: tags is not None and len(tags) > 0 and any(t not in keep_set for t in tags)
        )
    return pd.DataFrame(out)


def build() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_parquet(DATASET_PATH)

    price_usd_cols, product_id_cols = _subtype_cols(df)
    df = df.drop(columns=DROP_COLS + price_usd_cols + product_id_cols)

    group_key = df["oracle_id"]
    y = np.log(df["price_avg"])
    has_target = df["price_avg"].notna()

    feature_frames = [
        df[BOOL_COLS + NUMERIC_COLS + CATEGORICAL_COLS].reset_index(drop=True),
    ]
    for col in VARIABLE_VALUE_COLS:
        feature_frames.append(_parse_variable_value(df[col]).reset_index(drop=True))
    for col, prefix in [("colors", "color"), ("color_identity", "color_identity"),
                        ("produced_mana", "produced_mana"), ("color_indicator", "color_indicator")]:
        feature_frames.append(_explode_fixed_vocab(df, col, prefix).reset_index(drop=True))
    for col, top_n in TAG_TOP_N.items():
        feature_frames.append(_explode_top_n(df, col, top_n).reset_index(drop=True))

    legalities_cols = [c for c in df.columns if c.startswith("legalities.")]
    legal_bools = pd.DataFrame({
        c.replace("legalities.", "legal_"): (df[c] == "legal").reset_index(drop=True)
        for c in legalities_cols
    })
    feature_frames.append(legal_bools)

    rate_cols = [c for c in df.columns if c.endswith("_rate_percent")]
    feature_frames.append(df[rate_cols].reset_index(drop=True))

    X = pd.concat(feature_frames, axis=1)
    X = X[has_target.reset_index(drop=True)].reset_index(drop=True)
    y = y[has_target].reset_index(drop=True)
    group_key = group_key[has_target].reset_index(drop=True)

    return X, y, group_key


def main() -> int:
    X, y, group_key = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    out = X.copy()
    out["__target_log_price_avg"] = y
    out["__group_oracle_id"] = group_key
    out.to_parquet(OUT_PATH, index=False)
    print(f"{len(out)} rows x {len(X.columns)} feature cols (+ target + group key) -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
