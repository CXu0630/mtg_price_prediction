"""Builds Model 2's training tables -- local only (gitignored), not published to HF
(see training/README.md). Same target/split philosophy as Model 1
(scripts/prepare_training_data.py; see its docstring and SPECS.md S3/S4.1),
extended for the deep-kernel GP's wide+deep architecture (SPECS.md S4.2):

- **Wide tabular features**: mostly the same as Model 1 (fixed-vocab
  colors, top-N tags, legalities, variable power/toughness parsing), minus
  raw text/oracle_text (replaced by the dense text embedding) and minus
  `set`/`set_name`/`set_id` (same generalization argument as Model 1).
- **Off-card features get the inference input contract from SPECS.md
  S4.2**, not just dropped or passed through raw:
  - `{subtype}_rate_percent`: real value when the card is a known printing,
    else the (rarity, finish) default from
    `training/pull_rate_defaults.parquet` -- plus a per-subtype
    `_is_default` flag so the model can tell real from typical.
  - `edhrec_rank`, `edhrec_salt`, `penny_rank`, `num_paper_printings`: no
    principled default exists for these (rank especially is a relative
    ordering over the whole card pool) -- left as raw NaN + a companion
    `_isna` flag here; `train_model2.py` median-imputes using the *train
    split's* median only, computed after the group split, to avoid leaking
    test-set distribution into a training statistic.
- **Embeddings**: joined by `printing_key` (text, via
  `embeddings/text_row_index.parquet` -> `embeddings/text_embeddings.parquet`)
  and by `illustration_id` (image, via `embeddings/image_embeddings.parquet`).
  A card missing an image (no illustration_id, or art_crop fetch failed --
  518 + 14 rows) gets a zero vector + a `has_image` flag, rather than being
  dropped -- the model should still be able to price a card art hasn't been
  fetched for yet, from text/tabular alone.

Outputs (all row-aligned by position, not just by key):
- `training/features_model2.parquet` -- wide tabular features +
  `__target_log_price_avg` + `__group_oracle_id` + `__printing_key` (kept
  for traceability/debugging, not a model feature) + `has_image`.
- `training/model2_text_emb.npy` -- (N, 768) float32.
- `training/model2_image_emb.npy` -- (N, 512) float32, zero rows where
  `has_image` is False.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
PULL_RATE_DEFAULTS_PATH = ROOT / "training" / "pull_rate_defaults.parquet"
TEXT_ROW_INDEX_PATH = ROOT / "embeddings" / "text_row_index.parquet"
TEXT_EMB_PATH = ROOT / "embeddings" / "text_embeddings.parquet"
IMAGE_EMB_PATH = ROOT / "embeddings" / "image_embeddings.parquet"
OUT_DIR = ROOT / "training"

TAG_TOP_N = {
    "keywords": 50,
    "oracle_tags": 100,
    "art_tags": 100,
    "type_tags": 60,
    "promo_types": 20,
    "frame_effects": None,
}

# oracle_tags/art_tags are the two messiest, highest-cardinality tag
# columns (4,332 / 11,220 distinct values -- community-tagged, mixing real
# signal with naming trivia). Selecting their top-N by raw frequency picks
# the most *common* tags, not the most price-*relevant* ones -- a common
# tag like "vanilla-creature" crowds out a rarer but genuinely
# price-moving one like "extra-turn" purely on count. keywords/type_tags/
# promo_types come from curated, structured vocabularies (official
# keyword-ability lists, type-line-derived, a fixed promo enum) rather
# than free-form community tagging, so there's no equivalent evidence they
# have this problem -- left on frequency selection.
CORRELATION_SELECTED_COLS = {"oracle_tags", "art_tags"}
MIN_TAG_COUNT_FOR_SELECTION = 30

COLOR_LETTERS = ["W", "U", "B", "R", "G"]

DROP_COLS = [
    "scryfall_id", "printing_key", "uuid", "name",
    "set_id", "set", "set_name", "collector_number", "variation_of", "tcgplayer_id",
    "artist_ids", "illustration_id",
    "all_parts", "attraction_lights",
    "mana_cost", "type_line", "oracle_text",
]

BOOL_COLS = [
    "reserved", "booster", "promo", "reprint", "borderless", "full_art", "textless",
    "story_spotlight", "game_changer", "can_lead_commander", "can_lead_oathbreaker",
    "can_lead_brawl",
]

NUMERIC_COLS = ["conv_mana_cost"]

# No principled default -- median-impute (train-only) + _isna flag.
EXTERNAL_NO_DEFAULT_COLS = ["edhrec_rank", "edhrec_salt", "penny_rank", "num_paper_printings"]

CATEGORICAL_COLS = [
    "lang", "set_type", "rarity", "finish", "layout", "border_color", "frame",
    "watermark", "security_stamp", "artist",
]

VARIABLE_VALUE_COLS = ["power", "toughness", "loyalty", "defense"]


def _parse_variable_value(series: pd.Series) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    is_variable = series.notna() & numeric.isna()
    return pd.DataFrame({series.name: numeric, f"{series.name}_is_variable": is_variable})


def _explode_fixed_vocab(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    out = {}
    for letter in COLOR_LETTERS:
        out[f"{prefix}_{letter}"] = df[col].apply(lambda tags: letter in tags if tags is not None else False)
    return pd.DataFrame(out)


def _select_top_n_by_frequency(df: pd.DataFrame, col: str, top_n: int) -> list[str]:
    from collections import Counter
    counts = Counter()
    for tags in df[col]:
        if tags is not None:
            counts.update(tags)
    return [tag for tag, _ in counts.most_common(top_n)]


def _select_top_n_by_price_correlation(df: pd.DataFrame, col: str, top_n: int, min_count: int) -> list[str]:
    """Ranks candidate tags by |Pearson correlation| with log(price_avg),
    not raw frequency -- a common tag isn't necessarily a price-relevant
    one, and a rare-but-important tag (e.g. "extra-turn") shouldn't lose
    its slot to a common-but-irrelevant one (e.g. "vanilla-creature") just
    because it's less frequent. `min_count` floors which tags are even
    considered, since a tag on a handful of cards gives an unstable
    correlation estimate that's more likely to be noise than signal.

    Uses a sparse tag-occurrence matrix (most candidate tags survive the
    min_count floor here -- thousands for oracle_tags/art_tags -- and a
    dense one-hot matrix at this row count would be gigabytes) and the
    closed-form Pearson correlation for a binary variable, so this scales
    to all rows/candidates without one Python-level `.apply()` per tag.
    """
    from collections import Counter
    from scipy import sparse

    counts = Counter()
    for tags in df[col]:
        if tags is not None:
            counts.update(tags)
    candidates = [tag for tag, c in counts.items() if c >= min_count]
    if not candidates:
        return []
    tag_to_idx = {tag: i for i, tag in enumerate(candidates)}

    has_price = df["price_avg"].notna()
    y = np.log(df.loc[has_price, "price_avg"].to_numpy())
    tag_lists = df.loc[has_price, col].tolist()

    rows, cols = [], []
    for row_i, tags in enumerate(tag_lists):
        if tags is None:
            continue
        for tag in tags:
            j = tag_to_idx.get(tag)
            if j is not None:
                rows.append(row_i)
                cols.append(j)
    X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(tag_lists), len(candidates)))

    n = len(y)
    sum_y, sum_y2 = y.sum(), (y ** 2).sum()
    count_j = np.asarray(X.sum(axis=0)).ravel()  # binary column -> sum == sum of squares
    sum_xy_j = np.asarray(X.T @ y).ravel()

    numerator = n * sum_xy_j - count_j * sum_y
    denom = np.sqrt(count_j * (n - count_j)) * np.sqrt(n * sum_y2 - sum_y ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, numerator / denom, 0.0)

    order = np.argsort(-np.abs(corr))[:top_n]
    return [candidates[i] for i in order]


def _select_all_seen(df: pd.DataFrame, col: str) -> list[str]:
    """Every distinct tag, in deterministic first-seen order -- NOT a raw
    `set`/set-comprehension, whose iteration order depends on Python's
    per-process string hash randomization and is not reproducible across
    runs even for identical input (a real bug caught here: it silently
    permuted `frame_effects`' column order between two runs, corrupting a
    reloaded checkpoint's features without erroring)."""
    from collections import Counter
    counts = Counter()
    for tags in df[col]:
        if tags is not None:
            counts.update(tags)
    return list(counts)


def _explode_top_n(df: pd.DataFrame, col: str, top_n: int | None, keep: list[str] | None = None) -> pd.DataFrame:
    if keep is None:
        keep = _select_top_n_by_frequency(df, col, top_n) if top_n else _select_all_seen(df, col)
    keep_set = set(keep)

    out = {f"{col}__{tag}": df[col].apply(lambda tags: tag in tags if tags is not None else False) for tag in keep}
    if top_n:
        out[f"{col}__other"] = df[col].apply(
            lambda tags: tags is not None and len(tags) > 0 and any(t not in keep_set for t in tags)
        )
    return pd.DataFrame(out)


def _fill_pull_rates(df: pd.DataFrame) -> pd.DataFrame:
    rate_cols = [c for c in df.columns if c.endswith("_rate_percent")]
    defaults = pd.read_parquet(PULL_RATE_DEFAULTS_PATH).set_index(["rarity", "finish"])

    default_vals = df.set_index(["rarity", "finish"]).index.map(
        lambda k: defaults.loc[k] if k in defaults.index else pd.Series({c: np.nan for c in rate_cols})
    )
    default_df = pd.DataFrame(list(default_vals), columns=rate_cols).reset_index(drop=True)

    out = {}
    real = df[rate_cols].reset_index(drop=True)
    for col in rate_cols:
        is_default = real[col].isna()
        out[col] = real[col].where(~is_default, default_df[col])
        out[f"{col}__is_default"] = is_default
    return pd.DataFrame(out)


def build(tag_selection: str = "frequency") -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """`tag_selection`: "frequency" (top-N most common, the original
    method) or "correlation" (top-N by |correlation| with log-price for
    oracle_tags/art_tags specifically, SPECS.md S4.2's tag-selection
    follow-up). Both are archived under distinct output filenames by
    main() -- never overwrite one in place with the other, since a
    checkpoint trained against one is silently misinterpretable against
    the other (same column count, different column meanings)."""
    df = pd.read_parquet(DATASET_PATH)

    illustration_id = df["illustration_id"].reset_index(drop=True)
    printing_key = df["printing_key"].reset_index(drop=True)
    df = df.drop(columns=DROP_COLS)

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
        if tag_selection == "correlation" and col in CORRELATION_SELECTED_COLS and top_n:
            keep = _select_top_n_by_price_correlation(df, col, top_n, MIN_TAG_COUNT_FOR_SELECTION)
        else:
            keep = None
        feature_frames.append(_explode_top_n(df, col, top_n, keep=keep).reset_index(drop=True))

    legalities_cols = [c for c in df.columns if c.startswith("legalities.")]
    legal_bools = pd.DataFrame({
        c.replace("legalities.", "legal_"): (df[c] == "legal").reset_index(drop=True)
        for c in legalities_cols
    })
    feature_frames.append(legal_bools)

    feature_frames.append(_fill_pull_rates(df))

    external = df[EXTERNAL_NO_DEFAULT_COLS].reset_index(drop=True)
    for col in EXTERNAL_NO_DEFAULT_COLS:
        external[f"{col}__isna"] = external[col].isna()
    feature_frames.append(external)

    feature_frames.append(illustration_id.notna().rename("has_image").to_frame())

    X = pd.concat(feature_frames, axis=1)

    keep = has_target.reset_index(drop=True)
    X = X[keep].reset_index(drop=True)
    y = y[has_target].reset_index(drop=True)
    group_key = group_key[has_target].reset_index(drop=True)
    illustration_id = illustration_id[keep].reset_index(drop=True)
    printing_key = printing_key[keep].reset_index(drop=True)

    text_emb = _join_text_embeddings(printing_key)
    image_emb = _join_image_embeddings(illustration_id)

    return X, text_emb, image_emb, y, group_key, printing_key


def _join_text_embeddings(printing_key: pd.Series) -> np.ndarray:
    row_index = pd.read_parquet(TEXT_ROW_INDEX_PATH).set_index("printing_key")["text_id"]
    text_embeddings = pd.read_parquet(TEXT_EMB_PATH).set_index("text_id")["embedding"]
    text_ids = printing_key.map(row_index)
    return np.stack(text_ids.map(text_embeddings).to_numpy()).astype(np.float32)


def _join_image_embeddings(illustration_id: pd.Series) -> np.ndarray:
    image_embeddings = pd.read_parquet(IMAGE_EMB_PATH).set_index("illustration_id")["embedding"]
    dim = len(image_embeddings.iloc[0])
    zero = np.zeros(dim, dtype=np.float32)
    embs = illustration_id.map(lambda iid: image_embeddings.get(iid, zero) if pd.notna(iid) else zero)
    return np.stack(embs.to_numpy()).astype(np.float32)


def main(tag_selection: str = "frequency") -> int:
    """Writes `features_model2.parquet` when `tag_selection == "frequency"`
    (the live default every training script reads unless told otherwise --
    it's the empirically best-performing choice on the calibrated buckets
    that matter, SPECS.md S4.2), or the archived
    `features_model2_corr_tags.parquet` for "correlation" -- never the
    same filename for both, so a checkpoint trained against either stays
    safely reloadable."""
    X, text_emb, image_emb, y, group_key, printing_key = build(tag_selection=tag_selection)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = X.copy()
    out["__target_log_price_avg"] = y
    out["__group_oracle_id"] = group_key
    out["__printing_key"] = printing_key

    suffix = "" if tag_selection == "frequency" else f"_{tag_selection}_tags"
    out_path = OUT_DIR / f"features_model2{suffix}.parquet"
    out.to_parquet(out_path, index=False)
    # text/image embeddings are unaffected by tag selection (row set/order
    # never changes) -- only written once, alongside the live default.
    if tag_selection == "frequency":
        np.save(OUT_DIR / "model2_text_emb.npy", text_emb)
        np.save(OUT_DIR / "model2_image_emb.npy", image_emb)

    print(f"{len(out)} rows x {len(X.columns)} wide feature cols (+ target + group key) -> {out_path.name}")
    print(f"text_emb {text_emb.shape}, image_emb {image_emb.shape}")
    print(f"has_image: {out['has_image'].sum()}/{len(out)}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(tag_selection=sys.argv[1] if len(sys.argv) > 1 else "frequency"))
