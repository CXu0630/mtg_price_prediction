"""Builds mtg-card-market-dataset/dataset.parquet, the working joined training dataset.

Grain is one row per (printing, finish) -- see mtg-card-market-dataset/README.md for why.
Currently just the per-card dataset (mtg-card-prices/raw/cards.parquet),
deduped, exploded by finish, with a few columns dropped: platform IDs we
don't need (SPECS.md doesn't use them for anything -- not join keys, not
features) and price, held back for now until the target definition
(SPECS.md §3, relative/log-price vs. a contemporaneous baseline) is settled.
Re-run as more sources (rarity/pull-rate, EDHREC) get joined in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CARDS_PATH = ROOT / "mtg-card-prices" / "raw" / "cards.parquet"
OUT_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"

# Platform/marketplace IDs not used as join keys or features -- pure clutter.
# resource_id: undocumented, 94% null, looks internal to this dataset's build
# (not a native Scryfall/MTGJSON field). card_back_id: which card-back image
# a printing uses -- not relevant to pricing. Keep only tcgplayer_id as the
# one external marketplace id -- mtgo_foil_id/cardKingdomId/mcmId dropped,
# tcgplayerProductId dropped too (duplicate of tcgplayer_id).
DROP_ID_COLS = [
    "multiverse_ids", "mtgo_id", "arena_id", "cardmarket_id", "image_updated_at",
    "resource_id", "card_back_id",
    "mtgo_foil_id", "cardKingdomId", "mcmId", "tcgplayerProductId",
]

# Held back until the target definition (SPECS.md §3) is settled -- not
# dropped for good, just not part of the feature set yet.
DROP_PRICE_COLS = ["prices.usd", "prices.usd_foil", "prices.usd_etched", "prices.eur", "prices.eur_foil", "prices.tix"]

# Dropped for good, once we're paper-only (see _drop_digital below):
# first/last_printed_at add little once digital-only rows -- the ones most
# likely to skew these -- are gone; games is redundant once every remaining
# row includes paper; variation is redundant with variation_of (which row
# it points to already tells you whether it's a variant); foil/nonfoil are
# printing-level flags, redundant with the exploded finish column.
DROP_MISC_COLS = [
    "first_printed_at", "last_printed_at", "games", "variation", "foil", "nonfoil",
    # Confirmed 100% empty on every remaining row (checked directly, not
    # assumed): life_modifier/hand_modifier are Vanguard-only and Vanguard
    # cards are oversized (already excluded, see _drop_oversized);
    # card_faces described multi-face cards, which are now excluded
    # entirely too (see MULTIFACE_LAYOUTS) -- dead weight, not just a
    # "maybe" drop candidate.
    "life_modifier", "hand_modifier", "card_faces",
    # printings: card-concept-level (not per-row), undercounts real
    # printings whenever a set has multiple arts under one code -- too
    # coarse to be a reliable "has this card been reprinted" feature.
    # content_warning: 99.98% null, not worth the column.
    "printings", "content_warning",
]

# Non-English print variants -- not needed unless modeling foreign-print
# price premiums, which isn't planned.
DROP_LOCALIZATION_COLS = ["printed_name", "printed_text", "printed_type_line", "flavor_name", "flavor_text"]

# layouts that aren't a purchasable playing card -- tokens, emblems, and
# other game accessories bundled into Scryfall's card data. Confirmed via
# type_line: these are junk values ("Card", "Card // Card") or non-card
# object types (Emblem, Scheme, Vanguard, Plane), not real MTG card types.
NONCARD_LAYOUTS = {
    "token", "emblem", "scheme", "vanguard", "planar", "art_series",
    "double_faced_token", "front_card",
}

# Real, purchasable multi-face cards -- but the top-level mana_cost/
# oracle_text/type_line/etc. columns are ~100% null for them (that data
# only lives in the unflattened card_faces struct, which has no flattening
# plan yet -- see COLUMNS.md group 5), and they're a source of uuid-match
# gaps in join_pull_rates.py too. Confirmed via name/oracle_text nullness:
# these 7 layouts are 100% "X // Y" names with 100% null oracle_text; no
# other layout shows that pattern (e.g. meld's front-half cards are
# complete single-name rows, kept). Only 2.4% of rows either way -- v1
# scope cut, drop entirely rather than carry half-populated rows; revisit
# if card_faces flattening ever gets built.
MULTIFACE_LAYOUTS = {
    "adventure", "flip", "modal_dfc", "prepare", "reversible_card", "split", "transform",
}

# promo_types values confirmed (against live TCGplayer product names, not
# just the tag name) to mark a genuinely distinct foil-treatment product --
# see mtg-card-market-dataset/README.md for the verification. Only ever applies to the
# 'foil' finish row. Values not in this list (e.g. doublerainbow, embossed,
# glossy, neonink, gleaminggold, moonlitland, schinesealtart) were checked
# and don't show up as a distinguishing trait in TCGplayer's own product
# naming, so they're left as plain promo_types, not treated as a treatment.
SPECIALTY_FOIL_TAGS = {
    "chocobotrackfoil", "confettifoil", "cosmicfoil", "dazzlefoil", "dragonscalefoil",
    "facetfoil", "galaxyfoil", "halofoil", "manafoil", "rainbowfoil", "raisedfoil",
    "ripplefoil", "silverfoil", "singularityfoil", "surgefoil", "textured",
    "oilslick", "stepandcompleat", "silverscroll",
}


def _specialty_foil_type(promo_types) -> str | None:
    if promo_types is None:
        return None
    hits = sorted(SPECIALTY_FOIL_TAGS & set(promo_types))
    if not hits:
        return None
    # e.g. Phyrexian Vindicator's "Oil Slick Raised Foil" splits across two
    # tags (oilslick, raisedfoil) for one physical treatment -- join rather
    # than pick one arbitrarily.
    return "_".join(hits)


def _explode_by_finish(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (printing, finish) instead of one row per printing with a
    finishes list -- each treatment (nonfoil/foil/etched) gets its own row,
    since price (when it comes back in) only makes sense per finish, not per
    printing. 3 digital-only rows have an empty finishes list and drop out
    here (no purchasable finish to explode into).

    `finish` holds the specialty-foil type directly (e.g. "silverfoil", not
    "foil") when one applies, matching mtg-print-distribution.booster_rates
    .finish's granularity -- a single column carrying the full treatment,
    not a coarse finish plus a separate flag."""
    df = df.explode("finishes").rename(columns={"finishes": "finish"})
    df = df.dropna(subset=["finish"])
    specialty = df["promo_types"].apply(_specialty_foil_type)
    has_specialty = (df["finish"] == "foil") & specialty.notna()
    df.loc[has_specialty, "finish"] = specialty[has_specialty]
    df["printing_key"] = df["scryfall_id"] + ":" + df["finish"]
    return df


def _clean_promo_types(df: pd.DataFrame) -> pd.DataFrame:
    """Specialty-foil tags are now represented in `finish` -- strip them out
    of promo_types so they aren't double-represented in two places."""
    def strip(tags):
        if tags is None:
            return None
        kept = [t for t in tags if t not in SPECIALTY_FOIL_TAGS]
        return kept or None

    df["promo_types"] = df["promo_types"].apply(strip)
    return df


def _add_borderless(df: pd.DataFrame) -> pd.DataFrame:
    """Split out as its own column rather than left as a border_color value
    -- matches mtg-print-distribution.booster_rates.borderless, "its own
    field, not a frame effect"."""
    df["borderless"] = df["border_color"] == "borderless"
    return df


def _drop_digital(df: pd.DataFrame) -> pd.DataFrame:
    """Digital-only (MTGO/Arena) printings have no TCGplayer product --
    confirmed 9,604/9,608 digital=True rows have a null tcgplayer_id -- so
    they carry no price signal and aren't purchasable the way this project
    means "printing." Drop before exploding."""
    return df[~df["digital"]].drop(columns=["digital"])


def _drop_oversized(df: pd.DataFrame) -> pd.DataFrame:
    """Oversized cards (Commander/Planechase/Archenemy oversized promos,
    etc.) aren't tournament-legal, purchasable singles in the sense this
    project cares about -- out of scope."""
    return df[~df["oversized"]].drop(columns=["oversized"])


def _drop_noncard_layouts(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tokens/emblems/schemes/etc. -- see NONCARD_LAYOUTS. `layout`
    itself is kept -- the remaining values (normal/split/saga/...) are
    real, meaningful card-structure distinctions."""
    return df[~df["layout"].isin(NONCARD_LAYOUTS)]


def _drop_multiface(df: pd.DataFrame) -> pd.DataFrame:
    """Drop real multi-face cards whose top-level fields are unusably
    incomplete -- see MULTIFACE_LAYOUTS."""
    return df[~df["layout"].isin(MULTIFACE_LAYOUTS)]


def _drop_no_uuid(df: pd.DataFrame) -> pd.DataFrame:
    """A handful of rows (layout='normal', so NONCARD_LAYOUTS misses them)
    are Scryfall-only accessory objects with no MTGJSON uuid at all --
    "Double-Faced Substitute Card", "Poison Counter", "Storm Counter",
    dungeon cards, etc. Not real MTGJSON-tracked cards, and uuid is now the
    join key to mtg-print-distribution, so they're dead weight either way.
    Confirmed precise: every null-uuid row is one of these, no real card
    in the dataset has a null uuid."""
    return df[df["uuid"].notna()]


def _add_type_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-hot-style tag list parsed straight out of type_line -- every
    word (supertype, card type, and subtype alike, e.g. "legendary",
    "creature", "human", "wizard") becomes its own tag, same list-of-tags
    shape as keywords/oracle_tags/art_tags. type_line's vocabulary is small
    and fixed (official supertypes/types/subtypes), so naive whitespace
    splitting is enough -- no need for an official type list to parse
    against. Handles multi-face type lines (" // ") and the em-dash
    separating types from subtypes (" — ")."""
    def tags(type_line):
        if not isinstance(type_line, str):
            return None
        words = set()
        for face in type_line.split(" // "):
            for part in face.split(" — "):
                words.update(w.lower() for w in part.split())
        return sorted(words) if words else None

    df["type_tags"] = df["type_line"].apply(tags)
    return df


def _merge_etched_id(df: pd.DataFrame) -> pd.DataFrame:
    """explode() duplicates non-list columns onto every finish row of a
    printing, so tcgplayer_id/tcgplayer_etched_id both land on all of a
    printing's nonfoil/foil/etched rows identically -- each row's real
    product id is now just "whichever of the two is this row's own", so
    fold etched's id into tcgplayer_id and drop the separate column."""
    is_etched = df["finish"] == "etched"
    df.loc[is_etched, "tcgplayer_id"] = df.loc[is_etched, "tcgplayer_etched_id"]
    df = df.drop(columns=["tcgplayer_etched_id"])
    return df


def build() -> pd.DataFrame:
    df = pd.read_parquet(CARDS_PATH)
    df = df.drop(columns=DROP_ID_COLS + DROP_PRICE_COLS + DROP_MISC_COLS + DROP_LOCALIZATION_COLS)
    df = df.rename(columns={"cmc": "conv_mana_cost"})

    # Source has exact duplicate rows (2,448/121,057 scryfall_ids repeated,
    # mostly 2x, a few 3x/5x -- same id, same values throughout, not a
    # finish/printing distinction, just a dupe in the upstream build).
    n_before = len(df)
    df = df.drop_duplicates(subset="scryfall_id", keep="first")
    print(f"  deduped {n_before - len(df)} exact-duplicate scryfall_id rows")

    n_before = len(df)
    df = _drop_digital(df)
    print(f"  dropped {n_before - len(df)} digital-only rows")

    n_before = len(df)
    df = _drop_oversized(df)
    print(f"  dropped {n_before - len(df)} oversized rows")

    n_before = len(df)
    df = _drop_noncard_layouts(df)
    print(f"  dropped {n_before - len(df)} non-card-layout rows (tokens/emblems/etc.)")

    n_before = len(df)
    df = _drop_multiface(df)
    print(f"  dropped {n_before - len(df)} multi-face-card rows (transform/split/adventure/etc.)")

    n_before = len(df)
    df = _drop_no_uuid(df)
    print(f"  dropped {n_before - len(df)} rows with no MTGJSON uuid (non-card accessory objects)")

    df = _add_type_tags(df)

    n_before = len(df)
    df = _explode_by_finish(df)
    print(f"  exploded {n_before} printings -> {len(df)} (printing, finish) rows")

    df = _merge_etched_id(df)
    df = _clean_promo_types(df)
    df = _add_borderless(df)
    return df


def main() -> int:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"{len(df)} rows x {len(df.columns)} cols -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
