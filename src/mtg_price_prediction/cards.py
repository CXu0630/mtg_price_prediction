"""Custom (hypothetical) cards -> dataset-shaped rows, so every model reads custom and
existing printings through the same feature builders (docs/APP_DESIGN.md §2, §4).

Only what's printed on the card or declared at design time is required. The rest:
  - EDHREC rank      <- `demand` tier (EDHREC is the #1 driver of every model; see SPECS)
  - pull rates       <- (rarity, finish) defaults, play + collector boosters only for
                        expansion/core/masters sets (claiming every product fakes supply)
  - legalities       <- the real card's when `reprint_of` is given, else eternal formats,
                        plus standard-era formats for expansion/core sets
  - tags             <- the real card's oracle tags when `reprint_of` is given, else none
  - printing history <- the real card's printings when `reprint_of` is given
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Built by scripts/build_pull_rate_defaults.py (training/, local); the published deep-GP repo ships a copy.
PULL_RATE_DEFAULTS = next((p for p in (ROOT / "training" / "pull_rate_defaults.parquet",
                                       ROOT / "models" / "deep_gp" / "pull_rate_defaults.parquet") if p.exists()),
                          ROOT / "training" / "pull_rate_defaults.parquet")

# EDHREC rank by demand tier: points on the dataset's per-card rank distribution
# (staple ~ top 2%, popular ~ top 5%, playable ~ top 25%, niche ~ median).
DEMAND_RANK = {"staple": 650.0, "popular": 1600.0, "playable": 8000.0, "niche": 16000.0, "unplayed": None}
BOOSTER_SET_TYPES = {"expansion", "core", "masters", "draft_innovation"}
ETERNAL = {"commander", "legacy", "vintage", "oathbreaker", "duel", "predh", "paupercommander"}
STANDARD_ERA = {"standard", "pioneer", "modern", "historic", "timeless", "brawl", "standardbrawl", "future",
                "gladiator", "alchemy"}
FINISHES = {"nonfoil", "foil", "etched"}


@dataclass
class CustomCard:
    name: str
    rarity: str                      # common / uncommon / rare / mythic / special / bonus
    type_line: str
    mana_cost: str = ""
    oracle_text: str = ""
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    defense: str | None = None
    finish: str = "nonfoil"
    set_type: str = "expansion"
    frame: str = "2015"
    border_color: str = "black"
    frame_effects: list[str] = field(default_factory=list)
    promo_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    full_art: bool = False
    textless: bool = False
    released_at: str | None = None   # ISO date; None = "today"
    demand: str = "playable"
    reprint_of: str | None = None    # a scryfall_id, oracle_id or exact card name of a real card
    set: str | None = None           # optional real set code, for set price context
    artist: str | None = None
    quantity: int = 1


def _symbols(text: str) -> list[str]:
    return re.findall(r"\{([^}]*)\}", text or "")


def mana_value(cost: str) -> float:
    total = 0.0
    for s in _symbols(cost):
        if s.isdigit():
            total += int(s)
        elif s in ("X", "Y", "Z"):
            continue
        else:  # colored, hybrid ({G/U} = 1), twobrid ({2/W} = 2), Phyrexian ({G/P} = 1)
            digits = [int(p) for p in s.split("/") if p.isdigit()]
            total += max(digits) if digits else 1
    return total


def colors_of(text: str) -> list[str]:
    return [c for c in "WUBRG" if any(c in s.split("/") for s in _symbols(text))]


def _resolve_reprint(ref: str | None, dataset: pd.DataFrame) -> pd.DataFrame:
    if not ref:
        return dataset.iloc[:0]
    for col in ("scryfall_id", "oracle_id", "name"):
        hit = dataset[dataset[col] == ref]
        if len(hit):
            return dataset[dataset["oracle_id"] == hit["oracle_id"].iloc[0]]
    raise ValueError(f"reprint_of {ref!r}: no card with that scryfall_id, oracle_id or name")


def custom_rows(cards: list[CustomCard], dataset: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One dataset-shaped row per custom card (printing_key "custom:<i>")."""
    defaults = pd.read_parquet(PULL_RATE_DEFAULTS).set_index(["rarity", "finish"])
    rate_cols = [c for c in dataset.columns if c.endswith("_rate_percent")]
    legal_cols = [c for c in dataset.columns if c.startswith("legalities.")]
    rows = []
    for i, c in enumerate(cards):
        if c.finish not in FINISHES:
            raise ValueError(f"{c.name}: finish must be one of {sorted(FINISHES)}")
        if c.demand not in DEMAND_RANK:
            raise ValueError(f"{c.name}: demand must be one of {list(DEMAND_RANK)}")
        real = _resolve_reprint(c.reprint_of, dataset)
        ref = real.iloc[0] if len(real) else None
        type_line = c.type_line or ""
        legendary_creature = "Legendary" in type_line and "Creature" in type_line
        row = {
            "printing_key": f"custom:{i}", "scryfall_id": f"custom:{i}", "name": c.name,
            "oracle_id": ref["oracle_id"] if ref is not None else f"custom:{i}",
            "reprint_of_oracle": ref["oracle_id"] if ref is not None else None,
            "set": (c.set or "").lower() or "custom", "set_type": c.set_type, "rarity": c.rarity, "finish": c.finish,
            "released_at": pd.Timestamp(c.released_at) if c.released_at else as_of,
            "mana_cost": c.mana_cost, "conv_mana_cost": mana_value(c.mana_cost), "type_line": type_line,
            "oracle_text": c.oracle_text, "power": c.power, "toughness": c.toughness, "loyalty": c.loyalty,
            "defense": c.defense, "colors": colors_of(c.mana_cost),
            "color_identity": sorted(set(colors_of(c.mana_cost)) | set(colors_of(c.oracle_text)), key="WUBRG".index),
            "produced_mana": sorted({s for m in re.findall(r"Add ((?:\{[WUBRGC]\})+)", c.oracle_text or "")
                                     for s in _symbols(m)}),
            "keywords": c.keywords, "type_tags": [w.lower() for w in re.split(r"[\s—-]+", type_line) if w and w != "//"],
            "oracle_tags": list(ref["oracle_tags"]) if ref is not None and ref["oracle_tags"] is not None else [],
            "art_tags": [], "frame": c.frame, "frame_effects": c.frame_effects, "promo_types": c.promo_types,
            "border_color": c.border_color, "borderless": c.border_color == "borderless",
            "full_art": c.full_art, "textless": c.textless, "layout": "normal", "lang": "en",
            "security_stamp": None, "watermark": None, "artist": c.artist,
            "reserved": False, "game_changer": False, "promo": bool(c.promo_types), "reprint": ref is not None,
            "booster": c.set_type in BOOSTER_SET_TYPES, "story_spotlight": False,
            "can_lead_commander": legendary_creature or "can be your commander" in (c.oracle_text or ""),
            "can_lead_oathbreaker": "Planeswalker" in type_line, "can_lead_brawl": legendary_creature or "Planeswalker" in type_line,
            "edhrec_rank": ref["edhrec_rank"] if ref is not None and pd.notna(ref["edhrec_rank"]) else DEMAND_RANK[c.demand],
            "edhrec_salt": ref["edhrec_salt"] if ref is not None else float(dataset["edhrec_salt"].median()),
            "penny_rank": None, "all_parts": None, "variation_of": None, "illustration_id": None,
            "num_paper_printings": (real["scryfall_id"].nunique() + 1) if ref is not None else 1,
        }
        for col in legal_cols:
            fmt = col.split(".", 1)[1]
            if ref is not None:
                row[col] = ref[col]
            else:
                row[col] = "legal" if fmt in ETERNAL or (fmt in STANDARD_ERA and c.set_type in ("expansion", "core")) else "not_legal"
        boosters = c.set_type in BOOSTER_SET_TYPES and (c.rarity, c.finish) in defaults.index
        for col in rate_cols:
            product = col.split("_rate")[0]
            row[col] = float(defaults.loc[(c.rarity, c.finish), col]) if boosters and product in ("play", "collector") else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
