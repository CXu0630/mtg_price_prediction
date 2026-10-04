"""Custom-card API (docs/APP_DESIGN.md §6). Existing printings never come here -- the
frontend reads them from the static JSON the nightly pipeline publishes.

    GET  /health    price date, model versions, whether the GP backend is warm
    POST /predict   {"backend": "a" | "gp", "cards": [CustomCard, ...]} -> {"rows": [...]}

Run locally:  uvicorn api.app:app --reload        (from the repo root)
Config (env): APP_ROOT (repo/data root, default: this repo), ALLOWED_ORIGINS (comma-separated,
default "*"), MAX_CARDS (default 50).
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mtg_price_prediction.cards import DEMAND_RANK, CustomCard
from mtg_price_prediction.scoring import AppModels, score_custom

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parents[1]))
MAX_CARDS = int(os.environ.get("MAX_CARDS", "50"))
MAX_ART_BYTES = 4_000_000


class CardIn(BaseModel):
    name: str = Field(max_length=200)
    rarity: Literal["common", "uncommon", "rare", "mythic", "special", "bonus"]
    type_line: str = Field(max_length=200)
    mana_cost: str = Field("", max_length=100)
    oracle_text: str = Field("", max_length=2000)
    power: str | None = None
    toughness: str | None = None
    loyalty: str | None = None
    defense: str | None = None
    finish: Literal["nonfoil", "foil", "etched"] = "nonfoil"
    set_type: str = "expansion"
    frame: str = "2015"
    border_color: str = "black"
    frame_effects: list[str] = []
    promo_types: list[str] = []
    keywords: list[str] = []
    full_art: bool = False
    textless: bool = False
    released_at: str | None = None
    demand: Literal[tuple(DEMAND_RANK)] = "playable"  # type: ignore[valid-type]
    reprint_of: str | None = None
    set: str | None = None
    artist: str | None = None
    quantity: int = Field(1, ge=1, le=10_000)
    art: str | None = Field(None, description="optional data URL / base64 image; used by the GP backend")


class PredictIn(BaseModel):
    backend: Literal["a", "gp"] = "a"
    cards: list[CardIn] = Field(min_length=1)


def _decode_art(art: str | None):
    if not art:
        return None
    from PIL import Image

    raw = base64.b64decode(art.split(",", 1)[-1], validate=False)
    if len(raw) > MAX_ART_BYTES:
        raise HTTPException(413, "art image larger than 4 MB")
    return Image.open(io.BytesIO(raw))


class State:
    dataset: pd.DataFrame
    models: AppModels
    live: pd.DataFrame
    as_of: pd.Timestamp


state = State()
app = FastAPI(title="MTG price-range API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.on_event("startup")
def load() -> None:
    state.dataset = pd.read_parquet(ROOT / "mtg-card-market-dataset" / "dataset.parquet")
    state.models = AppModels.load(ROOT / "models" / "app", ROOT / "models" / "deep_gp")
    state.live = pd.read_parquet(ROOT / "data" / "app" / "live_rows.parquet")
    state.as_of = pd.Timestamp(state.models.manifest["price_date"])


@app.get("/health")
def health() -> dict:
    m = state.models.manifest
    return {"ok": True, "price_date": m["price_date"], "model_a_as_of": m["model_a"]["as_of"],
            "model_b_cutoff": m["model_b"]["cutoff"], "model_c_cutoff": m["model_c"]["cutoff"],
            "gp_loaded": state.models._gp is not None}


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    if len(body.cards) > MAX_CARDS:
        raise HTTPException(413, f"at most {MAX_CARDS} cards per request")
    cards = [CustomCard(**c.model_dump(exclude={"art"})) for c in body.cards]
    images = [_decode_art(c.art) for c in body.cards] if body.backend == "gp" else None
    try:
        out = score_custom(cards, state.dataset, state.models, state.live, state.as_of, body.backend, images)
    except ValueError as e:  # bad reprint_of / finish / demand
        raise HTTPException(422, str(e)) from e
    rows = out.replace({np.nan: None}).to_dict("records")
    for r in rows:
        r["model_a"] = "deep GP" if body.backend == "gp" else "Model A"
        r["caveats"] = ["custom card: one day of price data (the Model A estimate) -- the 14-day forecast "
                        "has no momentum or cross-market signal", "Model C n/a: no Card Kingdom buylist"]
    return {"price_date": str(state.as_of.date()), "rows": rows}
