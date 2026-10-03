"""Text embeddings for Model 2 (SPECS.md §4.2) -- a conventional general-
purpose sentence-embedding model over oracle text, NOT CLIP's text tower
(CLIP's text encoder aligns to image captions, not templated rules text --
decided earlier in this project).

Extends the embedded text beyond a card's own oracle_text: if the card
creates a token, or is a meld card, that token's/meld result's own name +
type + rules text is concatenated on -- "Creates: Spirit -- Token Creature
Spirit. Flying." or "Melds into: Brisela, Voice of Nightmares -- ...". A
card's own oracle_text often doesn't say what the thing it makes actually
*does* (just "create a 1/1 Spirit token"), so without this the embedding
would be blind to that. Resolved from `all_parts` (a JSON string of
related-card refs Scryfall attaches to a card) -- only the `token` and
`meld_result` components are used; `combo_piece` (the overwhelming
majority of entries -- 152,887 of ~170k total) is Scryfall's own
"related/synergy cards" list, not something physically part of this card,
and `meld_part` (the *other* meld half, not the combined result) is left
out for now, easy to add if wanted.

Deduplicated by the actual constructed text (many rows share identical
oracle_text -- every finish variant of the same printing, for a start), not
embedded once per row.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
CARDS_RAW_PATH = ROOT / "mtg-card-prices" / "raw" / "cards.parquet"
OUT_DIR = ROOT / "embeddings"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
WANTED_COMPONENTS = {"token", "meld_result"}
COMPONENT_LABELS = {"token": "Creates", "meld_result": "Melds into"}


def _build_lookup() -> dict[str, dict]:
    cards = pd.read_parquet(CARDS_RAW_PATH, columns=["scryfall_id", "name", "type_line", "oracle_text"])
    return {
        row.scryfall_id: {"name": row.name, "type_line": row.type_line, "oracle_text": row.oracle_text}
        for row in cards.itertuples()
    }


def _extra_text(all_parts_json: str | None, lookup: dict[str, dict], own_name: str) -> str:
    if not all_parts_json or not isinstance(all_parts_json, str):
        return ""
    parts = json.loads(all_parts_json)
    pieces = []
    for p in parts:
        component = p.get("component")
        if component not in WANTED_COMPONENTS:
            continue
        related = lookup.get(p.get("id"))
        if related is None:
            continue
        if related["name"] == own_name:
            # Scryfall's own all_parts for a meld *result* card lists itself
            # as a meld_result component (under a different scryfall_id --
            # a different printing than the one in all_parts). Skip it so a
            # result card doesn't get its own oracle_text duplicated onto
            # itself.
            continue
        label = COMPONENT_LABELS[component]
        bits = [related["name"]]
        if related.get("type_line"):
            bits.append(f"-- {related['type_line']}")
        if related.get("oracle_text"):
            bits.append(f". {related['oracle_text']}")
        pieces.append(f"{label}: {' '.join(bits)}")
    return " ".join(pieces)


def build_embedding_text() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH, columns=["printing_key", "name", "oracle_text", "all_parts"])
    lookup = _build_lookup()

    extra = df.apply(lambda r: _extra_text(r["all_parts"], lookup, r["name"]), axis=1)
    df["embedding_text"] = (df["oracle_text"].fillna("") + " " + extra).str.strip()
    return df[["printing_key", "embedding_text"]]


def main() -> int:
    df = build_embedding_text()
    unique_texts = df["embedding_text"].drop_duplicates().reset_index(drop=True)
    text_to_id = {t: i for i, t in enumerate(unique_texts)}
    print(f"{len(df)} rows -> {len(unique_texts)} unique embedding texts "
          f"({len(unique_texts)/len(df)*100:.1f}% unique)")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(unique_texts.tolist(), batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text_embeddings = pd.DataFrame({
        "text_id": range(len(unique_texts)),
        "embedding_text": unique_texts,
    })
    text_embeddings["embedding"] = list(embeddings)
    text_embeddings.to_parquet(OUT_DIR / "text_embeddings.parquet", index=False)

    row_to_text = pd.DataFrame({
        "printing_key": df["printing_key"],
        "text_id": df["embedding_text"].map(text_to_id),
    })
    row_to_text.to_parquet(OUT_DIR / "text_row_index.parquet", index=False)

    print(f"model: {MODEL_NAME}, dim: {embeddings.shape[1]}")
    print(f"-> {OUT_DIR / 'text_embeddings.parquet'} ({len(text_embeddings)} rows)")
    print(f"-> {OUT_DIR / 'text_row_index.parquet'} ({len(row_to_text)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
