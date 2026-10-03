# embeddings

Local only, gitignored (large arrays, fully re-derivable) -- frozen
embeddings for Model 2 (SPECS.md §4.2). Not fine-tuned (§4.3, deferred
deliberately).

## Text

`scripts/embed_text.py` -- `sentence-transformers/all-mpnet-base-v2` (768-
dim), a conventional general-purpose sentence embedding model, **not**
CLIP's text tower (CLIP's text encoder aligns to image captions, not
templated MTG rules text -- decided earlier in this project).

Embedded text is a card's own `oracle_text` **plus, when applicable, the
text of what it creates or melds into** -- resolved from `all_parts`
(only the `token` and `meld_result` components; `combo_piece`, the
overwhelming majority of `all_parts` entries, is Scryfall's own "related/
synergy cards" list, not something physically part of the card). A card's
own oracle_text often just says "create a 1/1 Spirit token" without
saying what that token *does* -- without this, the embedding would be
blind to that.

Guards against a Scryfall data quirk: a meld *result* card's own
`all_parts` lists itself as a `meld_result` component (under a different
printing's `scryfall_id`, so it isn't caught by an id check) -- 33 of 149
meld-augmented rows would otherwise duplicate their own oracle_text onto
themselves. Skipped by name match in `_extra_text`.

Deduplicated by the constructed text itself, not embedded once per row --
149,078 rows collapse to 30,698 unique texts (many rows share identical
oracle_text, e.g. every finish variant of the same printing).

- `text_embeddings.parquet` -- `text_id`, `embedding_text`, `embedding` (one row per *unique* text)
- `text_row_index.parquet` -- `printing_key`, `text_id` (join this to `mtg-card-market-dataset` to get every row's embedding)

## Images

`scripts/embed_images.py` -- CLIP ViT-B/32 (OpenAI's original checkpoint,
via `open_clip`). **Must load as `ViT-B-32-quickgelu`, not `ViT-B-32`** --
OpenAI's original training used the QuickGELU activation, not the newer
default GELU `open_clip` ships now; loading the wrong one silently gives a
subtly-mismatched model with no error, just a warning easy to miss.

Embedded over the `art_crop` images from `card_art/` (see that folder's
README for why art-only, not the full card). One embedding per distinct
`illustration_id`, matching how the images themselves were fetched --
47,461 rows, no dedup needed here since fetching was already deduped.

- `image_embeddings.parquet` -- `illustration_id`, `embedding`
