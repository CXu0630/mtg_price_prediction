# card_art

Local only, gitignored (~3GB of images) -- re-derivable via
`scripts/fetch_card_art.py`. Card art for Model 2's image embedding step
(SPECS.md §4.2).

## Why `art_crop`, not the full card image

A full card image bakes in two things we don't want in an "art style"
embedding:

1. **Redundancy with existing features** -- frame/border-treatment visual
   cues (legendary crown, showcase, extended art, borderless) are already
   explicit categorical features (`frame_effects`, `borderless`,
   `full_art`); spending embedding capacity re-deriving them from pixels
   is wasted capacity that isn't spent on the actually-new signal.
2. **CLIP can partially read the card.** CLIP models have documented
   text-reading/OCR-like behavior (the same mechanism behind the
   "typographic attack" line of CLIP research, e.g. Goh et al.,
   "Multimodal Neurons," 2021) -- a full card image risks the "art"
   embedding partly encoding a noisy version of the oracle text we
   already have as clean raw text, working against the point of having a
   separate text modality at all.

Scryfall's own `image_uris.art_crop` (no border, no name, no rules text)
solves both -- not something built with our own cropping/CV logic, Scryfall
generates and hosts it directly.

**Known residual caveat**: `art_crop` isn't perfectly uniform across
treatments -- full-art lands, showcase, and borderless cards genuinely show
more art context (different crop proportions) than a normal-frame card.
Smaller effect than having the whole card in frame, and it's information
those treatment features already carry anyway, but not zero.

## Fetched by `illustration_id`, not by row

Many printings reuse identical art (confirmed earlier: all 953 Forest
printings map to far fewer distinct illustrations) -- fetching per
`(printing, finish)` row would download the same image dozens of times.
`illustration_id` is the natural dedup key, already present in
`mtg-card-market-dataset`.

## Source: Scryfall's `unique_artwork` bulk data

One JSON object per distinct illustration (`illustration_id` +
`image_uris.art_crop`, among other fields) -- fetched via the bulk-data
API (`https://api.scryfall.com/bulk-data`, `type: unique_artwork`) rather
than one API call per card, since that would mean ~47k individual
metadata requests instead of one ~38MB download. Checked: **100%
coverage** (47,461/47,475 of our dataset's distinct `illustration_id`s
found in it).

## Files

- `unique_artwork.jsonl.gz` -- the bulk data file itself, cached (re-fetch
  with `--refresh` if it goes stale; Scryfall regenerates it periodically).
- `images/{illustration_id}.jpg` -- one file per distinct illustration.
- `manifest.parquet` -- `illustration_id`, `art_crop_url`, `local_path`,
  `status` (`downloaded` / `cached` / `not_found` / `missing_from_bulk_data`
  / `error: ...`) for every illustration_id in the dataset, so gaps are
  visible rather than silently missing.
