# mtg-card-prices (source: pcwoods/mtg-card-prices)

Pulled from https://huggingface.co/datasets/pcwoods/mtg-card-prices on 2026-09-24.
Not this project's own dataset -- third-party, per-printing card data. Re-pull with
`scripts/fetch_mtg_card_prices.py` (parquet URLs come from the HF datasets-server
`/parquet` endpoint for this dataset).

## raw/

- `cards.parquet` -- 121,057 rows x 116 cols. One row per card **printing**
  (`scryfall_id`), not per card concept -- e.g. Forest has 953 rows, one per
  distinct set/art, all sharing one `oracle_id` (Scryfall groups by rules-text
  identity; basic lands are functionally identical across every printing).
  Includes oracle text, type/mana cost, official `keywords`, `art_tags`,
  artist, set/treatment fields, `edhrec_rank`, current USD/EUR prices.
  **No image files** -- pull art from Scryfall's `image_uris` via `scryfall_id`
  if/when needed for Model 2 (SPECS.md §4.2).
- `price_history.parquet` -- 65,797,283 rows x 9 cols. Long/tidy format:
  `scryfall_id, uuid, medium, provider, list_type, finish, currency, date, price`.
  Multiple providers per card (e.g. manapool), daily granularity. This is what
  unblocks the temporal-robustness backtest and price-normalization baseline
  in SPECS.md §3/§6.

## Known gotchas

- `tcgplayerProductId`/`tcgplayer_id` here is a **per-card** TCGplayer product
  ID -- not the same namespace as `mtg-print-distribution`'s
  `booster_products.tcgplayer_product_id`, which is a **sealed pack/box**
  product ID. Don't join on it.
- **`set` + `collector_number` is not a valid join key to
  `mtg-print-distribution.booster_rates`, despite looking like one.** A
  card's set/collector_number don't determine which booster *product* it's
  actually obtainable from -- that's decided by MTGJSON's per-product
  `sheets`, which `booster_rates` doesn't retain at the per-card level. See
  `mtg-merged/README.md` and SPECS.md §2 for the full finding; this join is
  currently blocked, not just imprecise.
- `printings` (on `cards.parquet`) is a deduplicated list of **set codes**
  sharing a card's `oracle_id`, not a list of individual printings/arts --
  a single set can contain multiple distinct arts of the same card under one
  set code, so `len(printings)` undercounts actual printing rows.
