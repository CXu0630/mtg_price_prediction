# mtg-card-market-dataset

The working joined training dataset. Built by `scripts/build_merged.py`, not
edited by hand -- re-run the script to regenerate `dataset.parquet` (gitignored,
same as `mtg-card-prices/raw/`).

## Current state

Source: `mtg-card-prices/raw/cards.parquet`, deduped on `scryfall_id`
(source has some exact-duplicate rows). Digital-only (MTGO/Arena) printings
dropped -- no TCGplayer product, no price signal, not purchasable the way
this project means "printing." Oversized cards dropped -- not tournament-legal
purchasable singles, out of scope. Non-card layouts dropped -- tokens,
emblems, schemes, vanguards, planes, art-series cards, and double-faced
tokens are game accessories bundled into Scryfall's card data, not
purchasable playing cards (confirmed via `type_line`: these show junk
values like `"Card"` / `"Card // Card"`, not real MTG types). Multi-face
cards dropped too -- `adventure, flip, modal_dfc, prepare, reversible_card,
split, transform` layouts (2.4% of rows) have ~100% null `oracle_text`/
`mana_cost` at the top level (that data only lives in the unflattened
`card_faces` struct, no flattening plan yet), so they're a v1 scope cut,
not a permanent exclusion. `meld`'s front-half cards are *not* included in
this -- confirmed they're complete single-name rows, a different mechanic.
75 more rows dropped for having no MTGJSON `uuid` at all -- Scryfall-only
accessory objects ("Double-Faced Substitute Card," "Poison Counter,"
dungeon cards, ...) that slip past the layout filters because their
`layout` is `normal`; confirmed precise (every null-`uuid` row in the
source is one of these, no real card lacks a uuid).

**Grain: one row per (printing, finish)**, not one row per printing. The
unique row ID is `printing_key` (`scryfall_id:finish`) -- `scryfall_id`
alone is not unique here. 149,078 rows.

`type_tags` is a multi-hot-style tag list parsed straight out of
`type_line` -- every word (supertype, type, and subtype alike, e.g.
`legendary`, `creature`, `human`, `wizard`) becomes its own tag, same shape
as `keywords`/`oracle_tags`/`art_tags`. Handles multi-face type lines and
the em-dash separating types from subtypes. `type_line`'s vocabulary is
small and fixed, so naive whitespace splitting is enough.

`cmc` is renamed `conv_mana_cost` (converted mana cost).

`finish` carries the full treatment granularity directly, one column --
`nonfoil`/`foil`/`etched`, or the specific specialty-foil name when one
applies (`surgefoil, galaxyfoil, silverfoil, ripplefoil, rainbowfoil,
halofoil, textured, stepandcompleat, silverscroll, manafoil, raisedfoil,
confettifoil, oilslick_raisedfoil, chocobotrackfoil, dazzlefoil,
dragonscalefoil, facetfoil, cosmicfoil, singularityfoil`) -- matching
`mtg-print-distribution.booster_rates.finish`'s structure: a single
granular column, not a coarse finish plus a separate specialty flag. Each
specialty value is confirmed against TCGplayer's own live product naming,
not just the source tag name -- see `scripts/build_merged.py`'s
`SPECIALTY_FOIL_TAGS` comment for what was checked and excluded. The
specialty-foil tags are stripped out of `promo_types` once folded into
`finish`, so they aren't double-represented in two places.

`borderless` is split out as its own boolean (`border_color == "borderless"`),
also matching `booster_rates.borderless` -- "its own field, not a frame
effect."

`tcgplayer_id` holds each row's own TCGplayer product id -- nonfoil and foil
share one product (two price points under it), etched is its own product.
Every other marketplace id (`mtgo_foil_id`, `cardKingdomId`, `mcmId`,
`tcgplayerProductId`) is dropped; `tcgplayer_id` is the only one kept.

Price columns (`prices.usd`, `prices.usd_foil`, `prices.usd_etched`,
`prices.eur`, `prices.eur_foil`, `prices.tix`) are dropped from the source
-- superseded by `price_min`/`price_max`/`price_avg`, see "Card price,
joined in" below. These single-snapshot columns aren't a loss: they're one
point-in-time value each, strictly less information than the min/max/avg
over `price_history.parquet`'s ~3-month window that replaces them.

A handful of other columns are dropped outright as not price-relevant or
redundant: `multiverse_ids`, `mtgo_id`, `arena_id`, `cardmarket_id`,
`image_updated_at`, `resource_id`, `card_back_id`, `first_printed_at`,
`last_printed_at`, `games` (redundant once every row is paper-only),
`variation` (redundant with `variation_of`), `digital` (constant `False`
once digital-only rows are dropped), `oversized` (constant `False` once
oversized rows are dropped), `foil`/`nonfoil` (printing-level flags,
redundant with `finish` at this grain), `life_modifier`/`hand_modifier`
(Vanguard-only, and Vanguard cards are oversized -- already excluded --
confirmed 100% empty, not just suspected), `card_faces` (described
multi-face cards, which are now excluded entirely too -- confirmed 100%
empty), and the whole localization group -- `printed_name`, `printed_text`,
`printed_type_line`, `flavor_name`, `flavor_text` (non-English print
variants / narrative flavor text, not needed unless modeling foreign-print
premiums), `printings` (card-concept-level, not per-row -- undercounts real
printings whenever a set has multiple arts under one code, too coarse to
be a reliable "has this card been reprinted" feature), and `content_warning`
(99.98% null).

### Pull rates, joined in per product type

`scripts/join_pull_rates.py` (run after `build_merged.py`) joins in
`mtg-print-distribution`'s rarity/pull-rate data -- see "Not yet joined in"
below for why this took an upstream fix first, now resolved.

A card can be pullable from more than one *type* of product at once (e.g.
both a Draft Booster and a Collector Booster of the same set), each with
its own rate and its own product price, and there's no way to collapse
those into one "true" scarcity number without knowing real per-product
unit sales, which nobody publishes. So instead of one scarcity column,
there are three columns **per product subtype actually observed**
(`booster_products.subtype` -- the 15-value grouping, not the 52-value raw
`pack_code`, most of which are one-off per-set variants like individual
guild Theme Boosters that don't generalize):

- `{subtype}_product_id` -- a reference back to the exact
  `mtg-print-distribution` `booster_products.id` used (traceable to that
  repo's own db for the full product name/details)
- `{subtype}_rate_percent` -- `avg_card_rate_percent`, the odds of pulling
  *this exact card* (this row's own uuid+finish) from that product
- `{subtype}_price_usd` -- that product's own sealed-box/pack market price
  (context, not the card's own price)

Subtypes present: `advanced, collector, default, draft, gift_bundle,
jumpstart, other, play, premium, secret_lair, set, six-card, starter_deck,
theme, topper`. Note `default` here is a real, legitimate subtype value
(mtgjson's fallback classification for older sets whose one-and-only
booster type doesn't fit a modern category, e.g. Revised/Alliances/Fourth
Edition) -- **not** the synthetic all-time-average `default` *set*, which
is explicitly excluded from this join (`source == "computed_average"`
rows are dropped before joining).

All `NaN` where a card isn't obtainable from that product type at all --
e.g. Collector-Booster-exclusive specialty foils (`stepandcompleat`,
`textured`, ...) correctly show no Draft/Play Booster match. Matched via
MTGJSON uuid membership (`booster_rates.card_uuids`) + exact `finish`
match, not `set`+`collector_number`. 104,321/149,078 rows matched at least
one product.

**Known collision**: a handful of sets have two distinct products under
the same subtype (e.g. LTR's "Jumpstart" and "Jumpstart Vol 2," both
`subtype: jumpstart`) -- 6,607 such collisions were resolved by keeping
the higher-rate match, per `join_pull_rates.py`.

**Price coverage is weaker than product-match coverage for a few
subtypes** -- `{subtype}_rate_percent` can be populated while
`{subtype}_price_usd` stays `NaN`, since product-matching and TCGplayer
price-resolution are independent steps. Checked per subtype: most are 0-2%
gap (price resolves almost whenever the match does), but `six-card`
(4,073 rows) and `advanced` (380 rows) are **100%** unpriced -- these are
old, discontinued mini-pack/demo products ("Six Card Booster Pack" from
2009-2016-era sets, Portal-era "Demo Game Booster Pack") that TCGplayer
appears to have no catalog listing for at all, not a resolver bug. `other`
(39% gap) and `default`/`theme`/`topper` (11-17% gap) are worth knowing
about but not investigated further.

### Card price, joined in

`scripts/join_price.py` (run after `join_pull_rates.py`) joins in the
card's own price -- `price_min`, `price_max`, `price_avg` -- from
`mtg-card-prices/raw/price_history.parquet`, matched on `uuid` + `finish`
(mapped down to price_history's coarser `normal`/`foil`/`etched`
vocabulary -- no specialty-foil-level price exists there, matching what
was already found querying TCGplayer's own pricepoints API directly: it
never returns a finer printingType than Normal/Foil either, regardless of
the physical treatment).

**Provider is hardcoded to `tcgplayer`, one of 4 paper providers in
`price_history.parquet`** -- checked all 4 before picking: `manapool`'s
series has real, sane values mixed with multi-million-dollar spikes for
completely ordinary cards, a genuine upstream data bug, not outliers to
clip (confirmed on Mind Stone: `cardkingdom` $2.29-$4.49, `cardmarket`
$4.47-$4.89, `tcgplayer` $5.16-$6.41, `manapool` $2.86-$3,537,205.22 --
same card, same ~90-day window). `cardkingdom` and `cardmarket` (EUR)
checked out clean too, but `tcgplayer` is what the rest of this project
already resolves prices through (`mtg-print-distribution`'s own
`tcgplayer.py`), so this stays consistent with that rather than mixing
providers.

`price_history.parquet` only spans **~3 months** (2026-06-22 to
2026-09-20 as pulled) -- daily granularity within that window (median 89
of ~90 possible days populated per (uuid, finish, provider) group, so
coverage is dense, not sparse), but this is *not* the multi-year history
that would be needed for the temporal-robustness backtest in SPECS.md §6
-- don't assume it is.

142,012/149,078 rows (95.3%) matched a price. The still-open item is
SPECS.md §3's actual target-definition decision (relative/log-price vs. a
contemporaneous baseline) -- `price_min`/`price_max`/`price_avg` are the
raw ingredient for that, not the final target themselves.

`price_early`/`price_mid`/`price_late` give a compact trend shape without
keeping the full ~90-day daily series in `mtg-card-market-dataset` -- the window split
into 3 equal *calendar* periods (same ~30 days for every card, not
per-card relative thirds, so they're comparable across cards) with the
mean price in each. A single linear slope was considered and rejected --
it collapses "spiked then came back down" into the same near-zero trend
as "flat the whole time"; three points keep that shape for barely more
cost. Slightly higher null rate than `price_avg` (~5-7% vs. ~4.7%) since a
card can have data in some periods but not others.

134 columns total -- see `COLUMNS.md` for the full breakdown by category.

## Known gotchas

- **A specialty-foil `finish` value does not always mean a distinct
  printing.** When a specialty foil is offered as an *alternative* pull
  alongside a plain foil of the same card, it's a genuinely separate
  printing -- its own `collector_number`, `scryfall_id`, `tcgplayer_id`, own
  observable price. Confirmed multiple ways: Dominaria United's Sol'Kanar
  the Tainted (`#318` nonfoil+foil vs. `#359` `finish: textured`);
  Phyrexian Vindicator (`one`), 4 separate printings (#27 plain, #300
  showcase, #347 Oil Slick Raised Foil, #425 Step-and-Compleat Foil);
  Scavenger Grounds (`eos`), 4 separate printings (#37/#82 nonfoil+foil,
  #127/#172 galaxyfoil-only) -- each its own `collector_number` and
  `tcgplayer_id`. Only when the specialty foil is the *sole* foil option for
  a printing does it live on that printing's own row instead -- e.g.
  Kenrith, the Returned King (`ltc` #515): one row for the foil finish,
  `finish: silverfoil`, no separate nonfoil-only or plain-foil sibling
  exists. Rule of thumb: a treatment gets a separate row/collector-number
  only when a plain-foil alternative for the same printing actually exists.
  This isn't purely type-based, though -- checked every specialty-foil
  value's share rate (shares a row with a nonfoil sibling vs. gets its own
  row): 14 of 19 types *never* share a row (`chocobotrackfoil, confettifoil,
  cosmicfoil, dazzlefoil, dragonscalefoil, facetfoil, halofoil, manafoil,
  oilslick_raisedfoil, raisedfoil, silverscroll, singularityfoil,
  stepandcompleat, textured`); of the other 5, `silverfoil` is the one
  exception that's unconditional -- **100%** (369/369) of silverfoil rows
  share a row with a nonfoil sibling, while `ripplefoil` (88%), `surgefoil`
  (27%), `rainbowfoil` (26%), and `galaxyfoil` (3%) are mixed depending on
  the specific set/product. Full list of the 1,439 shared-row cases:
  `shared_id_specialty_foils.csv` in this directory (not part of the build
  pipeline, a one-off reference -- note its `specialty_foil_type` column
  predates the `finish` merge described above, but the underlying cards are
  still correct).

## Not yet joined in

- EDHREC usage -- `edhrec_rank`/`edhrec_salt` are already present per-card
  above; richer inclusion-rate/synergy data, if pulled, still TBD (SPECS.md §8).

## Resolved gotchas (kept for history)

- **Rarity/pull-rate join was blocked, not just pending, until fixed
  upstream.** `set` + `collector_number` looks like a join key but isn't
  one: a card's set/collector_number don't determine which booster
  *product* it's actually obtainable from (that's decided by MTGJSON's
  per-product `sheets`, not anything on the card's own row), and
  `booster_rates` used to discard per-card membership when it aggregated
  sheets into buckets. Fixed by adding `booster_rates.card_uuids` in
  `mtg-print-distribution` (persists the uuids behind each bucket instead
  of just their count) -- see SPECS.md §2 for the full finding and
  "Pull rates, joined in per product type" above for the result. Also
  still true: that repo's own `booster_products.tcgplayer_product_id` is a
  *sealed pack/box* product id, a different namespace than this dataset's
  per-card `tcgplayer_id` -- never join on it.
