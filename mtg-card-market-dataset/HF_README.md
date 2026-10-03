---
license: other
license_name: wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
tags:
- magic-the-gathering
- mtg
- trading-cards
- pricing
pretty_name: MTG Card Market Dataset
size_categories:
- 100K<n<1M
---

# mtg-card-market-dataset

> Magic: The Gathering is a trademark of Wizards of the Coast, LLC. This is an
> unofficial fan-made dataset, not produced or endorsed by Wizards of the
> Coast. Released under WotC's [Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy).

One row per **(card printing, finish)** — 149,078 rows, 134 columns — joining
a card's oracle text/mechanics/art, its real market price (current + a
3-month trend), and its actual pull rate from every type of booster product
it's obtainable from. Built for MTG card price-range prediction.

## Data sources

| Source | Credit | What it supplies |
|---|---|---|
| [`pcwoods/mtg-card-prices`](https://huggingface.co/datasets/pcwoods/mtg-card-prices) | **pcwoods** (teammate) | The per-card foundation — oracle text, type/mana cost, keywords, crowdsourced functional/art tags, artist, legalities, and the daily card-price history this dataset's price columns are derived from |
| [Scryfall](https://scryfall.com) | via `pcwoods/mtg-card-prices` | Canonical per-printing card data and identifiers |
| [MTGJSON](https://mtgjson.com) (MIT, © Zach Halpern) | via `pcwoods/mtg-card-prices` and `mtg-print-distribution` | `uuid` identifiers, booster/sheet structure |
| [TCGplayer](https://www.tcgplayer.com) | via both sources above | The actual USD price data (product-level and per-card), from TCGplayer's public pricing endpoints |
| [`CXu0630/mtg-print-distribution`](https://huggingface.co/datasets/CXu0630/mtg-print-distribution) | this project | Real per-product-type pull rates (odds of pulling this exact card from a Draft/Set/Collector/Jumpstart/... Booster) and each product's own sealed price |

## Dataset fields

### Identity / keys

| field | description |
|---|---|
| `printing_key` | unique row id (`scryfall_id:finish`) |
| `uuid` | MTGJSON id for this printing — the join key to `mtg-print-distribution`'s pull-rate data and to `price_history` |
| `scryfall_id` | Scryfall id for this printing (shared across a printing's finish rows, so not unique alone) |
| `oracle_id` | Scryfall id for the card's rules-text identity, shared across every printing of a card concept |
| `name` | card name |
| `set` / `set_name` | set code / full set name |
| `collector_number` | number printed on the card within its set |
| `variation_of` | for alternate-art variant rows, the `scryfall_id` of the base printing |
| `tcgplayer_id` | this row's own TCGplayer marketplace product id |

### Print history / edition

| field | description |
|---|---|
| `released_at` | this printing's release date |
| `num_paper_printings` | count of paper printings this card concept has had |
| `lang` | language of this printing |
| `set_type` | set category (expansion, core, masters, commander, promo, ...) |
| `booster` | whether this printing was available to open from a booster pack |
| `promo` | whether this printing is a promotional edition |
| `reprint` | whether this printing is a reprint of an earlier release |

### Rarity / scarcity

| field | description |
|---|---|
| `rarity` | printed rarity (common/uncommon/rare/mythic/special/bonus) |
| `reserved` | whether this card is on the Reserved List (guaranteed never reprinted) |

### Finish / treatment

| field | description |
|---|---|
| `finish` | physical finish/treatment — `nonfoil` / `foil` / `etched`, or a specialty-foil name (`silverfoil`, `surgefoil`, `galaxyfoil`, ...) when one applies |
| `borderless` | whether this printing is borderless |

### Mechanical / rules content

| field | description |
|---|---|
| `layout` | structural card-frame type (normal/saga/class/meld/...); non-card and multi-face layouts are excluded entirely, see Known limitations |
| `mana_cost` | printed mana cost symbols, e.g. `{2}{W}{W}` |
| `conv_mana_cost` | converted mana cost / mana value |
| `type_line` | full type line, e.g. "Legendary Creature — Human Wizard" |
| `type_tags` | every word of `type_line` (supertype, type, subtype) as its own tag |
| `oracle_text` | current official rules text |
| `colors` | colors present in the card's own mana cost |
| `color_identity` | every color the card touches (mana cost, rules text, color indicator) — used for Commander deckbuilding legality, broader than `colors` |
| `color_indicator` | explicit printed color marker, used when color isn't derivable from mana cost |
| `keywords` | keyword terms printed on the card (keyword abilities, keyword actions, ability words) |
| `oracle_tags` | crowdsourced functional tags (removal, card-advantage, ramp, ...) |
| `produced_mana` | colors of mana this card can produce |
| `power` / `toughness` | for creatures |
| `loyalty` | starting loyalty, for planeswalkers |
| `defense` | printed defense, for Battle cards |
| `all_parts` | references to other related card objects (tokens it creates, meld pairs, ...) |
| `attraction_lights` | which lights this Attraction card has |

### Format legality

| field | description |
|---|---|
| `legalities.<format>` | legal/not_legal/restricted/banned status, one column per format (standard, modern, legacy, vintage, commander, pauper, ...) |
| `game_changer` | whether this card is on WotC's curated Commander "Game Changers" list |
| `can_lead_commander` / `can_lead_oathbreaker` / `can_lead_brawl` | whether this card can serve as a Commander / Oathbreaker / Brawl commander |

### Demand / usage signal

| field | description |
|---|---|
| `edhrec_rank` | overall popularity rank on EDHREC |
| `edhrec_salt` | EDHREC's "salt score" — how much players dislike playing against this card |
| `penny_rank` | Penny Dreadful format popularity ranking |

### Physical / collectible treatment

| field | description |
|---|---|
| `border_color` | border color (black/white/gold/silver/yellow) |
| `frame` | card frame era/style this printing uses |
| `frame_effects` | special frame treatments (legendary crown, showcase, extended art, ...) |
| `full_art` | whether this printing has full-bleed artwork with no text-box border |
| `textless` | whether this printing omits rules text |
| `story_spotlight` | whether WotC tagged this card as tied to the set's marketing "story" content |
| `watermark` | printed watermark, if any |
| `security_stamp` | type of foil security stamp printed on the card |
| `promo_types` | promotional categories this printing belongs to |

### Art / illustration

| field | description |
|---|---|
| `artist` / `artist_ids` | credited artist |
| `illustration_id` | identifies the specific artwork used — groups rows that reuse identical art across printings |
| `art_tags` | crowdsourced tags describing the artwork's visual content |

### Pull rate per product type

Three columns per product *subtype* actually observed (`draft`, `play`,
`set`, `collector`, `jumpstart`, `theme`, `topper`, `six-card`, `premium`,
`advanced`, `other`, `default`, `secret_lair`, `starter_deck`,
`gift_bundle`) — `NaN` across all three means this card isn't obtainable
from that product type at all.

| field pattern | description |
|---|---|
| `{subtype}_rate_percent` | odds (%) of pulling this exact card from that product |
| `{subtype}_price_usd` | that product's own sealed-pack/box market price (context, not the card's own price) |
| `{subtype}_product_id` | reference id back to the source product |

### Card price

| field | description |
|---|---|
| `price_min` / `price_max` / `price_avg` | lowest / highest / mean tcgplayer retail USD price over the ~3-month window |
| `price_early` / `price_mid` / `price_late` | mean price in each of 3 equal calendar periods within that window — a compact trend shape |

## Known limitations

- **Price provider is tcgplayer only.** `price_history` includes 3 other
  providers; `manapool` was excluded after finding genuine data corruption
  (multi-million-dollar price spikes on ordinary cards). `cardkingdom`/
  `cardmarket` checked out clean but weren't used, to keep one consistent
  provider throughout.
- **Price history spans ~3 months** (2026-06-22 to 2026-09-20 as pulled),
  daily granularity — not a multi-year series.
- **Pull rate can't be collapsed into one scarcity number.** A card
  obtainable from multiple product types has a separate rate/price per
  type; there's no public per-product unit-sales data to weight them into
  a single figure, so each subtype gets its own columns instead.
- **Multi-face cards excluded** (transform, modal DFC, split, adventure,
  flip, and similar) — their mechanical fields are effectively unusable at
  the top level in the source data (~100% null oracle text/mana cost).
- **Only cards actually obtainable from a real (non-precon, non-promo)
  booster product** are included; oversized, digital-only, and non-card
  objects (tokens, emblems) are excluded.
