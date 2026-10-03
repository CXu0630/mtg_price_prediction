# mtg-card-market-dataset/dataset.parquet — column categories

Working doc, edited as we go through the schema together. Reflects
`dataset.parquet` as of the last rebuild (134 columns, 149,078 rows, grain = one row per
`(printing, finish)`, digital-only/oversized/non-card-layout printings
excluded, see `README.md`). Update this file by hand when a column's
status/group changes; it's notes, not generated output.

**Status legend**

| mark | meaning |
|---|---|
| 🔑 | key/reference only, not a model feature |
| ✅ | feature, settled |
| 🕓 | held back for now (not dropped for good) |
| 🗑️ | drop candidate, not yet dropped |
| ❓ | undecided, needs a look |

---

## 1. Identity / keys

Not features — join/reference only.

| column | status | description | notes |
|---|---|---|---|
| `scryfall_id` | 🔑 | Scryfall's id for this printing | not unique per row (shared across a printing's finish rows) |
| `printing_key` | 🔑 | derived unique row id | `scryfall_id:finish` |
| `oracle_id` | 🔑 | Scryfall's id for the card's rules-text identity | shared across every printing of a card *concept* (e.g. all 953 Forests share one) |
| `uuid` | 🔑 | MTGJSON's id for this printing | the actual join key to `mtg-print-distribution` (`booster_rates.card_uuids`, matched with `finish`) — see group 10. `set`+`collector_number` looked like a join key but isn't one, see README "Resolved gotchas". Rows with no `uuid` at all are dropped (see README) -- confirmed those are all non-card accessory objects, no real card lacks one |
| `name` | 🔑 | the card's name | |
| `set_id` | 🔑 | Scryfall's internal id for the set | redundant with `set` (the code) — drop candidate, not yet dropped |
| `set` | 🔑 | the set code this printing belongs to | |
| `set_name` | 🔑 | the set's full display name | |
| `collector_number` | 🔑 | the number/code printed on the card within its set | **not** a join key to `mtg-print-distribution` despite appearances — see `uuid` above |
| `variation_of` | 🔑 | non-null only for alternate-art variant rows; the `scryfall_id` of the base printing this is a variant of | |
| `tcgplayer_id` | 🔑 | each row's own TCGplayer marketplace product id | nonfoil/foil share one product; etched has its own. Only marketplace id kept — `mtgo_foil_id`, `cardKingdomId`, `mcmId`, `tcgplayerProductId` all dropped |

## 2. Print history / edition metadata

| column | status | description | notes |
|---|---|---|---|
| `released_at` | ✅ | this printing's release date | |
| `num_paper_printings` | ✅ | count of paper printings this card concept has had | |
| `lang` | ✅ | which language this specific printing row is | |
| `set_type` | ✅ | the set's category (expansion, core, masters, commander, promo, ...) | |
| `booster` | ✅ | whether this printing was available to open from a booster pack | |
| `promo` | ✅ | whether this printing is a promotional edition | |
| `reprint` | ✅ | whether this printing is a reprint of an earlier release | |

## 3. Rarity / scarcity

Real pull-rate scarcity (per product type) is in group 10, not here.

| column | status | description | notes |
|---|---|---|---|
| `rarity` | ✅ | the printed rarity (common/uncommon/rare/mythic/special/bonus) | |
| `reserved` | ✅ | whether this card is on the Reserved List | guaranteed never reprinted, real price driver, don't lump into misc |

## 4. Finish / treatment

Structured to match `mtg-print-distribution.booster_rates`'s bucket key
(`finish`, `borderless` each their own field, not folded into a tag list).

| column | status | description | notes |
|---|---|---|---|
| `finish` | ✅ | which physical finish/treatment this row represents | `nonfoil` / `foil` / `etched`, or a specialty-foil name (`silverfoil`, `surgefoil`, ...) when one applies — one granular column, matching `booster_rates.finish`'s structure. Values verified against live TCGplayer product names (see `build_merged.py`) |
| `borderless` | ✅ | whether this printing is borderless | split out from `border_color`, matching `booster_rates.borderless` |

## 5. Mechanical / rules content

| column | status | description | notes |
|---|---|---|---|
| `layout` | ✅ | structural card-frame type (normal/saga/class/meld/...) | non-card layouts (token/emblem/scheme/vanguard/planar/art_series/double_faced_token/front_card) and multi-face layouts (adventure/flip/modal_dfc/prepare/reversible_card/split/transform) are both filtered out entirely, neither appears as a value here -- see README. `meld`'s front-half cards are kept (confirmed complete single-name rows, unlike the other multi-face layouts) |
| `mana_cost` | ✅ | the printed mana cost symbols, e.g. `{2}{W}{W}` | |
| `conv_mana_cost` | ✅ | converted mana cost / mana value, derived from `mana_cost` | renamed from source's `cmc` |
| `type_line` | ✅ | the card's full type line, e.g. "Legendary Creature — Human Wizard" | |
| `type_tags` | ✅ | every word of `type_line` (supertype, type, subtype alike) as its own tag | multi-hot-style list, same shape as `keywords`/`oracle_tags`/`art_tags`; handles multi-face lines and the em-dash separator |
| `oracle_text` | ✅ | the card's current official rules text | |
| `colors` | ✅ | colors present in the card's own mana cost | keep alongside `color_identity`, not redundant — lands/fixers diverge |
| `color_identity` | ✅ | every color the card touches — mana cost, rules text, color indicator | used for Commander-format deckbuilding legality |
| `color_indicator` | ✅ | an explicit printed color marker, used when a card's color isn't derivable from its mana cost | e.g. some back faces |
| `keywords` | ✅ | keyword terms printed on the card | broader than CR "keyword ability" — mixes keyword abilities, keyword actions, ability words; needs top-N + other bucketing before use, long tail of 890 values |
| `oracle_tags` | ✅ | crowdsourced functional tags (removal, card-advantage, ramp, ...) | likely replaces most of SPECS.md §2's planned regex/LLM extraction. Needs filtering: ~96.6% coverage, and mixes in non-mechanical trivia tags (alliteration, single-english-word-name) |
| `produced_mana` | ✅ | colors of mana this card can produce | lands/mana rocks/dorks |
| `power` | ✅ | printed power, for creatures | |
| `toughness` | ✅ | printed toughness, for creatures | |
| `loyalty` | ✅ | starting loyalty, for planeswalkers | |
| `defense` | ✅ | printed defense, for Battle cards | |
| `all_parts` | ❓ | references to other related card objects (tokens it creates, meld pairs, ...) | not yet used |
| `attraction_lights` | ❓ | which lights this Attraction card has | niche, Attractions-mechanic-only |

~~`card_faces`, `life_modifier`, `hand_modifier`~~ dropped -- confirmed
100% empty (multi-face cards excluded, Vanguard cards are oversized and
already excluded).

## 6. Format legality

| column | status | description | notes |
|---|---|---|---|
| `legalities.*` (23 cols) | ✅ | per-format legal/not_legal/restricted/banned status, one column per format | |
| `game_changer` | ✅ | whether this card is on WotC's curated Commander "Game Changers" list | functions more like a demand signal than a legality fact, cross-referenced in group 7 |
| `can_lead_commander` | ✅ | whether this card can serve as a Commander | |
| `can_lead_oathbreaker` | ✅ | whether this card can serve as an Oathbreaker | |
| `can_lead_brawl` | ✅ | whether this card can serve as a Brawl commander | |

## 7. Demand / usage signal

| column | status | description | notes |
|---|---|---|---|
| `edhrec_rank` | ✅ | this card's overall popularity rank on EDHREC | resolves most of SPECS.md §8's EDHREC open question — no scrape needed for rank |
| `edhrec_salt` | ✅ | EDHREC's "salt score" — how much players dislike playing against this card | |
| `penny_rank` | ❓ | Penny Dreadful format popularity ranking | relevance unclear |
| *(`game_changer`, see group 6)* | | | |

## 8. Physical / collectible treatment

| column | status | description | notes |
|---|---|---|---|
| `border_color` | ✅ | the card's border color | `borderless` is split out as its own column (group 4), but this still carries the other values (black/white/gold/silver/yellow) |
| `frame` | ✅ | which card frame era/style this printing uses (1993/1997/2003/2015/future) | |
| `frame_effects` | ✅ | special frame treatments applied (legendary crown, showcase, extended art, ...) | |
| `full_art` | ✅ | whether this printing has full-bleed artwork with no text-box border | |
| `textless` | ✅ | whether this printing omits rules text | |
| `story_spotlight` | ❓ | whether WotC tagged this card as tied to the set's marketing "story" content | a promo device, not a card attribute -- since Kaladesh (2016), certain flavor-relevant cards each set get a small banner icon linking to narrative articles on WotC's website. No gameplay/legality/treatment effect (same frame otherwise). Likely weak-to-no price signal; kept for now but a real candidate to drop |
| `watermark` | ✅ | the printed watermark, if any (e.g. a guild/faction symbol) | |
| `security_stamp` | ✅ | the type of foil security stamp printed on the card, used for authentication | oval/acorn/triangle/... |
| `promo_types` | ✅ | which promotional categories this printing belongs to | specialty-foil tags now folded into `finish` (group 4) are stripped out here, so they aren't double-represented |

## 9. Art / illustration

| column | status | description | notes |
|---|---|---|---|
| `artist` | ✅ | the credited artist name | |
| `artist_ids` | ✅ | Scryfall id(s) for the credited artist(s) | |
| `illustration_id` | ✅ | identifies the specific artwork used | groups rows that reuse the identical artwork across different printings — not the same as same-artist |
| `art_tags` | ✅ | crowdsourced tags describing the artwork's visual content | multi-hot, needs top-N + other bucketing like `keywords` |

## 10. Rarity / pull rate (per product type)

Joined in from `mtg-print-distribution` by `scripts/join_pull_rates.py` via
`uuid` + `finish` (see group 1) — 45 columns, three per product *subtype*
actually observed, not per raw `pack_code`. Not individually tabulated
below since they're all one repeating pattern; see README's "Pull rates,
joined in per product type" for the full explanation of why this is
per-subtype columns rather than one blended scarcity number.

| column pattern | status | description | notes |
|---|---|---|---|
| `{subtype}_product_id` | 🔑 | reference to the exact `mtg-print-distribution` `booster_products.id` used | traceable back to that repo's db for the full product name |
| `{subtype}_rate_percent` | ✅ | odds (%) of pulling this exact card from that product | `booster_rates.avg_card_rate_percent` |
| `{subtype}_price_usd` | ✅ | that product's own sealed-pack/box market price | context, not the card's own price |

`subtype` ∈ `advanced, collector, default, draft, gift_bundle, jumpstart,
other, play, premium, secret_lair, set, six-card, starter_deck, theme,
topper` (`default` here is mtgjson's real fallback subtype for older sets,
not the synthetic all-time-average set — that's excluded from the join).
`NaN` across all three columns for a subtype means this card isn't
obtainable from that product type at all.

## 11. Card price

Joined in from `mtg-card-prices/raw/price_history.parquet` by
`scripts/join_price.py` via `uuid` + `finish` (mapped to price_history's
coarser `normal`/`foil`/`etched` vocabulary). Provider hardcoded to
`tcgplayer` after checking all 4 paper providers — `manapool` has a
confirmed data bug (multi-million-dollar spikes on ordinary cards), see
README "Card price, joined in" for the full check.

| column | status | description | notes |
|---|---|---|---|
| `price_min` | ✅ | lowest tcgplayer retail USD price observed over the ~3-month window | |
| `price_max` | ✅ | highest tcgplayer retail USD price observed over the ~3-month window | |
| `price_avg` | ✅ | mean tcgplayer retail USD price over the ~3-month window | this is raw material for SPECS.md §3's target, not the final target itself -- the relative/log-price-vs-baseline decision is still open |
| `price_early` | ✅ | mean price in the window's first ~30 calendar days | same 3 calendar periods for every card, not per-card relative thirds -- comparable across cards |
| `price_mid` | ✅ | mean price in the window's middle ~30 calendar days | |
| `price_late` | ✅ | mean price in the window's last ~30 calendar days | |

A compact trend shape without keeping the full ~90-day daily series in
`mtg-card-market-dataset`. A single linear slope was considered and rejected -- it
collapses "spiked then came back down" into the same near-zero trend as
"flat the whole time"; three points keep that shape cheaply.

95.3% of rows (142,012/149,078) matched `price_avg`; `price_early`/
`price_mid`/`price_late` have a slightly higher null rate (~5-7%) since a
card can have data in some periods but not others. `price_history.parquet`
is only ~3 months deep, not the multi-year series that would be needed for
the temporal-robustness backtest in SPECS.md §6.

---

## Open threads

- Decide `set_id` drop (group 1).
- Decide `story_spotlight` drop (group 8) -- likely low/no price signal.
- Decide `{subtype}_product_id` dtype cleanup (group 10) -- currently
  `float64` (NaN forces float on pivot), could cast to nullable `Int64`.
- `six-card`/`advanced` subtype price columns (group 10) are 100% empty --
  old/discontinued products TCGplayer doesn't list, not fixable on our end.
- Pick the actual price target (group 11 / SPECS.md §3) -- `price_min`/
  `price_max`/`price_avg` are raw material, the relative/log-price-vs-
  baseline decision is still open.
- Not yet joined in: richer EDHREC data beyond rank/salt (SPECS.md §8).
