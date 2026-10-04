# App design sketch (v1, draft for discussion)

Status: **v1 built** (see §8 for what exists and how to run it). Decisions taken are
marked **[decided]**; the original options are kept for the record.

## 1. What it does

The user submits a **list of cards** — existing printings and/or custom
(hypothetical) cards — and gets back **one prediction row per card**: what it
should be worth (Model A), where its TCGplayer price is heading over the next 14
days (Model B), and where Card Kingdom's buylist is heading (Model C).

Models, all already trained:

| model | question | default | alternative (user toggle) |
|---|---|---|---|
| **A — fair value** | what should this printing cost, from what it is? | teammate's HGB Model A (`pcwoods/mtg-trading-models`) | our deep GP with text + image embeddings (`CXu0630/mtg-price-deep-gp`) |
| **B — TCGplayer 14-day move** | where will its TCGplayer price be in 14 days? | teammate's Model B | — |
| **C — CK buylist 14-day move** | what will CK's buylist pay in 14 days? | teammate's Model C | — |

## 2. Input

A list. Each item is one of:

```jsonc
// existing printing -- the key every model already uses
{"kind": "existing", "scryfall_id": "…", "finish": "foil", "quantity": 2}
// (the UI can also accept name + set [+ collector number] and resolve it via Scryfall,
//  like the teammate's check_cards; ambiguous matches come back as "ambiguous")

// custom card -- everything the models read that's printed on the card / declared at design time
{"kind": "custom", "quantity": 1,
 "name": "…", "mana_cost": "{2}{U}{U}", "type_line": "Creature — Wizard",
 "oracle_text": "…", "power": "2", "toughness": "3",
 "rarity": "mythic", "finish": "foil", "set_type": "expansion",
 "frame": "2015", "border_color": "borderless", "frame_effects": ["showcase"], "promo_types": [],
 "release_date": "2026-11-14",
 "demand": "staple",            // stands in for EDHREC rank -- see §5
 "reprint_of": null,            // optional scryfall_id/oracle_id if it's a new printing of a real card
 "set": null,                   // optional existing set code, for set-level context
 "art": "<uploaded image>"}     // optional; only the GP backend uses it
```

## 3. Output: one row per input card

Extends the teammate's `predict.py` schema (so their code path stays the
reference for existing cards):

| group | fields | existing | custom |
|---|---|---|---|
| identity | `printing_key`, name, set, finish, quantity, `status` (matched / ambiguous / not supported) | ✓ | ✓ (`custom:<n>`) |
| today's market | `price_today` (TCGplayer), `ask` (cheapest real ask), CK retail, CK buylist cash/credit, Cardmarket, `price_date` | ✓ | — |
| **A: fair value** | `fair_value`, `fair_lo`, `fair_hi` (80%), `gap` = log(price / fair), `model_a` (which backend) | ✓ | ✓ |
| **B: 14-day TCG forecast** | `pred_price`, `price_lo`, `price_hi` (80%), `signal_range` / `signal_median` (BUY/SELL/HOLD), `unconfirmed_jump` | ✓ | ✓ low-information, flagged |
| **C: 14-day CK buylist** | `buylist_pred`, `buylist_lo`, `buylist_hi`, `ck_offer_suspect` | ✓ | — (no buylist exists) |
| comps (GP backend only) | 5 nearest similar cards with their current prices — "priced like these" | ✓ | ✓ |
| provenance | `snapshot_date`, `model_cutoff`, model versions, caveat flags | ✓ | ✓ |

Plus list-level totals (Σ quantity × price_today, Σ quantity × fair_value,
Σ quantity × pred_price) and CSV export.

## 4. How a custom card flows through the chain

Rule (agreed): **a custom card has exactly one day of price data — the price
Model A predicts for it.**

1. **Features** from the declared attributes (same `static_features` as A's
   training). Unknowable ones: pull rates → the (rarity, finish) defaults table
   we already have; EDHREC rank → the `demand` input (§5); set price/chase level
   → the chosen `set`'s levels if given, else missing; reprint history → from
   `reprint_of` if given, else "first printing".
2. **Model A** (selected backend) → `fair_value` + 80% range. GP backend also
   embeds `oracle_text` (all-mpnet-base-v2) and the uploaded art (CLIP), and
   returns comps.
3. **Synthetic day-0 row for Model B**: `price_today = fair_value`; one day of
   history ⇒ `change_21d = 0`, `volatility_21d = 0`, `jump_7d = 0`, `gap = 0`;
   cross-market levels missing; siblings from `reprint_of`'s real printings if
   given (their *real* moves — the only live signal a custom card can have);
   set peers from `set` if given; `release_status` from `release_date`.
4. **Model B** → forecast. Expect it to say "≈ no move, wide range" for most
   custom cards; that's the honest answer with one day of data, and the UI says so.
5. **Model C**: not applicable (no buylist), returned as null.

## 5. Open design issues

1. **[decided: (b)] B was trained on the teammate's A.** B reads A's `gap` and
   `fair_log`. If the user switches A to the GP, B would be fed inputs from a
   model it never saw (different error profile) — silently wrong. Options:
   - (a) the toggle changes only the **displayed** fair value; B always uses the
     teammate's A internally. Simple, but then custom cards' B forecast doesn't
     match the A the user chose.
   - (b) **retrain B without `gap`/`fair_log`** — the teammate's own ablation shows
     no loss (Spearman 0.235 → 0.235), and our rebuild in `scripts/model_b/` can
     confirm it. Then A is purely a display/valuation choice and B is unaffected
     by the toggle. **Chosen.** Validated on our 3 walk-forward weeks: Spearman
     0.2419 -> 0.2407 (-0.5%), coverage unchanged, BUY/SELL hit rates ~2 points lower.
   - (c) retrain a second B on GP-derived OOF gaps. Most work, no expected gain.
2. **[decided] Demand for custom cards.** EDHREC rank is the #1 driver of every
   model (removing it costs ~35% accuracy). A custom card has none. Proposal: a
   5-step "expected demand" picker (staple / popular / playable / niche / unplayed)
   mapped to EDHREC-rank quantiles, plus showing how much the range widens.
3. **Unsupported printings.** Our dataset drops multi-face cards
   (transform/MDFC/split/adventure) and digital-only printings; return
   `status: "not supported"` rather than guessing.
4. **[decided] All three models are retrained in this repo** from the teammate's
   published method (A: `fair_value.py`, B/C: `market.py`) instead of loading
   their private scikit-learn 1.6.1 pickles. Model A reproduces their quality
   (OOF log-MAE 0.440 vs their 0.437, 80.0% coverage).
5. **Freshness.** Existing-card outputs are as fresh as the last nightly run
   (MTGJSON's newest day); shown on every row as `snapshot_date`.
6. **MTGGoldfish data is never used by the app** (private, academic use only).

## 6. Architecture

```
                    nightly GitHub Action (this repo)
   MTGJSON AllPrices ─┐   refresh prices → build daily rows (teammate's code)
   Scryfall bulk ─────┼─▶ score every tradeable printing with A (both backends), B, C
   HF model repos ────┘   → static JSON shards by set + manifest → deploy
                                         │
                                         ▼
 ┌───────────── GitHub Pages ─────────────┐        ┌──── on-demand API ────┐
 │ React (Vite + TS) app                  │        │ FastAPI on HF Space   │
 │ • card list editor (search existing,   │ custom │ (or Cloud Run)        │
 │   add custom)                          ├───────▶│ POST /predict         │
 │ • existing cards: read JSON shards     │ cards  │ • build features      │
 │   (no server)                          │◀───────┤ • A (HGB or GP+emb.)  │
 │ • results table + per-card detail      │        │ • synthetic day-0 → B │
 └────────────────────────────────────────┘        └───────────────────────┘
```

- **Existing cards never hit the API**: everything is precomputed nightly for
  both A backends, so the toggle is instant.
- **Custom cards** go to the API. It's CPU-only (GP + two embedding models fit
  in the free Space's 16 GB); cold start ~1 min after idle — acceptable for the
  custom path, and the UI shows a "warming up" state.
- One shared response schema (§3) for both paths, so the table doesn't care
  where a row came from.

## 7. Build order (proposal)

1. Pipeline: nightly scoring of existing printings → JSON shards (no UI yet).
2. Frontend: list input + results table over the shards (existing cards only).
3. API: custom cards with teammate's A + synthetic-day B.
4. GP backend toggle (+ comps), embeddings in the API.
5. Polish: totals, CSV import/export, per-card detail view with price chart.

## 7b. Change: custom cards are priced in the browser

Hosting the API turned out to need a paid tier (Hugging Face Docker Spaces require
PRO) or a cloud account, so **custom cards are now priced client-side** and the whole
app is static on GitHub Pages:

- `scripts/app/export_web_models.py` exports Models A and B to JSON nightly (trees,
  conformal factors, feature vocab, live sibling/set-move lookups, "reprint of" lookups).
- `web/src/model/` evaluates them: `hgb.ts` is a port of scikit-learn's tree prediction
  (including its internal categorical preprocessor), `custom.ts` ports the custom-card
  feature building (`cards.custom_rows`, `fair_value.static_features`,
  `market.custom_day0_rows`).
- `npm test` checks the port against Python on 48 varied fixture cards: Model A's
  feature vectors exactly, prices and forecasts to 1e-6. It runs in the nightly
  workflow before the site builds.
- The deep GP is not available for custom cards (it needs the text/art embedding
  models); existing printings keep their precomputed GP values. `api/` still works and
  can be deployed later (HF PRO or Cloud Run) to add the GP for custom cards.

## 8. What's built (v1)

| piece | where | notes |
|---|---|---|
| price refresh | `scripts/app/refresh_prices.py` | MTGJSON AllPrices (rolling ~90 days, all providers) -> `data/prices/` in ~1 min |
| Model A | `src/mtg_price_prediction/fair_value.py` | teammate's method; set-less custom cards get their set type's typical set context |
| Models B, C | `src/mtg_price_prediction/market.py` | no Model A inputs; Manapool excluded; CK buylist trusted only if <= CK retail <= 3x TCGplayer |
| custom cards | `src/mtg_price_prediction/cards.py` | demand tiers, (rarity, finish) pull rates in play + collector boosters, `reprint_of` |
| deep-GP backend | `gp_features.py`, `embed.py`, `scripts/app/build_gp_static.py` | features + text embeddings parity-checked exactly against training |
| training | `scripts/app/train_models.py` | A + B + C in ~8 min -> `models/app/` |
| nightly scoring | `scripts/app/score_all.py` | every printing -> `web/public/data/` (index + per-set JSON) |
| custom cards in the browser | `scripts/app/export_web_models.py`, `web/src/model/` | Models A + B as JSON, TypeScript port, parity-tested against Python (§7b) |
| API (optional, not deployed) | `api/app.py`, `api/Dockerfile` | FastAPI; adds the deep GP for custom cards when hosted |
| frontend | `web/` | React + Vite + TS; list editor, custom-card form, results table, CSV |
| automation | `.github/workflows/nightly.yml` | refresh -> retrain -> score -> export web models -> parity test -> build -> GitHub Pages |

Known limits: the card universe is the dataset build (newer printings come back as
"not in dataset" -- enter them as custom cards); multi-face cards are out of scope;
B's forecast for a custom card has no momentum or cross-market signal by construction.
