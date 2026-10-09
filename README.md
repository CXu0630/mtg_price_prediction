# MTG Price Intelligence

Prices Magic: The Gathering cards. For any printing, or a card that doesn't exist yet, the app gives a
fair value with an 80% range, a 14-day TCGplayer forecast with a BUY / SELL / HOLD signal, and a 14-day
Card Kingdom buylist forecast.

- **App:** https://cxu0630.github.io/mtg_price_prediction/ (static site, rebuilt nightly)
- **Deep GP model:** [`CXu0630/mtg-price-deep-gp`](https://huggingface.co/CXu0630/mtg-price-deep-gp)
- **Datasets:** [`CXu0630/mtg-card-market-dataset`](https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset),
  [`CXu0630/mtg-print-distribution`](https://huggingface.co/datasets/CXu0630/mtg-print-distribution)
- **App assets (nightly models, precomputed GP outputs):** [`CXu0630/mtg-price-app-assets`](https://huggingface.co/datasets/CXu0630/mtg-price-app-assets)

| Model | Question | Where it runs |
|---|---|---|
| A: fair value (boosted trees) | What should this printing cost, from what it is? | Retrained nightly; in the browser for custom cards |
| Deep kernel GP | Same question, plus rules text and art; finds similar cards | Trained once; outputs precomputed for every printing |
| B: TCGplayer forecast | Where will the TCGplayer price be in 14 days? | Retrained nightly; in the browser for custom cards |
| C: buylist forecast | What will Card Kingdom's buylist pay in 14 days? | Retrained nightly |

Models A, B and C follow the method of [pwoodsCMU/mtgpricepredictor](https://github.com/pwoodsCMU/mtgpricepredictor),
retrained here on our data.

## Layout

| Path | What |
|---|---|
| `.github/workflows/nightly.yml` | Nightly job: refresh prices, retrain A/B/C, score every printing, export A/B for the browser, parity test, build, deploy to GitHub Pages |
| `scripts/app/` | The nightly pipeline steps, plus `build_gp_static.py` (GP outputs) and `build_explore_stats.py` (Data page) |
| `src/mtg_price_prediction/` | Shared package: Model A (`fair_value`), Models B/C (`market`), deep GP (`deep_gp`, `gp_features`), custom cards (`cards`), scoring, calibration |
| `scripts/*.py` | Dataset build and deep GP training (below) |
| `web/` | React + TypeScript front end; `web/src/model/` is the browser port of Models A and B |
| `model_cards/` | Hugging Face cards for the deep GP and the app assets |
| `tests/` | Python tests (`pytest`); the front end's parity test is `npm test` in `web/` |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"          # the nightly pipeline alone needs only requirements/pipeline.txt
cd web && npm ci
```

## Run the app pipeline locally

```bash
python scripts/app/assets.py fetch --no-app       # dataset, deep GP, precomputed GP outputs from HF
python scripts/app/refresh_prices.py              # latest MTGJSON prices
python scripts/app/train_models.py                # Models A, B, C
python scripts/app/score_all.py                   # every printing -> web/public/data/
python scripts/app/export_web_models.py --fixture # Models A and B for the browser + parity fixture
cd web && npm test && npm run dev                 # parity test, then the site at localhost:5173
```

## Reproduce the dataset

```bash
python scripts/fetch_market_dataset.py            # or download the published dataset directly
```

To rebuild it from sources: `fetch_mtg_card_prices.py` → `build_merged.py` → `join_pull_rates.py` →
`join_price.py`. The pull-rate join reads the `mtg-print-distribution` export from
`./mtg-print-distribution/data/exports/hf_dataset/` (the files of the published dataset).

## Reproduce the deep GP

```bash
python scripts/fetch_card_art.py                  # art crops, ~3.6 GB -> card_art/
python scripts/embed_text.py                      # all-mpnet-base-v2 -> embeddings/
python scripts/embed_images.py                    # CLIP ViT-B/32 -> embeddings/
python scripts/build_pull_rate_defaults.py        # -> training/pull_rate_defaults.parquet
python scripts/prepare_training_data_model2.py    # -> training/features_model2.parquet
python scripts/add_model_a_features.py            # + Model A's inputs -> training/features_model2_plus_a.parquet
python scripts/train_model2.py                    # -> training/model2.pt (CPU)
python scripts/export_models.py                   # -> models/deep_gp/, with parity checks
python scripts/push_models_to_hub.py              # -> CXu0630/mtg-price-deep-gp
python scripts/app/build_gp_static.py             # GP price + similar cards for every printing
```

*Magic: The Gathering* is a trademark of Wizards of the Coast. This is an unofficial fan project, not
produced or endorsed by Wizards of the Coast.
