---
license: other
license_name: wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
tags:
- magic-the-gathering
- mtg
- pricing
---

# MTG price app — runtime assets

> Magic: The Gathering is a trademark of Wizards of the Coast, LLC. This is an
> unofficial fan project, not produced or endorsed by Wizards of the Coast.

Artifacts the [MTG price-range app](https://github.com/CXu0630/mtg_price_prediction)
loads at runtime, refreshed by its nightly pipeline (`scripts/app/`). Not meant
as a standalone dataset — see
[`CXu0630/mtg-card-market-dataset`](https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset)
for the data and [`CXu0630/mtg-price-deep-gp`](https://huggingface.co/CXu0630/mtg-price-deep-gp) for the
published deep GP.

| file | what |
|---|---|
| `models/app/model_a.joblib` | Model A — fair value (HistGradientBoosting + conformal sigma model), retrained on the newest MTGJSON prices |
| `models/app/model_b.joblib` | Model B — 14-day TCGplayer move, 10/50/90% quantiles + CQR |
| `models/app/model_c.joblib` | Model C — 14-day Card Kingdom buylist move |
| `models/app/manifest.json` | price date, training cutoffs, quality metrics |
| `models/app/gp_static.parquet` | deep-GP fair value + 80% range and nearest comps for every printing (price-independent) |
| `data/app/live_rows.parquet` | the newest day's market features (peers for custom cards) |

The `.joblib` files are scikit-learn 1.9.1 pickles — load them only with the
pinned versions in the app repo's `requirements/pipeline.txt`, and only if you
trust this repo (unpickling runs code).

Prices come from [MTGJSON](https://mtgjson.com) (TCGplayer, Card Kingdom, Cardmarket).
