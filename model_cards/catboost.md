---
license: other
license_name: wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
library_name: catboost
pipeline_tag: tabular-regression
tags:
- magic-the-gathering
- mtg
- pricing
- catboost
- quantile-regression
- conformal-prediction
datasets:
- CXu0630/mtg-card-market-dataset
- pcwoods/mtg-card-prices
---

# MTG card price range: CatBoost quantile baseline

> Magic: The Gathering is a trademark of Wizards of the Coast, LLC. This is an
> unofficial fan project, not produced or endorsed by Wizards of the Coast.

**Source code, training pipeline and design doc:** [mtg_price_predict_catboost_deep_gp](https://github.com/CXu0630/mtg_price_predict_catboost_deep_gp)

Predicts an **80% price range** (USD) for a Magic: The Gathering card
printing (one card in one set/treatment/finish) from 93 tabular features:
rarity, finish, frame/border treatment, mana cost, type, P/T, legalities,
booster pull rates, EDHREC rank and artist. This is the project's
"Model 1", a fast CPU-only baseline. The deep-kernel GP is
[`CXu0630/mtg-price-deep-gp`](https://huggingface.co/CXu0630/mtg-price-deep-gp).

## Method

- CatBoost `MultiQuantile` regression (p10 / p50 / p90) on `log(price_avg)`.
  Settings: depth 6, `l2_leaf_reg` 3, learning rate 0.08, early-stopped at iteration 2,709.
- "Tier A" (fundamentals-only) feature set. In an ablation, adding ~350
  sparse community tag columns made coverage worse in every price bucket.
- The training split was rebalanced by capping each of 30 log-price bins at
  8,000 rows, so bulk commons don't dominate the loss. Calibration and test
  splits keep the natural price distribution.
- **Group-conditional split-conformal calibration** fits a separate
  additive correction to the [p10, p90] interval for each *predicted*-median
  bucket (`<$1`, `$1–10`, `$10–100`, `$100+`) on a held-out calibration split.
- `set` / `set_name` are deliberately not features, so the model
  generalizes to sets it hasn't seen.

## Results

Held-out test split (28,377 printings, 20%), split by `oracle_id` so that no
card appears in both train and test. Log-price MAE of the median;
coverage is the calibrated 80% interval.

| true price | n | MAE (log) | 80% coverage | mean width (log) |
|---|---|---|---|---|
| < $1 | 17,179 | 0.42 | 77.6% | 1.40 |
| $1–$10 | 7,641 | 0.61 | 82.9% | 2.08 |
| $10–$100 | 3,236 | 0.74 | 71.8% | 2.05 |
| $100+ | 321 | 1.14 | 52.3% | 2.03 |
| overall | 28,377 | 0.52 | 78.1% | 1.66 |

Compared with the deep GP, this model has lower median error, but its
intervals under-cover expensive cards ($10–$100 and $100+). Each model was
evaluated on its own split, so the comparison is approximate.

## Usage

```python
# pip install catboost huggingface_hub pandas pyarrow
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("CXu0630/mtg-price-catboost")
sys.path.insert(0, path)  # the repo bundles its inference code in mtg_price_prediction/
from mtg_price_prediction.catboost_model import CatBoostPredictor

model = CatBoostPredictor.from_pretrained(path)
ranges = model.predict(features_df)  # -> price_low / price_mid / price_high (USD)
```

Or with plain CatBoost:

```python
import json, numpy as np
from catboost import CatBoostRegressor
from huggingface_hub import snapshot_download

path = snapshot_download("CXu0630/mtg-price-catboost")
cfg = json.load(open(f"{path}/config.json"))
model = CatBoostRegressor(); model.load_model(f"{path}/model.cbm")

X = features_df[cfg["feature_cols"]].copy()
for c in cfg["cat_features"]:
    X[c] = X[c].fillna("__missing__").astype(str)
p10, p50, p90 = model.predict(X).T
buckets = np.digitize(p50, np.log([1, 10, 100]))   # 0:<$1 1:$1-10 2:$10-100 3:$100+
corr = np.array([cfg["calibration_corrections"][str(b)] for b in buckets])
low, mid, high = np.exp(p10 - corr), np.exp(p50), np.exp(p90 + corr)
```

`features_df` has rows shaped like the project's `prepare_training_data.py`
output. `config.json` → `feature_cols` lists the exact 93 columns.

## Files

| file | contents |
|---|---|
| `model.cbm` | CatBoost model (native format) |
| `config.json` | feature columns, categorical columns, hyperparameters, calibration corrections, test metrics |
| `mtg_price_prediction/` | inference code (`CatBoostPredictor`, calibration) |

## Limitations

- **Expensive cards are under-covered.** Calibrated coverage is 52% at $100+
  and 72% at $10–$100, against an 80% target. The calibration sample for $100+
  is small (131 points).
- **The model goes stale.** It is a direct regressor on absolute log-price,
  trained on a ~3-month TCGplayer window ending September 2026, so it drifts
  with the market.
- **Results are from a single seed and a single split.**
- **Out of scope:** multi-face cards, digital-only printings, and oversized cards.

## Training data

[`CXu0630/mtg-card-market-dataset`](https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset)
has one row per (printing, finish), 142,012 of them with a price. It is
built on [`pcwoods/mtg-card-prices`](https://huggingface.co/datasets/pcwoods/mtg-card-prices)
(Scryfall card data + TCGplayer price history) and joined with booster pull
rates from
[`CXu0630/mtg-print-distribution`](https://huggingface.co/datasets/CXu0630/mtg-print-distribution).
