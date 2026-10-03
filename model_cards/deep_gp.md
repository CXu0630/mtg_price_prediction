---
license: other
license_name: wotc-fan-content
license_link: https://company.wizards.com/en/legal/fancontentpolicy
library_name: pytorch
pipeline_tag: tabular-regression
tags:
- magic-the-gathering
- mtg
- pricing
- gaussian-process
- deep-kernel-learning
- gpytorch
- conformal-prediction
datasets:
- CXu0630/mtg-card-market-dataset
- pcwoods/mtg-card-prices
---

# MTG card price range: deep kernel GP

> Magic: The Gathering is a trademark of Wizards of the Coast, LLC. This is an
> unofficial fan project, not produced or endorsed by Wizards of the Coast.

Predicts an **80% price range** (USD) for a Magic: The Gathering card
printing (one card in one set/treatment/finish) from what's printed on the
card. It works for released cards and for hypothetical or unreleased ones.
This is the project's "Model 2". The CatBoost baseline is
[`CXu0630/mtg-price-catboost`](https://huggingface.co/CXu0630/mtg-price-catboost).

## Architecture

- **Wide & deep feature extractor** (PyTorch). Each input is projected
  separately and the results are concatenated into a 48-dim joint embedding:
  - Frozen oracle-text embedding: `sentence-transformers/all-mpnet-base-v2`, 768 → 64.
  - Frozen card-art embedding: CLIP ViT-B/32, `ViT-B-32-quickgelu` / `openai`
    weights in `open_clip`, run on Scryfall's `art_crop`; 512 → 64.
  - One `nn.Embedding` per categorical column (rarity, finish, frame, artist, ...).
  - An MLP over ~465 boolean/numeric features: mechanics, keywords, top-N
    oracle/art/type tags, legalities, booster pull rates, EDHREC rank.
- **Sparse variational GP** (GPyTorch SVGP) with 750 inducing points and an
  RBF-ARD kernel over the joint embedding, Gaussian likelihood. Trained end to
  end on the ELBO against `log(price_avg)`.
- **Group-conditional split-conformal calibration**. The raw `mean ± 1.2816·std`
  interval gets a separate additive correction for each *predicted*-price
  bucket (`<$1`, `$1–10`, `$10–100`, `$100+`), fit on a held-out calibration split.
- Training used graduated oversampling of the $100–$1000 range to improve
  coverage on expensive cards.

## Results

Held-out test split (14,639 printings), split by `oracle_id` so that no card
appears in both train and test. Log-price MAE; coverage is the calibrated 80% interval.

| true price | n | MAE (log) | 80% coverage | mean width (log) |
|---|---|---|---|---|
| < $1 | 8,816 | 0.43 | 83.9% | 1.45 |
| $1–$10 | 3,939 | 0.74 | 75.4% | 2.30 |
| $10–$100 | 1,702 | 0.85 | **76.0%** | 2.46 |
| $100+ | 182 | 1.18 | **67.6%** | 2.73 |
| overall | 14,639 | 0.57 | 80.5% | 1.81 |

Compared with the CatBoost baseline after the same calibration, this model's
coverage is much better on the two expensive buckets: 76.0% vs 71.8% at
$10–$100, and 67.6% vs 52.3% at $100+. CatBoost has the lower *median* error,
though, so this model's gain comes from better-calibrated and wider
intervals, not sharper point estimates. Each model was evaluated on its own
split, so the comparison is approximate. Exact numbers are in
`config.json` → `test_metrics`.

## Usage

```python
# pip install torch gpytorch safetensors huggingface_hub pandas pyarrow
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("CXu0630/mtg-price-deep-gp")
sys.path.insert(0, path)  # the repo bundles its inference code in mtg_price_prediction/
from mtg_price_prediction.deep_gp import DeepGPPredictor

model = DeepGPPredictor.from_pretrained(path)
ranges = model.predict(features_df, text_emb, image_emb)  # -> price_low / price_mid / price_high (USD)
```

- `features_df` has rows shaped like the project's
  `prepare_training_data_model2.py` output: the columns listed in
  `preprocessor.json` (`categorical_cols` + `wide_cols`). Missing numeric
  values are imputed. Unseen categories map to `__unk__`.
- `text_emb` is `(n, 768)`, the card's oracle text (plus any token or meld-result
  text it creates) run through `all-mpnet-base-v2`.
- `image_emb` is `(n, 512)`, the CLIP ViT-B-32-quickgelu image embedding of the art crop.
- `pull_rate_defaults.parquet` gives the expected booster pull rate for each
  (rarity, finish). Use it to fill `*_rate_percent` for a card that isn't in a
  real product yet.

## Files

| file | contents |
|---|---|
| `model.safetensors` | feature extractor + GP + likelihood weights (`model.*`, `likelihood.*`) |
| `config.json` | architecture, training config, calibration corrections, test metrics |
| `preprocessor.json` | categorical vocabularies, wide-column order, impute medians, standardization stats |
| `pull_rate_defaults.parquet` | (rarity, finish) → expected pull-rate lookup for unreleased cards |
| `mtg_price_prediction/` | inference code: model classes, preprocessing, calibration |

## Limitations

- **$100+ is under-covered.** Calibrated coverage there is 67.6%, not 80%, and the
  bucket is small (182 test / 113 calibration points). Prices driven by
  combos, bans or collectibility are mostly invisible to these features. One
  example is Paradoxical Outcome: true price $104, predicted single digits.
- **The GP's uncertainty is nearly constant across cards.** Predictive std
  varies by only about 1% within a bucket, a known deep-kernel-learning
  collapse. Interval width adapts by *predicted price bucket*, not per card.
- **Ablations suggest most of the signal is tabular.** A tabular-only
  variant comes close to this model, and an embeddings-only one collapses.
- **Results are from a single seed and a single split.** Differences of a
  few points between variants are within noise.
- **The model goes stale.** It was trained on a ~3-month TCGplayer price window
  ending September 2026 and predicts absolute log-price, so it drifts as the
  market moves.
- **Out of scope:** multi-face cards (transform, MDFC, split, adventure, ...),
  digital-only printings, and oversized cards.

## Training data

[`CXu0630/mtg-card-market-dataset`](https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset)
has one row per (printing, finish), 142,012 of them with a price. It is
built on [`pcwoods/mtg-card-prices`](https://huggingface.co/datasets/pcwoods/mtg-card-prices)
(Scryfall card data + TCGplayer price history) and joined with booster pull
rates from
[`CXu0630/mtg-print-distribution`](https://huggingface.co/datasets/CXu0630/mtg-print-distribution).
Card art comes from Scryfall.
