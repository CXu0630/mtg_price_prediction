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

**Source code, training pipeline and design doc:** [mtg_price_prediction](https://github.com/CXu0630/mtg_price_prediction)

Predicts an **80% price range** (USD) for a Magic: The Gathering card
printing (one card in one set/treatment/finish) from what's printed on the
card. It works for released cards and for hypothetical or unreleased ones.
It is the second opinion to the project's boosted-tree fair-value model
(Model A): it sees the same card attributes and supply inputs, and also reads
the card's rules text and art.

## Architecture

- **Wide & deep feature extractor** (PyTorch). Each input is projected
  separately and the results are concatenated into a 48-dim joint embedding:
  - Frozen oracle-text embedding: `sentence-transformers/all-mpnet-base-v2`, 768 → 64.
  - Frozen card-art embedding: CLIP ViT-B/32, `ViT-B-32-quickgelu` / `openai`
    weights in `open_clip`, run on Scryfall's `art_crop`; 512 → 64.
  - One `nn.Embedding` per categorical column (rarity, finish, frame, artist, ...).
  - An MLP over Model A's boolean/numeric inputs: mechanics, keywords, top-N
    oracle/art/type tags, legalities, EDHREC rank, booster pull rates and supply
    (set size, rarity-slot size, products carrying the printing), mana-cost detail,
    reprint and age history, and the set's price level (mean and top-5 mean log
    price of the set's *other* training cards).
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
| < $1 | 8,816 | 0.39 | 84.7% | 1.36 |
| $1–$10 | 3,939 | 0.73 | 73.7% | 2.17 |
| $10–$100 | 1,702 | 0.81 | 75.9% | 2.28 |
| $100+ | 182 | 0.99 | **71.4%** | 2.44 |
| overall | 14,639 | 0.54 | 80.5% | 1.70 |

Compared with the boosted-tree fair-value model (Model A: log MAE 0.43, 80.0%
coverage overall, 65.4% at $100+), this model has the wider midpoint error but
better coverage on expensive cards. Its gain comes from its calibrated
intervals, not sharper point estimates. Each model was evaluated on its own
held-out split, so the comparison is approximate. Exact numbers are in
`config.json` → `test_metrics`.

Permutation importance on the test split (rise in log MAE when a group is
shuffled): EDHREC rank +95%, pull rates +46%, card/printing age +33%, finish
+26%, set price level +23% (+45% on $10+ cards). The text and art embeddings
add +4% and +3%.

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

To build `features_df` from rows of
[`CXu0630/mtg-card-market-dataset`](https://huggingface.co/datasets/CXu0630/mtg-card-market-dataset):

```python
from mtg_price_prediction.gp_features import gp_feature_frame
features_df = gp_feature_frame(rows, model.prep["wide_cols"], model.feature_context(dataset))
```

`feature_context` supplies what the Model A inputs need: the full dataset (set
sizes, each card's other printings), `set_prices.parquet`, and the as-of date
for age features (`config.json` → `model_a_inputs`).

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
| `set_prices.parquet` | (set, card) → max log price of the training cards, for the set price level input |
| `mtg_price_prediction/` | inference code: model classes, preprocessing, feature building, calibration |

## Limitations

- **$100+ is under-covered.** Calibrated coverage there is 71.4%, not 80%, and the
  bucket is small (182 test points). Prices driven by combos, bans or
  collectibility are mostly invisible to these features.
- **Hypothetical cards lose the set price input.** A card from a set with no
  priced training cards gets `set_price_missing` and an imputed level.
- **The GP's uncertainty is nearly constant across cards.** Predictive std
  varies by only about 1% within a bucket, a known deep-kernel-learning
  collapse. Interval width adapts by *predicted price bucket*, not per card.
- **Most of the signal is tabular.** A tabular-only variant comes close to
  this model, an embeddings-only one collapses, and the embeddings add only a
  few percent in permutation importance.
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
