# MTG Card Price Range Prediction — Model Specs

Status: both models built, tuned, calibrated, and published to Hugging Face
([`CXu0630/mtg-price-catboost`](https://huggingface.co/CXu0630/mtg-price-catboost),
[`CXu0630/mtg-price-deep-gp`](https://huggingface.co/CXu0630/mtg-price-deep-gp)).
For setup, reproduction steps and a results summary, start with README.md.
This document is the design rationale plus the chronological experiment log.
Some early sections ("pre-implementation" framing, §8 open questions) are
kept as written for the record, and later entries supersede them.

## 1. Goal

Given an MTG card **printing** (a specific card in a specific set/treatment —
released or unreleased), predict a price **range**: "cards like this are priced
from A to B." Not a single point estimate.

### Non-goals (v1)

- **Explaining/decomposing *why* a card is priced the way it is** (hedonic
  regression, residual analysis, "how much of this price is irrational hype vs.
  rarity vs. art") — deliberately shelved. It's a real, interesting direction
  (see "Shelved: analytical framing" below) but this spec is scoped to
  prediction only.
- **Trading/arbitrage tooling** (acting on mispricings, timing, transaction
  costs) — out of scope. The prediction model may be reused for that later, but
  building it isn't part of this spec.
- **Fine-tuning image/text encoders** — v1 uses frozen pretrained embeddings
  (see §4.1). Fine-tuning is a possible follow-up, not a v1 requirement.
- **Multi-face cards** (transform, modal DFC, split, adventure, flip,
  reversible, and Duskmourn's Omen/"prepare" layout) — dropped from
  `mtg-card-market-dataset` entirely, not just handled imprecisely. Their top-level
  `oracle_text`/`mana_cost`/etc. fields are ~100% null in the source data
  (that content only lives in an unflattened `card_faces` struct with no
  design yet), and they're only 2.4% of rows (and still only ~3.4% even
  restricted to 2019+ cards, so not a fast-growing share worth rushing a
  fix for). Revisit if `card_faces` flattening ever gets designed. `meld`
  is *not* included in this cut — its front-half cards are complete,
  single-name rows, confirmed empirically, a different mechanic from the
  others.

## 2. Data sources

Three datasets, joined on a card printing:

| Dataset | Owner | Grain | Key fields |
|---|---|---|---|
| Pull-rate / rarity | this repo (`mtg-print-distribution`) | one row per `(booster_product, rarity, finish, frame_effects, borderless, promo_tags)` bucket | `avg_card_rate_percent` (scarcity), `price_usd` (sealed *product* price, not card price — context only), set metadata — **joined into `mtg-card-market-dataset` per product subtype, see below** |
| Per-card | teammate (also independently covered by `mtg-card-market-dataset`, built this session from the public `pcwoods/mtg-card-prices` HF dataset — see below) | one row per (printing, finish) | oracle text, art tags, mechanical properties (mana cost, type, keywords, color identity), artist, `price_min`/`price_max`/`price_avg` — **all already joined and working in `mtg-card-market-dataset`**, sourced from `price_history.parquet` (tcgplayer, paper, retail, USD; `manapool` excluded, confirmed corrupted — see `mtg-card-market-dataset/README.md`) |
| EDHREC usage | to be pulled | one row per **card concept**, not per printing | inclusion rate / synergy / rank — a demand-side signal |

### Join-key asymmetry (important)

Target grain is **specific printing** (confirmed). Rarity and price are
naturally per-printing. EDHREC usage is per-card-concept and does **not** vary
by printing — it must be broadcast across all of a card's printings, not
treated as printing-specific. Don't let this get silently flattened during the
join; a card's EDHREC rank is the same feature value whether you're looking at
its cheap Set Booster printing or its expensive Collector Booster showcase
printing — the model needs to learn that the *other* features explain the
price gap between those two rows, not EDHREC.

### Rarity/pull-rate join — resolved (was blocked, not just approximate)

Originally scoped as "use the bucket average, refine to per-card later" —
that undersold the actual problem, found by reading `mtg-print-distribution`'s
own pipeline code directly (`classify.py`, `booster_math.py`):

- **A card's `set` + `collector_number` does not determine which booster
  *product* it's obtainable from, or whether it's obtainable at all.**
  Product membership comes entirely from MTGJSON's per-product `sheets` (an
  explicit `{card_uuid: weight}` map inside that product's `booster` config)
  — nothing on the card's own printing record. Two products from the same
  set (e.g. Play Booster vs. Collector Booster) routinely draw from
  disjoint card pools. A card can even sit on *multiple* sheets at once
  (documented case: the same uuid in both a `foil: false` "showcaseCommon"
  sheet and a `foil: true` "silverShowcaseCommon" sheet).
- **`booster_rates` as exported couldn't answer this either way** —
  `booster_math.py` aggregated every sheet's uuids into a bucket and stored
  only the bucket's count (`card_pool_size = len(bucket_card_uuids[...])`),
  discarding which specific uuids are in it. The per-card membership only
  existed transiently during that repo's own per-set computation, upstream
  of what actually got persisted to the `booster_rates` table.

**Fixed upstream**: added `booster_rates.card_uuids` (semicolon-joined,
sorted) in `mtg-print-distribution` — `booster_math.RateRow` now carries
the uuids alongside the count instead of discarding them, `db.py` persists
them with a migration for the existing db, and the full pipeline was
rebuilt (191 sets, cache-warm) to backfill it. `mtg-card-market-dataset` now joins on
`uuid` + `finish` (not `set`+`collector_number`) via
`scripts/join_pull_rates.py` — see its README section "Pull rates, joined
in per product type" for the result. One further finding along the way:
whether foil/nonfoil/etched share an MTGJSON uuid isn't a fixed rule either
— nonfoil/foil reliably do; etched is set-dependent (100% shared in some
sets, 0% in others) — which is exactly why the join matches on uuid *and*
finish together, not uuid alone.

### Mechanical properties buried in oracle text

Mana cost and type line are already structured (grab directly). Other
mechanical properties aren't separate fields but live inside the oracle text
— extract them rather than relying on the full-text embedding alone,
following the same "discrete/categorical over dense vector" preference as
art tags (§4.1):

1. **Official keywords are already structured — use them first.** Scryfall
   exposes a `keywords` array per card (Flying, Trample, Deathtouch, Menace,
   Vigilance, Haste, Lifelink, Hexproof, Ward, Flash, Indestructible,
   Convoke, Delve, ...) — no extraction needed, multi-hot flags same as art
   tags.
2. **Beyond official keywords, rule-based extraction is unusually tractable
   here**, because Oracle text is deliberately templated by the
   Comprehensive Rules (consistent phrasing, not free prose) — a modest
   regex/pattern library gets real coverage: `has_removal` ("destroy
   target"/"exile target"), `card_advantage` + count ("draw a/two/three
   cards"), `damage_amount` (numeric, from "deals N damage"), `is_etb_effect`
   ("when ~ enters the battlefield"), `tutor` ("search your library"),
   `lifegain` + amount, `counters_matter` ("+1/+1 counter"), etc.
3. **Alternative to hand-writing/maintaining that pattern library: a
   one-time LLM extraction pass.** Oracle text is short (1–3 sentences);
   feed each card through a cheap LLM once, with a fixed structured schema
   to fill in (removal? card advantage type? combat-relevant? ...). Cheap,
   parallelizable, one-time — and the output is still discrete/categorical,
   so it slots into Model 1 exactly like a hand-written flag would, without
   needing to anticipate every pattern by hand. Preferred over the regex
   route unless there's a reason to want the extraction logic auditable/
   deterministic.

Either way, these land as GBDT-native features for Model 1, same category as
official keywords and art tags. Model 2 still additionally embeds the full
oracle text for whatever nuance the discrete extraction misses — this isn't
a replacement for that, it's the Model-1-appropriate complement to it.

## 3. Target definition — the staleness problem

Price data is a moving target: absolute dollar price drifts with overall
market conditions (hype cycles, supply from new sets/reprints, general
inflation) for reasons unrelated to any single card's fundamentals. A model
trained to predict absolute price goes stale as soon as the market level
shifts, independent of how good the model is.

**Decision: don't predict absolute price directly.** Predict price *relative
to a contemporaneous baseline* (e.g., log-price minus a time-and-set-matched
reference, or percentile rank within a contemporaneous cohort). This separates
two signals moving at very different speeds:

- **Structural/relative value** (slow-moving): how a card compares to others
  given rarity, mechanical power, and art. This is what the model learns.
- **Market level** (fast-moving): overall market conditions at a point in
  time. This is *not* learned into model weights — it's supplied at inference
  time from current data.

This is also why the model is a **comps/similarity model, not a direct
regressor** (§4): a prediction is realized as a weighted combination of
*other cards' current prices*, so the dollar figure returned is always
current — never a value memorized from a stale training snapshot. Only the
*similarity structure* (which cards are analogous to which) is learned, and it
needs far less frequent retraining than a direct regressor would.

**Correction, made concrete when building Model 1 (2026-09-25+): the
staleness protection comes from *always re-fetching comps' current prices
at inference*, not from any particular transform of the training target.**
An earlier draft of this section suggested subtracting a cohort baseline
(e.g. same-set-and-rarity mean) from the target before training — that
turned out to work against the comps mechanism, not with it: reconstructing
a dollar prediction as `Σ w_i · y_i` only stays a clean weighted sum of
prices if every `y_i` was put on the same footing to begin with (plain
`log(price)`), not baselined against each card's own, different, reference
group. **Settled: `y = log(price_avg)`, no cohort/time adjustment**,
for both models. `price_min`/`price_max`/`price_avg`/`price_early`/
`price_mid`/`price_late` are the only source of this target and are
excluded from the input features entirely (would otherwise leak the target
straight into the embedding/tree splits). `{subtype}_price_usd` (sealed
*product* price) isn't direct leakage but is still a raw dollar value with
the same staleness problem one hop removed, and its information content
is already redundant with `subtype`/`rarity`/`frame_effects`/`borderless` --
excluded from features too, kept only as dataset context.

Also note: **this protection is specific to Model 2's comps-reweighting
inference step. Model 1 (§4.1) is a direct quantile regressor with no such
step, so it *will* go stale as the market drifts, regardless of target
transform** -- acceptable given its role (fast baseline / sanity check),
but worth being explicit that only Model 2 actually solves the problem
this section describes.

## 4. Models

Two models, not competing — Model 1 is the fast baseline and a sanity check;
Model 2 is the target architecture.

### 4.1 Model 1 — GBDT baseline + leaf-similarity

- LightGBM or CatBoost (CatBoost preferred: native high-cardinality
  categorical handling, relevant for `artist`), trained with a **quantile
  objective** on the relative/log-price target (§3).
- Features: scarcity — **per product-subtype columns**
  (`{subtype}_rate_percent` for each of draft/play/set/collector/jumpstart/
  theme/topper/..., `NaN` where a card isn't pullable from that subtype at
  all), not one blended number — no real per-product unit-sales data
  exists to weight them into a single scarcity figure, so the GBDT is left
  to learn any implicit weighting against price directly (§2, "Rarity/
  pull-rate join"; `{subtype}_price_usd` itself is excluded from features,
  see §3 correction), mechanical properties,
  EDHREC usage (broadcast per printing), categorical set/treatment fields,
  and **art tags** (multi-hot binary flags — one column per tag, or top-N
  tags + an "other" bucket for long-tail ones; already present in the
  teammate's per-card dataset, no new collection needed).
  **Deliberately no dense image/text embeddings** — GBDTs don't extract much
  from a raw dense embedding (tree splits work one feature at a time; pooling
  an embedding down to something split-friendly throws away most of what
  makes it useful). Art tags are exempt from that objection: they're
  categorical, not continuous, so a GBDT uses them natively and you can read
  tag importance straight off the model (unlike an embedding dimension).
  Keeping *embeddings* out (tags are fine) still makes Model 1 a clean
  ablation baseline, and this actually improves the ablation ladder: it adds
  a middle rung between "fundamentals alone" and "fundamentals + full
  learned embedding + GP" (Model 2) — fundamentals + art tags — letting you
  separate "does art matter at all" from "does the richer embedding
  representation of art matter beyond coarse tags." It also means Model 1 is
  buildable now, without waiting on an embedding pipeline. Same logic
  applies to mechanical properties extracted from oracle text (official
  keywords, removal/card-draw/ETB flags, etc.) — see "Mechanical properties
  buried in oracle text" in §2.
- Similarity, as a side effect: two printings that co-occur in the same
  leaves across many trees are "similar" per the model's own splits. Used as
  a quick sanity check on whether the *structured + tag* feature set produces
  sensible comps, before investing in Model 2.
- Compute: CPU only. Trains in seconds–minutes on this dataset's scale.
  Not a GPU workload — don't spend GPU budget here.

**First iteration built 2026-09-25+** (`scripts/prepare_training_data.py`,
`scripts/train_model1.py`, training data local-only, never published --
see `training/README.md`):

- Target: `log(price_avg)`, no cohort adjustment (§3 correction). Rows
  without a price match are dropped (142,012/149,078 remain).
- Quantile levels: **p10/p50/p90** (80% interval + median) — a placeholder
  default, not the considered final choice §8 still calls for.
- Split: `GroupShuffleSplit` on `oracle_id`, 80/20 (per §6 — no printing of
  the same card concept in both train and test).
- **`set`/`set_name`/`set_id` excluded from features**, on top of already
  being non-features per COLUMNS.md — the stated goal includes pricing
  *unreleased* cards, and a raw set-code categorical has zero learnable
  signal for a set the model has never seen. `released_at` + `set_type` +
  `promo_types` (already includes `universesbeyond` for crossover-hype
  sets like LTR) capture the generalizable part of "which set" without
  needing the exact code.
- `{subtype}_price_usd` and all `price_*` columns excluded from features
  (§3 correction, above).
- Multi-hot tag columns (`keywords`, `oracle_tags`, `art_tags`,
  `type_tags`, `promo_types`) trimmed to their top-N most frequent values
  (chosen per column from actual frequency concentration, not guessed) +
  an `_other` catch-all flag, rather than one column per distinct value
  (`oracle_tags`/`art_tags` have 4,332/11,220 distinct values, almost all
  long-tail). `frame_effects` (only 17 distinct total) kept in full.
- Ablation ladder run: A) fundamentals only, B) + `art_tags`, C) full
  (+ `keywords`/`oracle_tags`/`type_tags`/`promo_types`/`frame_effects`).
- **Training-set rebalancing**: raw price is dominated by bulk cards (two
  histogram bins alone, $0.19-$0.31 and $0.31-$0.50, hold 35% of all
  142,012 rows), which both skews the loss toward nailing bulk commons
  (the least interesting part of "price range prediction") and starves
  the sparser, harder, higher-value tail of training signal. Fixed with a
  per-bin row cap on `log(price_avg)` (`RESAMPLE_CAP_PER_BIN = 8000` over
  30 bins, chosen from the actual per-bin counts, not guessed) —
  **applied to the training split only**, never the test split, so
  evaluation still reflects the real population. Cut training set
  113,635 → 86,039 rows.
- **First-pass results (2026-09-25, before calibration fix)**: pinball
  p50 ≈ 0.25-0.26 (log-price scale) across all three ablation tiers, but
  **80%-interval coverage sits at 68-73%, not 80%** — `CatBoost`'s
  `MultiQuantile` loss optimizes each quantile head independently, with
  nothing enforcing that the resulting interval hits nominal coverage.
  Coverage got *worse*, not better, both with more training iterations
  and with more features — a real overfitting-on-the-tails signal, not
  noise. **Conclusion: don't trust these ranges as calibrated yet** —
  post-hoc conformal calibration (hold out a calibration slice, measure
  the real coverage gap, widen the interval to close it) is the next
  concrete step, not yet done.
- **Coverage is badly uneven across the price range, not just off by a
  constant amount** — checked bucketed by true price on the same
  untouched test set: 72-76% coverage for `<$1`/`$1-$10`, but only 59-61%
  at `$10-$100` and 40-43% at `$100+`. `scripts/calibrate_model1.py`
  implements **group-conditional (Mondrian) conformal calibration** — a
  separate interval-widening correction per price bucket (bucketed by the
  model's own *predicted* median, since true price isn't known at
  inference time), rather than one global correction that would be
  dominated by the huge cheap-card population and barely touch the high
  end. Split three ways, all by `oracle_id` group: fit (rebalanced,
  trained on) / calibration (natural distribution, used only to compute
  the per-bucket correction) / test (natural distribution, untouched,
  final evaluation only).
- **Calibration result**: overall coverage 71.4% → 78.9% (right next to
  the 80% target). `<$1` and `$1-$10` (87% of the test set) land close to
  nominal (78.7%, 83.4%). `$10-$100` and `$100+` improve (59.6%→71.8%,
  40.2%→49.5%) but stay meaningfully under-covered — traced to a small
  calibration sample in those buckets (1,468 / 123 points) plus
  bucket-assignment mismatch (a card whose *predicted* median lands in
  the wrong bucket doesn't get that bucket's correction, so part of the
  remaining gap is really "the underlying point prediction is wrong," not
  a calibration failure). Intervals widen substantially as an honest cost
  of the fix (e.g. `$1-$10`'s width roughly doubles) — this is what
  trustworthy uncertainty looks like at this data scale, not a regression.
- **Checked whether the full tag feature set (tier C) helps the
  under-covered buckets specifically** (`scripts/experiments/check_ablation_by_bucket.py`)
  , since the aggregate ablation numbers were nearly flat and that's
  exactly the kind of thing that can hide a real per-bucket effect (as
  the rebalancing check demonstrated). It doesn't -- **tier C makes
  coverage *worse* in every single bucket** (e.g. `<$1`: 70.9%→65.9%,
  `$10-$100`: 59.1%→54.7%, `$100+`: 42.1%→38.6%) for only marginal/mixed
  pinball-loss movement. The extra ~350 sparse tag columns (top-N-by-raw-
  frequency, see §2/§4.1) are adding overfitting surface, not signal.
  **Tier A (fundamentals only) is the better base model for now**, not
  just the simpler one -- revisit tag inclusion only with better-curated
  tags (not pure frequency cutoff) or once embeddings replace them.
- **Hyperparameter tuning** (`scripts/experiments/tune_model1.py`) —
  depth/`l2_leaf_reg`/`learning_rate` were never actually tuned before
  this, just one guessed config; motivated directly by the overfitting
  signal above. Small randomized search (8 of 18 grid candidates),
  selected by pinball loss on a calibration split, upweighted toward the
  `$10-100`/`$100+` buckets specifically. **Result: depth=6,
  l2_leaf_reg=3, learning_rate=0.08** (close to CatBoost's defaults; early
  stopping against the calibration split does the real regularizing).
- **Final / published Model 1** (`scripts/calibrate_model1.py`, which now
  also saves `training/model1.cbm`; published as
  `CXu0630/mtg-price-catboost`): tier A + the tuned config above, early-
  stopped at iteration 2,709. Calibrated test coverage: `<$1` 77.6%,
  `$1-$10` 82.9%, `$10-$100` 71.8%, `$100+` 52.3%, overall 78.1%;
  overall median MAE 0.516 (log-price).

### 4.2 Model 2 — deep kernel Gaussian Process

The actual target architecture: a learned embedding + a Gaussian Process
whose prediction *is*, by construction, a kernel-weighted combination of
training targets — formalizing the "comps" idea in §3 rather than
approximating it with an after-the-fact nearest-neighbor lookup.

**Pipeline:**

1. **Frozen embeddings** (no training, no fine-tuning — considered and
   deliberately deferred, see §4.3): run each card's art image through a
   pretrained CLIP image encoder, and its oracle text through a
   **conventional general-purpose sentence-embedding model** (e.g. a
   sentence-transformers MPNet/BGE-family model) rather than CLIP's text
   tower — CLIP's text encoder is trained to align with natural image
   captions, not templated rules text, so a general sentence encoder is
   plausibly a better frozen starting point for oracle text. Cache both
   embeddings once — do **not** recompute them inside the training loop
   (this is the single biggest lever on training time; see §5).
2. **Learned projection head**: a small MLP mapping
   `[CLIP image emb, oracle-text emb, art tags, extracted mechanical flags
   (§2), mana/type/other encoded tabular features]` → a joint embedding
   space (32–128 dim). Small (low millions of params), trained jointly with
   the GP below, not pretrained separately.

   **Discrete features (art tags, extracted mechanical flags, keywords) are
   concatenated in alongside the embeddings, not replaced by them** — same
   Model-1 features, reused here as an explicit "wide" pathway next to the
   "deep" embedding pathway (Cheng et al., *"Wide & Deep Learning for
   Recommender Systems,"* 2016). A pure embedding *can* in principle recover
   the same signal (the flags are deterministic functions of the same text
   the embedding sees), but it dilutes a sharp, price-relevant signal across
   many less-relevant embedding dimensions, asking the GP to rediscover from
   a modest dataset what the flags hand it directly — and it collapses the
   interpretability the discrete features exist to provide: with them kept
   as explicit input dimensions, effects can still be attributed to "this
   card has removal," not just "the embedding moved." The redundancy between
   flags and embedding is intentional, not wasteful.
3. **Sparse/variational GP** (SVGP, e.g. via GPyTorch) with ~500–1000
   inducing points, kernel defined over the joint embedding from step 2.
   Trained end-to-end (embedding + kernel + inducing points) by maximizing
   the variational ELBO against the relative/log-price target.

   **Do not use an exact GP.** Exact GPs are O(N³) compute / O(N²) memory in
   training-set size — infeasible at this dataset's scale (tens of thousands
   of printings) regardless of hardware. SVGP/inducing-point methods scale
   roughly O(N·M²) and are the standard fix.

**Inference** (for a released *or* unreleased card):

1. Embed the query card (steps 1–2 above; works identically whether or not
   the card has ever had a market price).
2. Get the GP posterior — mean and variance — over the relative-price
   target.
3. Calibrate back to a dollar range using **current** prices of the query's
   nearest training points (or the posterior variance directly, converted
   through the same baseline used in §3). This step reads live data at
   prediction time; it is the reason the model doesn't go stale.

**Inference input contract** — the goal is to be able to price a card from
just what's printed on it (including a hypothetical/unreleased one), not
require it to already exist in the real-world print/market databases this
project otherwise relies on. Auditing the current feature set
(`prepare_training_data.py`) against that bar:

- **On-card / declarable at design time (always required):** mana cost,
  colors/color identity, type line, oracle text (+ token/meld-result text,
  §4.2 step 1), power/toughness/loyalty/defense, keywords, rarity, layout,
  frame/border/promo treatment, finish, artist, card art image, a release
  (or "value as of") date.
- **Booster pull rates (`{subtype}_rate_percent`):** structurally
  unavailable for a card not yet in a real product — but rarity and finish
  (both on-card) are its two strongest predictors, so
  `training/pull_rate_defaults.parquet`
  (`scripts/build_pull_rate_defaults.py`) provides an expected rate per
  (rarity, finish), two-level-fallback (rarity+finish mean ->
  rarity-only mean -> global mean, each gated on >=30 real observations).
  Real per-product rates are used when the card is a known printing;
  the default lookup is used otherwise — same feature, same model, no
  separate code path.
- **Harder externalities (EDHREC rank/salt, oracle/art tags,
  `num_paper_printings`):** no equivalently principled default exists (rank
  in particular is a relative ordering over the *entire* card pool, not a
  property with a sensible "typical value" by rarity/finish) — kept as
  optional features with a missingness flag, not a fabricated value.
  Real cards get the real signal; a hypothetical card gets the model's
  learned "unknown" behavior for that flag.
- **Legalities:** auto-computed from (card type/color/text + a maintained
  banlist snapshot) rather than requested as user input, since it's not
  something printed on the card either way.

**First iteration — built and run** (`scripts/prepare_training_data_model2.py`,
`scripts/train_model2.py`, `scripts/calibrate_model2.py`,
`scripts/experiments/check_model2_by_bucket.py`; all local-only, see `training/README.md`):

- Wide+deep feature extractor: text (768→64) / image (512→64) branches,
  a `nn.Embedding` per categorical column (`emb_dim = min(32, sqrt(vocab))`),
  a wide branch over the ~465 boolean/numeric tabular features (same top-N
  tag/legalities/fixed-vocab treatment as Model 1, plus the pull-rate
  default lookup and `_isna` flags from the inference input contract
  above) → concatenated → 48-dim joint embedding.
- SVGP: 750 inducing points (random-subsample init through the untrained
  extractor), RBF+ARD kernel, `GaussianLikelihood`, trained end-to-end via
  `VariationalELBO`. Split: 80/10/10 by `oracle_id` group (train/val/test);
  val doubles as the calibration set (early-stopping only, no gradient
  updates on it, so it's a legitimate held-out set — same reasoning as
  Model 1's calib split). CPU, not MPS — GPyTorch's variational strategy
  uses ops (Cholesky) not yet reliable on MPS. ~2.6s/epoch, early-stopped
  at epoch 9/25.
- **Bucketed results (same group-conditional conformal calibration as
  Model 1, §4.1) beat Model 1's ceiling at the buckets that motivated
  building this model in the first place:**

  | bucket | Model 1 (calibrated) | Model 2 (calibrated) |
  |---|---|---|
  | overall | 78.9% | 81.1% |
  | \$10–\$100 | 71.8% | 78.4% |
  | \$100+ | ~50% | 57.7% |

  \$100+ (n=182 in this test split) is still short of the 80% target and
  the calibration correction there is drawn from only ~100 calibration
  points, so treat it as noisy — but it's a real improvement over Model 1's
  hard ceiling on the same bucket, not just a different number from a
  different random split. The raw (uncalibrated) aggregate number (79.3%)
  is misleading on its own — same lesson as Model 1's rebalancing check:
  it's inflated by the huge, easy `<$1` bucket (coverage 88.5% raw); always
  read bucketed, never aggregate alone.
**Tuning & ablations** (`scripts/experiments/tune_model2.py`,
`scripts/experiments/check_model2_ablation_by_bucket.py`; all checkpoints in
`training/tune_candidates/` and `training/ablation_checkpoints/`, local
only) — all numbers below are after the same per-bucket calibration as the
baseline, since raw/uncalibrated rankings are not reliable (calibration
changes bucket ranking substantially, same lesson as Model 1):

| variant | $10–$100 | $100+ | overall |
|---|---|---|---|
| baseline (full, joint_dim=48) | 76.1% | 58.8% | 80.9% |
| **tuned (full, joint_dim=96)** | **79.3%** | 57.7% | 81.0% |
| ablation: no_image (text+tabular) | 74.8% | 54.4% | 81.0% |
| ablation: no_text (image+tabular) | 72.1% | 42.3% | 80.0% |
| **ablation: tabular_only (no embeddings)** | 78.2% | **64.8%** | 80.5% |
| tabular_only + joint_dim=96 | 73.6% | 47.8% | 80.5% |

Honest, somewhat uncomfortable finding: **the embeddings' contribution is
mixed, not a clean win.** `tabular_only` — the exact same wide tabular
features Model 1 had, no text/image embedding at all — matches or beats
`full` on both hard buckets. `no_text` (image without text) is clearly the
worst variant on `$100+` (42.3%), showing the image embedding alone is not
a good substitute for text — but adding image back on top of text+tabular
(`full` vs `no_image`) doesn't clearly help either. And architecture/
hyperparameter choices interact with which inputs are present:
`joint_dim=96` helps `full` but hurts `tabular_only` — there's no single
"best settings" independent of which pathways are enabled.

Two honest takeaways, not yet resolved:
1. Model 2 as an *architecture* (deep-kernel GP + this feature set) beats
   Model 1's ceiling regardless of which ablation variant — even
   `tabular_only`, with identical information content to Model 1, clears
   Model 1's $10-100/$100+ numbers. So the win so far looks like it's
   coming from the model class change (GP + its calibration), not
   specifically from the embeddings.
2. Given (1), the embeddings may need more capacity, more epochs, a larger
   inducing-point budget, or fine-tuning (§4.3, currently deferred) before
   they earn their complexity — or the CLIP/text-embedding signal really
   is largely redundant with the curated tag/mechanical-flag features
   already in the wide branch for *this* target. Not distinguished yet.
3. All of these numbers are one random split (seed 0) and small buckets
   ($100+ is n=182 test / 38-105 calibration) — differences of a few
   points between variants are within plausible noise, not yet validated
   across multiple seeds.

Not yet done: multi-seed validation of the above, live-comps-lookup
staleness protection described in the inference steps (currently evaluated
purely on held-out log-price, not recalibrated against current prices at
inference time), and deciding whether to keep the image pathway at all
given ablation (2).

**Tail-robustness experiments** (`train_model2.py`'s `oversample_high_price`
and `likelihood_type` config fields; checkpoints in
`training/robustness_checkpoints/`) — prompted by checking whether the
`$100+` ceiling was an artifact of a few extreme outliers (it isn't: a
finer breakdown showed coverage already bad at `$100-200` and degrading
monotonically with price, not a tail-driven distortion — so capping/
removing high-price rows was rejected as a fix, since it would just hide
the real degradation and remove the model's only reason to handle
expensive cards at all). Two mechanisms tried instead, both against the
`$10-100`/`$100+`/overall calibrated baseline (76.1% / 58.8% / 80.9%):

| variant | $10–$100 | $100+ | overall |
|---|---|---|---|
| **oversample high-price bins (Gaussian)** | 75.7% | **61.5%** | 80.6% |
| Student-t likelihood | 77.4% | 44.0% | 80.6% |
| Student-t + oversample | 69.6% | 47.3% | 81.1% |

Oversampling (duplicating, with replacement, any log-price histogram bin
above $100 whose train-split count is below 500 — same imbalance-fix
idea as Model 1's rebalancing, applied to the opposite/sparser tail
instead) gives a real, if modest, win concentrated exactly where intended:
a finer breakdown shows `$1000+` coverage going from 0% to 25% and
`$500-1000` from 32% to 42% — at a real cost to `$100-200` (61%→48%), so
it's a genuine reallocation of fit quality toward the rarer/harder end,
not a free improvement everywhere.

**Student-t likelihood makes it worse, and the mechanism is
understandable in hindsight, not just a null result:** a heavy-tailed
noise model achieves robustness precisely by *discounting* the influence
of large residuals on training — that's the intended behavior when big
residuals are unwanted outliers corrupting a fit to the bulk of the data.
Here, the "large residuals" the likelihood learns to discount **are**
$100+ cards — the exact regime we want the model paying *more* attention
to, not less. Fine-grained MAE confirms this isn't just a calibration
artifact: point-prediction error at `$500-1000`/`$1000+` got
*categorically* worse under Student-t (MAE roughly doubled), not just
under/over-covered. Robust likelihoods are the right tool when outliers
are noise to be resisted; they're the wrong tool when the "outliers" are
the target regime itself. Dropped from further consideration on this
basis, not just because the number came out lower.

Also caught and fixed along the way: `StudentTLikelihood`'s `marginal()`
approximates via sampling (`_draw_likelihood_samples`, 10 draws) rather
than GaussianLikelihood's closed form, so `pred.mean`/`pred.variance` come
back shaped `(10, batch)` instead of `(batch,)` — `_predict()` in
`train_model2.py` now detects this and collapses it via the law of total
variance (`Var[y] = E[Var(y|f)] + Var[E(y|f)]`) rather than assuming a
single Gaussian-style moment per point.

**Follow-up: graduated oversampling + a low-price cap** (`cap_low_price`/
`oversample_ranges` config fields, same file). The flat oversample above
treats every bin >=$100 the same; asked instead to weight $100-200/$200-500
most, $500-1000 less, and leave $1000+ untouched (20 test points whose
price plausibly reflects collectibility/provenance this feature set can't
capture, not something worth forcing more training weight onto), and to
try combining that with Model 1's low-price cap (downsample the two
oversized cheap bins, `_cap_bins`, same 30-bin/8000-cap convention).
Three variants, calibrated the same way:

| variant | $10–$100 | $100+ | $100-200 | $200-500 | $500-1000 | $1000+ |
|---|---|---|---|---|---|---|
| baseline (no oversample) | 76.1% | 58.8% | 60.9% | 43.1% | 31.6% | 0.0% |
| flat oversample (all $100+ equally) | 75.7% | 61.5% | 47.8% | 52.9% | 42.1% | 25.0% |
| cap low-price + graduated oversample | 70.9% | 58.8% | 52.2% | 58.8% | 47.4% | 25.0% |
| **graduated oversample only (no cap)** | **76.0%** | **67.6%** | 44.6% | 49.0% | 47.4% | 10.0% |

(Sub-bucket columns are raw/uncalibrated coverage — too few points per
sub-bucket, 19-92, to calibrate independently; $10-100/$100+ are the
calibrated numbers.)

Dropping the low-price cap and keeping only the graduated high-end
oversampling is the best result so far: `$100+` calibrated coverage
reaches 67.6% (up from 58.8% baseline) with essentially zero cost to
`$10-100` (76.0% vs. 76.1% baseline) — the cap was what caused the
$10-100 regression in the combined variant, not the oversampling.
Interesting secondary finding: this variant's raw/uncalibrated sub-bucket
coverages aren't the best individually in every sub-range (e.g. $100-200
is worse than baseline's raw number), but they're more *uniform* across
$100-200 through $500-1000 (44-49%) than baseline's widely uneven spread
(61% down to 0%) — and a single group-conditional correction can equalize
a bucket's coverage much better when the underlying miscalibration is
uniform across it rather than lopsided, which is plausibly why this
variant calibrates best in aggregate despite not winning every raw
sub-bucket. **Current best Model 2 configuration**: full architecture
(text+image+tabular, joint_dim=48) + graduated high-price oversampling,
no low-price cap. Not yet re-combined with the joint_dim=96 tuning result
or the tabular_only ablation — an open next step, not yet tried.

**Follow-up: is `num_inducing=750` too few?** Swept 250/500/750/1500/2500
on top of the current best config. Answer: no clear evidence it's a
bottleneck — a genuine undersizing would show a monotonic improvement as
the count increases, but the calibrated results are noisy and
non-monotonic (1500 marginally best, but 2500 — more points than
anything — is one of the *worst*; 250 beats 750 on the overall number).
Single-seed-per-candidate, so some of this is run-to-run noise, but the
total absence of any upward trend across a 10x range means inducing-point
count isn't the constraint; whatever's limiting `$100+` performance is
elsewhere (feature richness / fundamental predictability), not GP
capacity. Not worth further tuning.

**Follow-up: tag selection by price-correlation instead of raw
frequency.** `oracle_tags`/`art_tags` are the two messiest, highest-
cardinality tag columns (community-tagged, mixing real signal with naming
trivia) — their existing top-N-by-frequency selection picks the most
*common* tags, not the most price-*relevant* ones. Reselected both by
|Pearson correlation| with log(price_avg) instead (min. 30 occurrences to
be a candidate, to avoid spurious correlations off a handful of cards;
see `_select_top_n_by_price_correlation` in
`prepare_training_data_model2.py`). Only 33/100 (oracle_tags) and 25/100
(art_tags) tags survived the switch — a substantial change, not a minor
reranking. The new top tags are intuitively far more price-relevant:
`ramp`/`repeatable-card-advantage`/`draw-engine`/`manaless-value` for
oracle_tags (genuinely powerful, price-driving mechanics) vs. the old
list's `triggered-ability`/`alliteration` (structural/flavor, not
price-relevant); `external-ip`/`the-lord-of-the-rings`/`fallout-universe`
for art_tags (Universes Beyond crossover cards, a real, well-known price
premium) vs. the old list's generic style descriptors (`pose`, `clothing`,
`male`/`female`).

Result, same config (full + graduated oversample) retrained on the new
feature set:

| | MAE | pinball | $10–100 | $100+ | overall |
|---|---|---|---|---|---|
| frequency-selected tags | 0.5731 | 0.1958 | 76.0% | 67.6% | 80.5% |
| **correlation-selected tags** | **0.5632** | **0.1903** | 72.6% | 65.9% | 80.5% |

Mixed, not a clean win: overall point accuracy (MAE, pinball) genuinely
improved, but the specific calibrated $10-100/$100+ numbers this whole
line of experiments has been optimizing are slightly *down*, not up
(within plausible noise given small bucket sizes — n=182/~100-113 — but
also plausibly a real interaction with the oversampling ranges, which were
tuned against the *old* tag set, not this one). Net conclusion: worth
keeping for the accuracy gain and the much more sensible/interpretable tag
list, but not the fix for the $100+ ceiling specifically — that driver is
still elsewhere. Re-tuning the oversample ranges against the new tag set
is an open next step, not yet tried.

**Follow-up: stacking the "best" tactics together, plus the missing
embedding-only/tags-only ablation cells.** Retrained combining every
individually-good change so far (joint_dim=96 from tuning, graduated
high-price oversampling, correlation-selected tags), across three
architecture variants — `both` (text+image+tabular, the actual "retrain"),
`embedding_only` (text+image, **no tabular/tags at all** — the one cell
the original ablation matrix never tested, since `use_tabular` had
defaulted to `True` in every prior run), and `tags_only` (tabular/tags,
no embeddings):

| variant | MAE | $10–100 | $100+ | overall |
|---|---|---|---|---|
| best-so-far (jd48, frequency tags, oversample only) | 0.5731 | **76.0%** | **67.6%** | 80.5% |
| correlation tags (jd48, oversample) | 0.5632 | 72.6% | 65.9% | 80.5% |
| both (jd96, correlation tags, oversample) | 0.5606 | 72.3% | 63.2% | 80.3% |
| tags_only (jd96, correlation tags, oversample, no embeddings) | 0.5747 | 74.0% | 64.3% | 78.7% |
| **embedding_only (jd96, oversample, no tabular/tags at all)** | **1.1624** | 48.1% | 23.6% | 80.3%¹ |

¹ `embedding_only`'s overall number is misleading in the same way raw
aggregates always are here — its calibration step had *zero* validation
points predicted into the `$100+` bucket at all (the model never once
predicted a price that high without tabular features to anchor it), so
the aggregate is propped up entirely by the huge `<$1` population.

Two things confirmed clearly:
1. **`embedding_only` is a genuine collapse** (MAE roughly 2x every other
   variant, and it structurally can't represent an expensive-card
   prediction) — this is the most direct evidence yet that essentially
   all of Model 2's real price-discrimination is coming from the
   tabular/tag branch, not the text/image embeddings. Embeddings alone are
   nowhere close to sufficient.
2. **Stacking "individually best" choices didn't compound.** joint_dim=96
   was the best tuning result *against the old tag set*; paired with the
   new correlation-selected tags, both `both` and `tags_only` come out
   *worse* than the older, simpler jd48 + old-tags + oversample-only
   combination on the exact buckets that matter (`$10-100`, `$100+`) —
   the single best result across every experiment in this whole section
   remains that earlier, simpler configuration. Hyperparameters tuned
   against one feature set don't reliably transfer to a different one;
   they need to be re-tuned together, not assumed to add up.

**Revised current best Model 2 configuration, unchanged from before this
round** (this is `train_model2.BEST_CONFIG` → `training/model2.pt`,
published as `CXu0630/mtg-price-deep-gp`): full architecture, joint_dim=48, frequency-selected tags,
graduated high-price oversampling, no low-price cap. Correlation-selected
tags and joint_dim=96 are each individually defensible (better raw
MAE/pinball, or better on the old tag set respectively) but neither beats
this combination on the calibrated buckets that matter, and stacking them
together does worse still. Re-tuning joint_dim/oversample ranges jointly
against the new correlation-selected tags, rather than reusing values
tuned against the old ones, is the natural next step if this is worth
continuing to chase.

**Follow-up: interval width.** Calibrated 80% intervals are wide in
absolute terms -- converting `width_cal` (log-price) to a price ratio,
$10-100 spans ~11.7x (q90/q10) and $100+ ~15.3x. Some of that is genuine
irreducible difficulty (comps/combo-driven prices like Paradoxical
Outcome aren't visible in any feature we have), but the *group-conditional
flat correction* is also a blunt instrument: it adds the same absolute
widening to every card in a bucket, sized to fix the worst-calibrated card
in it.

Tried **normalized/studentized conformal calibration** (Lei et al.) to fix
this specifically: scale the correction by each card's own GP posterior
std instead of applying a flat bucket-wide add, so confident predictions
get a smaller widening than uncertain ones (`scripts/calibrate_model2.py`
now runs both "flat" and "normalized" side by side). Result: **no
difference** -- flat and normalized give essentially identical coverage
and width. Root cause, confirmed empirically rather than assumed: the
GP's own posterior std is **nearly constant across every prediction** --
within any given predicted-price bucket its coefficient of variation is
only 0.3-2% (checked directly: min/max std per bucket differ by only a
few hundredths). Checked the actual learned hyperparameters to find out
why, rather than guessing: observation noise = 0.340, kernel outputscale
= 0.871 -- noise does *not* dominate outputscale, so the naive "noise
swamps signal" story is wrong. What's actually happening: mean predictive
variance (std^2 ~ 0.585^2 = 0.342) sits almost exactly *at* the noise
floor (0.340), meaning the kernel's own uncertainty-reduction term
(`k(x*,Z) K(Z,Z)^-1 k(Z,x*)`, how much a query is "explained away" by
proximity to the inducing points) is canceling out nearly all of the
0.871 outputscale for virtually every point, training and test alike --
the model reports near-total (epistemic) confidence almost everywhere,
with only a handful of points showing any residual uncertainty above the
noise floor. This is a known deep-kernel-learning pathology, not a bug
specific to this implementation: when the neural feature extractor is
trained *jointly* with the GP, it can learn to compress every input
toward looking similar to some inducing point (in embedding space) simply
because doing so lowers the ELBO's complexity penalty -- which defeats
the GP's uncertainty quantification without technically breaking
anything. All of the adaptivity this pipeline has achieved so far comes
from bucketing on the *predicted mean*, not from the model's own
uncertainty at all.

This means normalized calibration isn't a dead end so much as *blocked*:
it's only as good as the uncertainty signal feeding it, and that signal
has structurally collapsed.

**Two fixes tried for the collapse itself, both negative:**
1. Proper residual/bi-Lipschitz spectral norm (`residual_spectral_norm`
   in `Config` -- `x + spectral_norm(Linear)(relu(x))` blocks, the actual
   DUE-style fix, not the incomplete plain-feedforward version tried
   first). CV of predictive std: 0.79%, barely different from baseline's
   1.23% -- noise/outputscale still collapse to the same pattern
   (std^2 ~= noise almost exactly). Even a genuine bi-Lipschitz
   constraint on the feature extractor didn't unlock heteroscedastic
   uncertainty. This suggests the cause isn't (purely) embedding
   collapse specifically -- more likely a structural property of this
   SVGP setup: 750 inducing points may densely cover the ~48-dim joint
   embedding manifold regardless of its geometry, leaving most queries
   "close enough" to some inducing point either way.
2. **Replacing the GP with a direct quantile-regression head**
   (`scripts/experiments/train_model2_quantile.py` -- same wide+deep feature
   extractor, but a small MLP head outputting monotonic q10/q50/q90 via
   pinball loss, parameterized as `q50 = q10 + softplus(...)`,
   `q90 = q50 + softplus(...)` so ordering can't invert). This is the one
   that actually worked *on the specific thing being fixed*: interval-
   width coefficient of variation is **22-41% within buckets**, vs. the
   GP's ~1% -- a genuine ~7x spread in interval width across different
   cards (0.60 to 4.42 log-units), confirming this sidesteps the collapse
   entirely, exactly as Model 1's CatBoost quantile heads already showed
   was possible.

   But it's a real trade-off, not a strict win: calibrated bucket
   coverage is *worse* than the GP on both hard buckets -- $10-100: 73.0%
   vs. GP's 76.0%; $100+: 55.5% vs. GP's 67.6% (overall ties at ~81%).
   This is a first, untuned pass (the GP went through many rounds of
   oversampling/tuning before reaching its current numbers; the quantile
   head has had none), so this isn't a fully fair comparison yet -- but
   as-is, it doesn't beat the GP on the metric this whole line of work has
   been optimizing, even though it wins decisively on interval
   adaptivity. Also caught: a numerical-stability bug in the unconstrained
   quantile head (one test row's median prediction overflowed to `inf`,
   turning a real median error of 1.14 at $100+ into a misleading mean of
   7.62) -- needs a bounding activation or gradient clipping if this path
   is pursued further, not disqualifying on its own since coverage doesn't
   depend on miss magnitude.

   Undecided: whether to invest tuning effort into the quantile head to
   see if it can match/beat the GP's coverage while keeping its adaptivity
   advantage, or accept the GP's current numbers (better calibrated,
   uniform-width intervals) as good enough and treat the collapse as a
   known, documented limitation.

**Quantile head: tuned, ablated, bug-fixed.** First, a real bug caught
mid-tuning: gradient clipping alone did *not* prevent the runaway-output
pathology (several tuning candidates hit MAE 4.9-6.3, dominated by single
catastrophic predictions -- confirmed the tuning search's own ranking
score was corrupted by this too, not just the reported metric). Fixed by
clamping the head's raw pre-softplus output to +-10 log-price-units
(directly bounds the achievable range regardless of what the optimizer
does over many steps, rather than only bounding the speed of change) --
after the fix, mean and median error track each other closely again
(0.578 vs 0.393) and max error is a bounded 6.6, not inf.

`scripts/experiments/tune_model2_quantile.py` (8 candidates over joint_dim/
head_hidden_dim/lr) found joint_dim=32, head_hidden_dim=128, lr=0.003 best.
Calibrated result: $10-100 = 71.1% (untuned quantile head: 73.0%, GP:
76.0%), **$100+ = 65.9%** (untuned quantile head: 55.5%, GP: 67.6% -- now
essentially tied with the GP), overall 80.3%. Width CV stays high
(22-53% across buckets) -- tuning recovered most of the coverage gap
without giving up the adaptivity win.

`scripts/experiments/check_model2_quantile_ablation_by_bucket.py` reproduces the
GP's ablation finding independent of architecture: `tabular_only` is
again the best performer at `$100+` (54.4% raw coverage, lowest median
MAE at 0.97) -- the tags/tabular signal dominates regardless of which
model sits on top of the embeddings.

`scripts/experiments/show_model2_quantile_examples.py` against the same held-out
cards as the GP's demo (same seed, directly comparable): the quantile
head *covers* Snapcaster Mage (true $41.63, predicted [$4.55, $61.88])
where the GP missed it, but narrowly misses Sol Ring ($869.99 against an
$818 cap) where the GP covered it -- different failure patterns, not a
strict improvement. Both models miss Paradoxical Outcome identically
(true $104, both predict single digits) -- strong confirmation that's a
genuine blind spot (plausibly combo/banlist-driven pricing invisible to
either architecture) rather than a model-specific weakness.

**Current verdict**: after tuning, the quantile head is roughly on par
with the GP on calibrated coverage (tied at $100+, GP slightly ahead at
$10-100) while decisively winning on interval adaptivity (genuine per-
card-varying width vs. the GP's near-constant uncertainty). Given the
GP's uncertainty collapse looks structural (survived both attempted
fixes) rather than a specific bug, the quantile head is the more
promising path forward if per-card-sharp intervals matter going forward
-- though this trades away the "prediction is literally a kernel-weighted
sum of comps" framing that motivated choosing a GP in SPECS.md S3.
Not yet decided which becomes the primary/default Model 2.

### 4.3 Deferred: fine-tuning the text/image encoders

Considered, not doing it for v1. The small MLP projection head (step 2 above)
is trained jointly with the GP and can already re-weight a frozen embedding
for the downstream task; that's usually enough when the frozen embedding's
problem is "signal's in there, just not on the axis we want" rather than the
embedding having genuinely discarded the relevant information. Fine-tuning a
full encoder needs more data than this project has, risks overfitting, and
is partially redundant with what the projection head + GP already learn
end-to-end. Revisit only if a qualitative check (embed a handful of cards,
inspect nearest neighbors, confirm they're functionally sensible and not just
topically similar) shows the frozen embeddings are the actual bottleneck —
don't invest here speculatively.

## 5. Hardware / compute plan

Available: T4/L4, and H100 access.

- **Model 1**: CPU only.
- **Model 2**: the MLP head is trivial; the GP is the part to watch, and
  it's about algorithm choice (SVGP, not exact), not raw hardware.
  Estimated **15–60 min per training run** on a T4/L4 at ~50k training
  examples (rescale linearly if actual N differs), assuming embeddings are
  precomputed/cached. A full experimentation cycle (sweeping inducing-point
  count, kernel choice, embedding dim — call it 5–15 runs) is a few hours to
  about a day of wall-clock time on a single T4/L4.
- **H100**: not required for either model as specified. Reserve it for (a)
  fast one-time batch embedding of the full card-art corpus through CLIP, or
  (b) if frozen CLIP embeddings prove insufficient and fine-tuning the
  image/text encoder becomes necessary — not a v1 requirement.

## 6. Evaluation (draft — not yet agreed)

- **Point/quantile accuracy**: pinball (quantile) loss on held-out printings,
  at whatever quantile levels are chosen for the "range" output (see Open
  Questions).
- **Calibration**: for the GP, check that the stated posterior interval
  actually contains the true price at the expected rate (e.g., an 80%
  interval should contain ~80% of held-out true prices).
- **Temporal robustness backtest**: per the discussion in §3, this is the
  most important check specific to this project — hold out a recent time
  window, train only on data before it, and verify that a card's *comps
  structure* (its nearest neighbors) stays sensible across the gap, not just
  that point accuracy is good. This directly tests the assumption the whole
  architecture rests on.
- **Held-out split must respect printing, not card-concept**, given target
  grain — don't let two printings of the same card leak across train/test.

## 7. Shelved: analytical framing (not in scope, kept for later)

Recorded so the reasoning isn't lost, not because it's happening now:

- Reframe price as `f(fundamentals) + residual`, where fundamentals =
  scarcity + mechanical value + EDHREC play rate (deliberately excluding
  hype-adjacent signals), and the residual is read as an "irrationality"/hype
  premium. Outliers like LTR would show up as large positive residuals.
- To isolate art's effect *net of* rarity (since showcase/borderless
  treatments are correlated with rarity by construction): residualize the art
  embedding on rarity, residualize price on rarity, correlate the two
  residuals (Frisch–Waugh–Lovell).
- Would favor an interpretable/decomposable model (GBDT + SHAP, or an
  explicit hedonic/GAM) over the black-box GP, since the goal shifts from
  prediction to attribution.
- References: Rosen, *"Hedonic Prices and Implicit Markets"* (1974); Mei &
  Moses, *"Art as an Investment and the Underperformance of Masterpieces"*
  (AER, 2002); Lundberg & Lee, SHAP (NeurIPS 2017).

## 8. Open questions (blocking implementation)

- [x] ~~Confirm the per-card dataset's actual grain~~ — done: `mtg-card-market-dataset`
  is one row per (printing, finish), matching §1's target grain exactly.
- [x] ~~Confirm actual row count (N)~~ — done: 149,078 rows in `mtg-card-market-dataset`,
  142,012 (95.3%) with a price match, 104,321 (70%) with a pull-rate match.
  Real N is ~3x the ~50k assumed in §5 — rescale those time estimates
  before relying on them.
- [ ] **Pick the price-normalization baseline** for §3 (contemporaneous
  cohort percentile? ratio to same-set booster price? something else?).
- [ ] **Pick the quantile levels** the "range" output actually reports (e.g.
  p10/p90? min/max of weighted neighbor spread? GP ±1.28σ?).
- [ ] **EDHREC data collection** — no pipeline exists yet; also confirm
  EDHREC's terms of use permit the intended collection/use before building
  one (this repo's existing TCGplayer price-fetching used public,
  unauthenticated-but-permissive endpoints — EDHREC's policy hasn't been
  checked and may differ).
- [x] ~~Fix the rarity/pull-rate join at the source~~ — done (see §2).
