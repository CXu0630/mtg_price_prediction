"""Model 2 -- deep kernel Gaussian Process (SPECS.md S4.2).

Wide & deep architecture: a small MLP projects [text embedding, image
embedding, categorical embeddings, wide numeric/boolean tabular features]
into a joint embedding space, and a sparse variational GP (SVGP) puts a
kernel over that joint space, trained end-to-end by maximizing the
variational ELBO against log(price_avg) -- same target as Model 1 (SPECS.md
S3), same train/test split philosophy (group by oracle_id, SPECS.md S6).

Model classes + preprocessing live in src/mtg_price_prediction/deep_gp.py
(shared with inference / the published model). Running this script trains
BEST_CONFIG -> training/model2.pt.

Config-driven (`Config` below) so scripts/experiments/tune_model2.py and
scripts/experiments/check_model2_ablation_by_bucket.py can reuse `load_data()` /
`run()` directly instead of re-deriving the pipeline -- the data loading
(embedding joins, standardization) is the expensive/shared part and is
identical across every hyperparameter or ablation variant, only the model
architecture and training config change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "training" / "features_model2.parquet"  # live default: frequency-selected tags
CORRELATION_FEATURES_PATH = ROOT / "training" / "features_model2_correlation_tags.parquet"  # archived variant
TEXT_EMB_PATH = ROOT / "training" / "model2_text_emb.npy"
IMAGE_EMB_PATH = ROOT / "training" / "model2_image_emb.npy"
MODEL_OUT_PATH = ROOT / "training" / "model2.pt"

from mtg_price_prediction.deep_gp import (  # noqa: F401 -- re-exported for the experiment scripts
    CATEGORICAL_COLS, NEEDS_ISNA_FLAG, NUMERIC_TO_SCALE, DKLModel, WideAndDeepFeatureExtractor,
    _build_vocab, _encode_categorical, fit_preprocessor, predict_moments, transform,
)

SEED = 0


@dataclass(frozen=True)
class Config:
    joint_dim: int = 48
    num_inducing: int = 750
    lr: float = 3e-3
    batch_size: int = 1024
    n_epochs: int = 25
    patience: int = 4
    use_text: bool = True
    use_image: bool = True
    use_tabular: bool = True  # categorical embeddings + wide numeric/boolean branch
    seed: int = SEED
    checkpoint_path: Path = MODEL_OUT_PATH
    # Which wide-feature snapshot this config trains against -- oracle_tags/
    # art_tags selection method changes column *meaning* without changing
    # column *count* (SPECS.md S4.2's tag-correlation follow-up), so a
    # checkpoint loaded against the wrong snapshot fails silently (same
    # shape, wrong semantics) rather than erroring. Recorded here and
    # threaded through load_data() so every checkpoint is self-describing.
    features_path: Path = FEATURES_PATH
    label: str = "default"
    # "gaussian" or "student_t" -- Student-t is more robust to the heavy
    # right tail (a few $1000s-$20000+ cards) than a Gaussian likelihood,
    # which weights large residuals quadratically and can let a handful of
    # extreme points distort the learned noise scale for everyone. (Tried,
    # rejected -- it discounts exactly the residuals we want the model
    # paying more attention to; see SPECS.md S4.2.)
    likelihood_type: str = "gaussian"
    # Downsamples (without replacement) any log-price histogram bin above
    # `low_price_cap` rows, same bins/cap as Model 1's rebalancing
    # (RESAMPLE_N_BINS=30, RESAMPLE_CAP_PER_BIN=8000, SPECS.md S4.1) --
    # only the couple of oversized cheap bins are actually large enough to
    # get touched.
    cap_low_price: bool = False
    low_price_cap: int = 8000
    low_price_cap_n_bins: int = 30
    # Oversamples (with replacement) explicit named price ranges up to a
    # target count each -- graduated by request: $100-200/$200-500 get the
    # most weight (moderate boost, ~2-3x their natural count), $500-1000
    # less, and $1000+ is left untouched on the judgment that cards up
    # there are driven by collectibility/provenance factors this feature
    # set mostly can't capture, so forcing more training weight onto that
    # handful of points is more likely to overfit noise than learn signal.
    oversample_high_price: bool = False
    oversample_ranges: tuple[tuple[float, float, int], ...] = (
        (float(np.log(100)), float(np.log(200)), 1200),
        (float(np.log(200)), float(np.log(500)), 1200),
        (float(np.log(500)), float(np.log(1000)), 400),
    )
    # Both address the same diagnosed problem (SPECS.md S4.2's interval-
    # width follow-up): jointly-trained deep kernel GPs can collapse their
    # embedding space -- the neural feature extractor learns to make every
    # input look similarly "close" to some inducing point, since that
    # lowers the ELBO's complexity penalty, which silently destroys the
    # GP's per-point uncertainty (confirmed here: predictive variance was
    # nearly constant everywhere; noise ~= mean predictive variance).
    # - `spectral_norm`: constrains the feature extractor to be
    #   bi-Lipschitz (can't arbitrarily contract distances between
    #   different inputs) -- the fix from van Amersfoort et al.'s DUE
    #   ("Deterministic Uncertainty Estimation"), which names and solves
    #   exactly this failure mode for deep kernel GPs.
    # - `feature_extractor_lr_scale`: an older, simpler mitigation from
    #   Wilson's original DKL work -- give the feature extractor a much
    #   smaller learning rate than the GP's own parameters, so it can't
    #   race ahead and collapse the representation before the kernel
    #   hyperparameters have a chance to adapt to it.
    spectral_norm: bool = False
    feature_extractor_lr_scale: float = 1.0
    # The *proper* DUE-style fix, distinct from plain `spectral_norm`
    # above: residual blocks (`x + spectral_norm(Linear)(relu(x))`) give a
    # genuine bi-Lipschitz bound (can't expand OR contract distances
    # beyond a bounded factor). Plain spectral norm on a feedforward layer
    # only bounds expansion -- it does nothing to stop the collapse
    # (contraction) this is meant to fix, which is why the first attempt
    # (SPECS.md S4.2) showed no change in predictive-variance CV at all.
    residual_spectral_norm: bool = False
    residual_hidden_dim: int = 128
    residual_n_blocks: int = 2


def _split(group_key: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(group_key))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, groups=group_key))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_idx, groups=group_key[temp_idx]))
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def load_data(seed: int = SEED, smoke: bool = False, features_path: Path = FEATURES_PATH) -> dict:
    """Loads + preps everything that's shared across every hyperparameter/
    ablation variant (embedding joins, categorical vocabs, standardization
    -- all fit on the train split only, via deep_gp.fit_preprocessor). The
    fitted preprocessor is returned as `data["prep"]` so it can be exported
    alongside the weights. Independent of Config, so callers that sweep
    configs over a *fixed* `features_path` should call this once and reuse
    the result -- but a sweep across `features_path` values (e.g. comparing
    tag-selection methods) must call it again per path, since column
    meaning differs between snapshots even at equal width."""
    df = pd.read_parquet(features_path)
    text_emb_all = np.load(TEXT_EMB_PATH)
    image_emb_all = np.load(IMAGE_EMB_PATH)

    if smoke:
        df = df.iloc[:4000].reset_index(drop=True)
        text_emb_all = text_emb_all[:4000]
        image_emb_all = image_emb_all[:4000]

    y = df["__target_log_price_avg"].to_numpy(dtype=np.float32)
    group_key = df["__group_oracle_id"].to_numpy()

    train_idx, val_idx, test_idx = _split(group_key, seed)
    prep = fit_preprocessor(df, train_idx)
    cat_idx_all, wide_all = transform(df, prep)

    device = torch.device("cpu")  # gpytorch's variational strategy uses ops not yet reliable on MPS

    def to_tensors(idx):
        return (
            torch.from_numpy(cat_idx_all[idx]).to(device),
            torch.from_numpy(wide_all[idx]).to(device),
            torch.from_numpy(text_emb_all[idx]).to(device),
            torch.from_numpy(image_emb_all[idx]).to(device),
            torch.from_numpy(y[idx]).to(device),
        )

    return {
        "train": to_tensors(train_idx),
        "val": to_tensors(val_idx),
        "test": to_tensors(test_idx),
        "prep": prep,
        "cat_vocab_sizes": [len(prep["cat_vocabs"][c]) for c in CATEGORICAL_COLS],
        "n_wide": wide_all.shape[1],
        "text_dim": text_emb_all.shape[1],
        "image_dim": image_emb_all.shape[1],
        "device": device,
    }


def _build_likelihood(config: Config) -> gpytorch.likelihoods.Likelihood:
    if config.likelihood_type == "gaussian":
        return gpytorch.likelihoods.GaussianLikelihood()
    if config.likelihood_type == "student_t":
        return gpytorch.likelihoods.StudentTLikelihood()
    raise ValueError(f"unknown likelihood_type: {config.likelihood_type!r}")


def _cap_bins(y: np.ndarray, cap: int, n_bins: int, seed: int) -> np.ndarray:
    """Row positions (into `y`) to keep after capping each log-price bin at
    `cap` rows, randomly subsampling bins that exceed it -- identical
    method to Model 1's rebalancing (scripts/train_model1.py's
    `_cap_bins`), reproduced here rather than imported so Model 2's
    pipeline stays independent of Model 1's."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(y.min(), y.max(), n_bins + 1)
    bin_ids = np.digitize(y, edges[1:-1])

    keep = []
    for b in range(n_bins):
        in_bin = np.flatnonzero(bin_ids == b)
        if len(in_bin) > cap:
            in_bin = rng.choice(in_bin, size=cap, replace=False)
        keep.append(in_bin)
    return np.sort(np.concatenate(keep))


def _oversample_ranges(y: np.ndarray, ranges: tuple[tuple[float, float, int], ...], seed: int) -> np.ndarray:
    """Row positions into `y`, with `np.arange(len(y))` plus extra
    duplicated (with-replacement) positions for any named (lo, hi, target)
    range whose natural count is below `target`. Ranges outside the given
    list (e.g. $1000+, deliberately left out) are untouched."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(y))

    extra = []
    for lo, hi, target in ranges:
        range_idx = all_idx[(y >= lo) & (y < hi)]
        if 0 < len(range_idx) < target:
            extra.append(rng.choice(range_idx, size=target - len(range_idx), replace=True))

    return np.concatenate([all_idx] + extra) if extra else all_idx


def build_model(data: dict, config: Config) -> tuple[DKLModel, gpytorch.likelihoods.Likelihood]:
    train_cat, train_wide, train_text, train_image, _ = data["train"]
    device = data["device"]

    feature_extractor = WideAndDeepFeatureExtractor(
        n_wide=data["n_wide"], cat_vocab_sizes=data["cat_vocab_sizes"],
        text_dim=data["text_dim"], image_dim=data["image_dim"], joint_dim=config.joint_dim,
        use_text=config.use_text, use_image=config.use_image, use_tabular=config.use_tabular,
        spectral_norm=config.spectral_norm, residual_spectral_norm=config.residual_spectral_norm,
        residual_hidden_dim=config.residual_hidden_dim, residual_n_blocks=config.residual_n_blocks,
    ).to(device)

    n_inducing = min(config.num_inducing, train_wide.shape[0])
    rng = np.random.default_rng(config.seed)
    init_idx = rng.choice(train_wide.shape[0], size=n_inducing, replace=False)
    with torch.no_grad():
        inducing_points = feature_extractor(
            train_cat[init_idx], train_wide[init_idx], train_text[init_idx], train_image[init_idx]
        ).clone()

    model = DKLModel(feature_extractor, inducing_points).to(device)
    likelihood = _build_likelihood(config).to(device)
    return model, likelihood


def _predict(model: DKLModel, likelihood, tensors, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    cat, wide, text_emb, image_emb, _ = tensors
    return predict_moments(model, likelihood, cat, wide, text_emb, image_emb, batch_size)


def run(data: dict, config: Config, verbose: bool = True) -> dict:
    """Trains one model per `config` against `data` (from load_data()),
    early-stopping on validation pinball loss, saves the best checkpoint at
    `config.checkpoint_path`, and returns test-set metrics."""
    torch.manual_seed(config.seed)

    model, likelihood = build_model(data, config)
    train_cat, train_wide, train_text, train_image, train_y = data["train"]
    val_cat, val_wide, val_text, val_image, val_y = data["val"]

    # Cap first (genuinely removes rows -- a real subsetting of the
    # training set, same as Model 1's rebalancing), then oversample on top
    # of what's left (duplicates rows -- not genuinely new data). Distinct-
    # row count (post-cap, pre-oversample) is what `num_data` should
    # reflect: it scales the ELBO's likelihood term against the KL term,
    # and inflating it just because rows are being *repeated* would be an
    # unprincipled way to move that balance.
    if config.cap_low_price:
        idx_after_cap = _cap_bins(train_y.numpy(), config.low_price_cap, config.low_price_cap_n_bins, config.seed)
        if verbose:
            print(f"[{config.label}] low-price cap: {train_y.shape[0]} -> {len(idx_after_cap)} distinct rows")
    else:
        idx_after_cap = np.arange(train_y.shape[0])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(idx_after_cap))
    optimizer = torch.optim.Adam([
        {"params": model.feature_extractor.parameters(), "lr": config.lr * config.feature_extractor_lr_scale},
        {"params": model.gp_layer.parameters()},
        {"params": likelihood.parameters()},
    ], lr=config.lr)

    if config.oversample_high_price:
        local_extra_idx = _oversample_ranges(train_y.numpy()[idx_after_cap], config.oversample_ranges, config.seed)
        epoch_base_idx = idx_after_cap[local_extra_idx]
        if verbose:
            print(f"[{config.label}] high-price oversample: {len(idx_after_cap)} -> {len(epoch_base_idx)} rows/epoch")
    else:
        epoch_base_idx = idx_after_cap
    epoch_base_idx_t = torch.from_numpy(epoch_base_idx)

    best_val_loss = float("inf")
    epochs_since_improve = 0

    for epoch in range(config.n_epochs):
        model.train()
        likelihood.train()
        perm = epoch_base_idx_t[torch.randperm(len(epoch_base_idx_t))]
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for start in range(0, len(perm), config.batch_size):
            batch = perm[start:start + config.batch_size]
            optimizer.zero_grad()
            output = model(train_cat[batch], train_wide[batch], train_text[batch], train_image[batch])
            loss = -mll(output, train_y[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_mean, val_std = _predict(model, likelihood, data["val"], config.batch_size)
        val_pinball = np.mean([
            _pinball_loss(val_y.numpy(), val_mean + val_std * z, q)
            for q, z in [(0.1, -1.2816), (0.5, 0.0), (0.9, 1.2816)]
        ])

        if verbose:
            print(f"[{config.label}] epoch {epoch+1}/{config.n_epochs}  train_elbo_loss={epoch_loss/n_batches:.4f}  "
                  f"val_pinball={val_pinball:.4f}  ({time.time()-t0:.1f}s)")

        if val_pinball < best_val_loss - 1e-4:
            best_val_loss = val_pinball
            epochs_since_improve = 0
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "likelihood": likelihood.state_dict(), "config": config},
                       config.checkpoint_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config.patience:
                if verbose:
                    print(f"[{config.label}] early stopping at epoch {epoch+1}")
                break

    checkpoint = torch.load(config.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    likelihood.load_state_dict(checkpoint["likelihood"])

    test_mean, test_std = _predict(model, likelihood, data["test"], config.batch_size)
    test_y_np = data["test"][4].numpy()
    mae = float(np.mean(np.abs(test_mean - test_y_np)))
    q10 = test_mean - 1.2816 * test_std
    q90 = test_mean + 1.2816 * test_std
    coverage = float(np.mean((test_y_np >= q10) & (test_y_np <= q90)))
    test_pinball = float(np.mean([
        _pinball_loss(test_y_np, test_mean + test_std * z, q)
        for q, z in [(0.1, -1.2816), (0.5, 0.0), (0.9, 1.2816)]
    ]))

    if verbose:
        print(f"[{config.label}] test MAE={mae:.4f}  coverage80={coverage*100:.1f}%  pinball={test_pinball:.4f}")

    return {
        "label": config.label, "best_val_pinball": best_val_loss,
        "test_mae": mae, "test_coverage80": coverage, "test_pinball": test_pinball,
    }


# Best calibrated result so far (SPECS.md S4.2, "graduated oversample
# only"): full text+image+tabular architecture, joint_dim=48, frequency-
# selected tags, graduated $100-$1000 oversampling, no low-price cap.
BEST_CONFIG = Config(oversample_high_price=True, label="best")


def main(smoke: bool = False) -> int:
    np.random.seed(SEED)
    data = load_data(smoke=smoke)
    config = BEST_CONFIG
    if smoke:  # quick pipeline check -- never overwrite the real checkpoint
        config = replace(config, n_epochs=2, checkpoint_path=MODEL_OUT_PATH.with_name("model2_smoke.pt"))
    run(data, config)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(smoke="--smoke" in sys.argv))
