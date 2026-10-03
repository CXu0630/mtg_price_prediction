"""Model 2, quantile-regression variant: same wide+deep feature extractor
as train_model2.py, but a direct quantile-regression MLP head (trained via
pinball loss, same mechanism as Model 1's CatBoost MultiQuantile) instead
of an SVGP.

Why this exists (SPECS.md S4.2's interval-width follow-up): the SVGP's
predictive variance collapsed to being nearly constant across every card
(confirmed empirically -- coefficient of variation ~1%), which defeated
the entire point of choosing a GP (a genuine per-card uncertainty
estimate). Neither of two GP-side fixes (plain spectral norm,
feature-extractor LR scaling) changed that. This sidesteps the problem
entirely: a pinball-loss quantile head has no equivalent collapse
pathology -- Model 1 already proved this works (its calibration
corrections varied meaningfully bucket-to-bucket with no special
tricks needed) -- at the cost of giving up the "prediction is literally a
kernel-weighted sum of comps" framing from SPECS.md S3 that motivated the
GP in the first place.

Quantiles are parameterized to be monotonic by construction (q50 = q10 +
softplus(delta), q90 = q50 + softplus(delta)) rather than trusting three
independent pinball losses not to cross, which they aren't guaranteed to
avoid on their own.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_model2 import (
    FEATURES_PATH, TEXT_EMB_PATH, IMAGE_EMB_PATH, SEED,
    WideAndDeepFeatureExtractor, _cap_bins, _oversample_ranges, _split,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_OUT_PATH = ROOT / "training" / "model2_quantile.pt"
QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class QuantileConfig:
    joint_dim: int = 48
    head_hidden_dim: int = 64
    lr: float = 3e-3
    batch_size: int = 1024
    n_epochs: int = 25
    patience: int = 4
    use_text: bool = True
    use_image: bool = True
    use_tabular: bool = True
    seed: int = SEED
    checkpoint_path: Path = MODEL_OUT_PATH
    features_path: Path = FEATURES_PATH
    label: str = "quantile_default"
    cap_low_price: bool = False
    low_price_cap: int = 8000
    low_price_cap_n_bins: int = 30
    oversample_high_price: bool = False
    oversample_ranges: tuple[tuple[float, float, int], ...] = (
        (float(np.log(100)), float(np.log(200)), 1200),
        (float(np.log(200)), float(np.log(500)), 1200),
        (float(np.log(500)), float(np.log(1000)), 400),
    )


class QuantileHead(nn.Module):
    """Maps the joint embedding to 3 monotonic quantiles (q10 <= q50 <=
    q90) by construction -- q50/q90 are q10 plus a softplus'd gap, rather
    than three independent outputs that pinball loss alone doesn't
    guarantee stay ordered."""
    def __init__(self, joint_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(joint_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        # Gradient clipping alone wasn't enough (confirmed empirically --
        # several tuning candidates still produced catastrophic runaway
        # predictions that dominated their mean-based metrics, including
        # the tuning search's own ranking score). Clamping bounds the
        # *achievable* output range directly regardless of what the
        # optimizer does over many steps: +-10 log-price-units covers the
        # real observed range (max real log-price ~10.07) with room to
        # spare, and softplus(10) ~= 10 bounds the max quantile gap too.
        raw = torch.clamp(raw, -10.0, 10.0)
        q10 = raw[:, 0]
        q50 = q10 + F.softplus(raw[:, 1])
        q90 = q50 + F.softplus(raw[:, 2])
        return torch.stack([q10, q50, q90], dim=1)


class QuantileModel(nn.Module):
    def __init__(self, feature_extractor: WideAndDeepFeatureExtractor, joint_dim: int, head_hidden_dim: int = 64):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.quantile_head = QuantileHead(joint_dim, head_hidden_dim)

    def forward(self, cat_idx, wide, text_emb, image_emb) -> torch.Tensor:
        z = self.feature_extractor(cat_idx, wide, text_emb, image_emb)
        return self.quantile_head(z)


def _pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles=QUANTILES) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        diff = target - preds[:, i]
        losses.append(torch.maximum(q * diff, (q - 1) * diff))
    return torch.stack(losses, dim=1).mean()


def load_data(seed: int = SEED, smoke: bool = False, features_path: Path = FEATURES_PATH) -> dict:
    """Same shape as train_model2.py's load_data(), reimplemented here
    rather than imported since it's a small, self-contained function and
    this file should stand alone as the quantile-model pipeline."""
    from train_model2 import (
        CATEGORICAL_COLS, NEEDS_ISNA_FLAG, NUMERIC_TO_SCALE, _build_vocab, _encode_categorical,
    )

    df = pd.read_parquet(features_path)
    text_emb_all = np.load(TEXT_EMB_PATH)
    image_emb_all = np.load(IMAGE_EMB_PATH)

    if smoke:
        df = df.iloc[:4000].reset_index(drop=True)
        text_emb_all = text_emb_all[:4000]
        image_emb_all = image_emb_all[:4000]

    y = df["__target_log_price_avg"].to_numpy(dtype=np.float32)
    group_key = df["__group_oracle_id"].to_numpy()

    for col in NEEDS_ISNA_FLAG:
        df[f"{col}__isna"] = df[col].isna()

    train_idx, val_idx, test_idx = _split(group_key, seed)

    cat_vocabs = {col: _build_vocab(df[col].iloc[train_idx]) for col in CATEGORICAL_COLS}
    cat_idx_all = np.stack([_encode_categorical(df[col], cat_vocabs[col]) for col in CATEGORICAL_COLS], axis=1)

    rate_cols = [c for c in df.columns if c.endswith("_rate_percent")]
    numeric_to_scale = NUMERIC_TO_SCALE + rate_cols
    wide_cols = [c for c in df.columns if c not in CATEGORICAL_COLS and not c.startswith("__")]
    wide_df = df[wide_cols].copy()
    for col in numeric_to_scale:
        median = wide_df[col].iloc[train_idx].median()
        wide_df[col] = wide_df[col].fillna(0.0 if pd.isna(median) else median)
    wide_df = wide_df.astype(np.float32)
    scale_cols = [c for c in numeric_to_scale if c in wide_df.columns]
    means = wide_df[scale_cols].iloc[train_idx].mean().fillna(0.0)
    stds = wide_df[scale_cols].iloc[train_idx].std().replace(0, 1.0).fillna(1.0)
    wide_df[scale_cols] = (wide_df[scale_cols] - means) / stds
    wide_all = wide_df.to_numpy(dtype=np.float32)

    device = torch.device("cpu")

    def to_tensors(idx):
        return (
            torch.from_numpy(cat_idx_all[idx]).to(device),
            torch.from_numpy(wide_all[idx]).to(device),
            torch.from_numpy(text_emb_all[idx]).to(device),
            torch.from_numpy(image_emb_all[idx]).to(device),
            torch.from_numpy(y[idx]).to(device),
        )

    return {
        "train": to_tensors(train_idx), "val": to_tensors(val_idx), "test": to_tensors(test_idx),
        "cat_vocab_sizes": [len(cat_vocabs[c]) for c in CATEGORICAL_COLS],
        "n_wide": wide_all.shape[1], "text_dim": text_emb_all.shape[1], "image_dim": image_emb_all.shape[1],
        "device": device,
    }


def build_model(data: dict, config: QuantileConfig) -> QuantileModel:
    feature_extractor = WideAndDeepFeatureExtractor(
        n_wide=data["n_wide"], cat_vocab_sizes=data["cat_vocab_sizes"],
        text_dim=data["text_dim"], image_dim=data["image_dim"], joint_dim=config.joint_dim,
        use_text=config.use_text, use_image=config.use_image, use_tabular=config.use_tabular,
    ).to(data["device"])
    return QuantileModel(feature_extractor, config.joint_dim, config.head_hidden_dim).to(data["device"])


def _predict(model: QuantileModel, tensors, batch_size: int) -> np.ndarray:
    cat, wide, text_emb, image_emb, y = tensors
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            sl = slice(start, start + batch_size)
            preds.append(model(cat[sl], wide[sl], text_emb[sl], image_emb[sl]))
    return torch.cat(preds).numpy()


def run(data: dict, config: QuantileConfig, verbose: bool = True) -> dict:
    torch.manual_seed(config.seed)
    model = build_model(data, config)
    train_cat, train_wide, train_text, train_image, train_y = data["train"]
    val_y = data["val"][4].numpy()

    if config.cap_low_price:
        idx_after_cap = _cap_bins(train_y.numpy(), config.low_price_cap, config.low_price_cap_n_bins, config.seed)
        if verbose:
            print(f"[{config.label}] low-price cap: {train_y.shape[0]} -> {len(idx_after_cap)} distinct rows")
    else:
        idx_after_cap = np.arange(train_y.shape[0])

    if config.oversample_high_price:
        local_extra_idx = _oversample_ranges(train_y.numpy()[idx_after_cap], config.oversample_ranges, config.seed)
        epoch_base_idx = idx_after_cap[local_extra_idx]
        if verbose:
            print(f"[{config.label}] high-price oversample: {len(idx_after_cap)} -> {len(epoch_base_idx)} rows/epoch")
    else:
        epoch_base_idx = idx_after_cap
    epoch_base_idx_t = torch.from_numpy(epoch_base_idx)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    epochs_since_improve = 0

    for epoch in range(config.n_epochs):
        model.train()
        perm = epoch_base_idx_t[torch.randperm(len(epoch_base_idx_t))]
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for start in range(0, len(perm), config.batch_size):
            batch = perm[start:start + config.batch_size]
            optimizer.zero_grad()
            preds = model(train_cat[batch], train_wide[batch], train_text[batch], train_image[batch])
            loss = _pinball_loss(preds, train_y[batch])
            # Unlike the GP, this head has no bound on output magnitude --
            # a single bad batch pushing a raw logit very large propagates
            # through softplus into a runaway quantile prediction (caught
            # empirically: one test row's median overflowed to inf).
            # Gradient clipping is the standard, cheap guard against this.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_preds = _predict(model, data["val"], config.batch_size)
        val_pinball = _pinball_loss(torch.from_numpy(val_preds), torch.from_numpy(val_y)).item()

        if verbose:
            print(f"[{config.label}] epoch {epoch+1}/{config.n_epochs}  train_pinball={epoch_loss/n_batches:.4f}  "
                  f"val_pinball={val_pinball:.4f}  ({time.time()-t0:.1f}s)")

        if val_pinball < best_val_loss - 1e-4:
            best_val_loss = val_pinball
            epochs_since_improve = 0
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": config}, config.checkpoint_path)
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config.patience:
                if verbose:
                    print(f"[{config.label}] early stopping at epoch {epoch+1}")
                break

    checkpoint = torch.load(config.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    test_preds = _predict(model, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    mae = float(np.mean(np.abs(test_preds[:, 1] - test_y)))
    coverage = float(np.mean((test_y >= test_preds[:, 0]) & (test_y <= test_preds[:, 2])))
    test_pinball = float(_pinball_loss(torch.from_numpy(test_preds), torch.from_numpy(test_y)).item())

    if verbose:
        print(f"[{config.label}] test MAE={mae:.4f}  coverage80={coverage*100:.1f}%  pinball={test_pinball:.4f}")

    return {"label": config.label, "best_val_pinball": best_val_loss,
            "test_mae": mae, "test_coverage80": coverage, "test_pinball": test_pinball}


def main(smoke: bool = False) -> int:
    np.random.seed(SEED)
    data = load_data(smoke=smoke)
    config = replace(QuantileConfig(), n_epochs=2 if smoke else QuantileConfig.n_epochs)
    run(data, config)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(smoke="--smoke" in sys.argv))
