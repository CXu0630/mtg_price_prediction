"""Experiment 2: a residual-correction ("ResNet") head stacked on the deep GP.

The head cannot see the actual price at inference -- that's the unknown. What
it learns instead is the GP's *error*: input = the card's features + the GP's
posterior mean and std, target = y - GP mean, output = a correction added to
the GP mean. To train it honestly the GP errors must come from cards the GP
didn't train on, so the train split's GP predictions are out-of-fold
(5-fold GroupKFold by oracle_id: GP trained on 4 folds, predicts the 5th).
Val/test use the published full-train GP (training/model2.pt).

Head: a fresh WideAndDeepFeatureExtractor trunk (cat embeddings + wide +
text + image -> 64) concatenated with [GP mean, GP std] -> 3 residual blocks
(LayerNorm -> Linear -> ReLU -> Dropout -> Linear, plus skip) -> 1. L1 loss,
early-stopped on val MAE; last layer zero-initialised so it starts as "no
correction". Intervals keep the GP's std, recalibrated per bucket on val.

    python scripts/experiments/exp_residual_head.py
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from _gp_common import EXP_DIR, ROOT, SPLIT_SEED, metrics, print_comparison, record
from mtg_price_prediction.deep_gp import WideAndDeepFeatureExtractor
from train_model2 import BEST_CONFIG, MODEL_OUT_PATH, _predict, _split, build_model, load_data, run

FEATURES = ROOT / "training" / "features_model2.parquet"
N_FOLDS = 5
HEAD_SEEDS = (0, 1, 2)
OOF_PATH = EXP_DIR / "gp_oof_train.npz"


class ResidualHead(nn.Module):
    def __init__(self, data: dict, trunk_dim: int = 64, hidden: int = 128, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.trunk = WideAndDeepFeatureExtractor(
            n_wide=data["n_wide"], cat_vocab_sizes=data["cat_vocab_sizes"], text_dim=data["text_dim"],
            image_dim=data["image_dim"], joint_dim=trunk_dim)
        self.inp = nn.Linear(trunk_dim + 2, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, cat, wide, text, image, gp_mean, gp_std):
        h = self.inp(torch.cat([self.trunk(cat, wide, text, image), gp_mean[:, None], gp_std[:, None]], dim=-1))
        for block in self.blocks:
            h = h + block(h)
        return self.out(torch.relu(h)).squeeze(-1)


def _subset(tensors, idx):
    idx = torch.as_tensor(idx)
    return tuple(t[idx] for t in tensors)


def oof_train_predictions(data: dict, group_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if OOF_PATH.exists():
        z = np.load(OOF_PATH)
        return z["mean"], z["std"]
    n = len(group_train)
    mean, std = np.empty(n, np.float32), np.empty(n, np.float32)
    for k, (fit, held) in enumerate(GroupKFold(N_FOLDS).split(np.arange(n), groups=group_train)):
        fold_data = {**data, "train": _subset(data["train"], fit)}
        config = replace(BEST_CONFIG, label=f"oof_fold{k}", checkpoint_path=EXP_DIR / f"gp_oof_fold{k}.pt")
        run(fold_data, config, verbose=False)
        model, lik = build_model(fold_data, config)
        ckpt = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        lik.load_state_dict(ckpt["likelihood"])
        m, s = _predict(model, lik, _subset(data["train"], held), config.batch_size)
        mean[held], std[held] = m, s
        print(f"  OOF fold {k}: {len(held)} rows, MAE {np.mean(np.abs(m - data['train'][4].numpy()[held])):.3f}", flush=True)
    np.savez(OOF_PATH, mean=mean, std=std)
    return mean, std


def train_head(data, base: dict, seed: int, epochs: int = 40, patience: int = 5, lr: float = 1e-3) -> nn.Module:
    torch.manual_seed(seed)
    head = ResidualHead(data)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    cat, wide, text, image, y = data["train"]
    gm, gs = torch.from_numpy(base["train"][0]), torch.from_numpy(base["train"][1])
    resid = y - gm
    best, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(y))
        for start in range(0, len(y), 1024):
            b = perm[start:start + 1024]
            loss = torch.mean(torch.abs(head(cat[b], wide[b], text[b], image[b], gm[b], gs[b]) - resid[b]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_mae = np.mean(np.abs(predict_head(head, data["val"], base["val"]) - data["val"][4].numpy()))
        if val_mae < best - 1e-4:
            best, best_state, bad = val_mae, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    print(f"  head seed {seed}: stopped at epoch {epoch + 1}, val MAE {best:.3f} "
          f"(GP alone {np.mean(np.abs(base['val'][0] - data['val'][4].numpy())):.3f})", flush=True)
    return head


def predict_head(head, tensors, base_split) -> np.ndarray:
    """Corrected mean = GP mean + head correction."""
    cat, wide, text, image, _ = tensors
    gm, gs = torch.from_numpy(base_split[0]), torch.from_numpy(base_split[1])
    head.eval()
    with torch.no_grad():
        delta = torch.cat([head(cat[s:s + 4096], wide[s:s + 4096], text[s:s + 4096], image[s:s + 4096],
                                gm[s:s + 4096], gs[s:s + 4096]) for s in range(0, len(gm), 4096)])
    return base_split[0] + delta.numpy()


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(seed=SPLIT_SEED, features_path=FEATURES)
    groups = pd.read_parquet(FEATURES, columns=["__group_oracle_id"])["__group_oracle_id"].to_numpy()
    train_idx, _, _ = _split(groups, SPLIT_SEED)

    print("== out-of-fold GP predictions on the train split ==", flush=True)
    oof_mean, oof_std = oof_train_predictions(data, groups[train_idx])

    ckpt = torch.load(MODEL_OUT_PATH, weights_only=False)
    gp, lik = build_model(data, ckpt["config"])
    gp.load_state_dict(ckpt["model"])
    lik.load_state_dict(ckpt["likelihood"])
    base = {"train": (oof_mean, oof_std),
            "val": _predict(gp, lik, data["val"], 1024), "test": _predict(gp, lik, data["test"], 1024)}
    val_y, test_y = data["val"][4].numpy(), data["test"][4].numpy()
    record("gp_published", "deep_gp", [metrics(*base["val"], val_y, *base["test"], test_y)])

    print("\n== residual head ==", flush=True)
    runs = []
    for seed in HEAD_SEEDS:
        head = train_head(data, base, seed)
        val_corr, test_corr = predict_head(head, data["val"], base["val"]), predict_head(head, data["test"], base["test"])
        runs.append(metrics(val_corr, base["val"][1], val_y, test_corr, base["test"][1], test_y))
    record("gp_resid_head", "deep_gp", runs)
    print_comparison(["gp_published", "gp_resid_head"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
