"""Grouped permutation importance for the two best models (training/model1.cbm,
training/model2.pt), each on its own held-out test split: shuffle one feature
group across rows (same permutation for every column in the group) and
measure the rise in log-price MAE (CatBoost p50 / GP posterior mean vs y),
overall and on $10+ rows. 3 repeats, averaged.

Measures what the *trained* model relies on, not what retraining without a
group would cost -- correlated groups (rarity <-> pull rates) split credit.

    python scripts/experiments/feature_importance.py   # -> figures/data/feature_importance.csv (~5 min)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))  # for train_model2
OUT_PATH = REPO / "figures" / "data" / "feature_importance.csv"
N_REPEATS = 3
TEN = np.log(10)


def group_of(col: str) -> str:
    base = re.sub(r"__isna$", "", col)
    if base in ("edhrec_rank", "edhrec_salt", "penny_rank", "num_paper_printings", "conv_mana_cost",
                "rarity", "finish", "set_type", "frame", "border_color", "layout", "lang", "watermark",
                "security_stamp", "artist", "has_image"):
        return base
    if base.endswith("_rate_percent"):
        return "pull rates"
    if base.startswith("legal_"):
        return "legalities"
    if base.startswith(("color_", "produced_mana_")):
        return "colors / color identity"
    if base.startswith("can_lead_"):
        return "commander flags"
    if re.match(r"(power|toughness|loyalty|defense)", base):
        return "P/T / loyalty / defense"
    for tag in ("oracle_tags", "art_tags", "type_tags", "keywords", "promo_types", "frame_effects"):
        if base.startswith(tag):
            return tag
    return "card flags (reserved, promo, reprint, borderless, ...)"


def mae_pair(pred, y):
    hi = y >= TEN
    return np.mean(np.abs(pred - y)), np.mean(np.abs(pred[hi] - y[hi]))


def run(name, groups: dict, predict, y, n_rows):
    base_all, base_hi = mae_pair(predict({}), y)
    rng = np.random.default_rng(0)
    rows = []
    for g, members in groups.items():
        d_all, d_hi = [], []
        for _ in range(N_REPEATS):
            perm = rng.permutation(n_rows)
            a, h = mae_pair(predict({m: perm for m in members}), y)
            d_all.append(a - base_all)
            d_hi.append(h - base_hi)
        rows.append((g, np.mean(d_all), np.mean(d_hi)))
    out = pd.DataFrame(rows, columns=["group", "dMAE_all", "dMAE_$10+"]).sort_values("dMAE_all", ascending=False)
    out["% of base (all)"] = (100 * out["dMAE_all"] / base_all).round(1)
    out["% of base ($10+)"] = (100 * out["dMAE_$10+"] / base_hi).round(1)
    print(f"\n=== {name} -- base MAE all {base_all:.3f}, $10+ {base_hi:.3f} (n={n_rows}, $10+ n={(y >= TEN).sum()}) ===")
    print(out.round(4).to_string(index=False))
    return out


# --- Model 1: CatBoost ---
f = pd.read_parquet(REPO / "training/features.parquet")
cfg = json.loads((REPO / "training/model1_calibration.json").read_text())
_, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(f, groups=f["__group_oracle_id"]))
X = f.iloc[test_idx][cfg["feature_cols"]].reset_index(drop=True)
for c in cfg["cat_features"]:
    X[c] = X[c].fillna("__missing__").astype(str)
y1 = f["__target_log_price_avg"].iloc[test_idx].to_numpy()
cb = CatBoostRegressor()
cb.load_model(str(REPO / "training/model1.cbm"))
groups1 = {}
for c in cfg["feature_cols"]:
    groups1.setdefault(group_of(c), []).append(c)


def predict1(perms):
    Xp = X.copy()
    for col, perm in perms.items():
        Xp[col] = X[col].to_numpy()[perm]
    return cb.predict(Xp)[:, 1]


imp1 = run("Model 1: CatBoost", groups1, predict1, y1, len(X))

# --- Model 2: deep GP (permute post-preprocessing tensors) ---
from train_model2 import load_data, build_model, MODEL_OUT_PATH, CATEGORICAL_COLS  # noqa: E402
from mtg_price_prediction.deep_gp import predict_moments  # noqa: E402

ckpt = torch.load(MODEL_OUT_PATH, weights_only=False)
cfg2 = ckpt["config"]
data = load_data(seed=cfg2.seed, features_path=cfg2.features_path)
model, lik = build_model(data, cfg2)
model.load_state_dict(ckpt["model"])
lik.load_state_dict(ckpt["likelihood"])
cat, wide, text, image, yt = data["test"]
y2 = yt.numpy()
wide_cols = data["prep"]["wide_cols"]
groups2 = {"text embedding": [("text", None)], "image embedding": [("image", None)]}
for i, c in enumerate(CATEGORICAL_COLS):
    groups2.setdefault(group_of(c), []).append(("cat", i))
for i, c in enumerate(wide_cols):
    groups2.setdefault(group_of(c), []).append(("wide", i))


def predict2(perms):
    cat_p, wide_p, text_p, image_p = cat.clone(), wide.clone(), text, image
    for (kind, i), perm in perms.items():
        p = torch.from_numpy(perm)
        if kind == "cat":
            cat_p[:, i] = cat[p, i]
        elif kind == "wide":
            wide_p[:, i] = wide[p, i]
        elif kind == "text":
            text_p = text[p]
        else:
            image_p = image[p]
    mean, _ = predict_moments(model, lik, cat_p, wide_p, text_p, image_p, 2048)
    return mean


imp2 = run("Model 2: deep GP", groups2, predict2, y2, len(y2))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
pd.concat([imp1.assign(model="CatBoost"), imp2.assign(model="Deep GP")]).to_csv(OUT_PATH, index=False)
print(f"\n-> {OUT_PATH}")
