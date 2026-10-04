"""Shared helpers for the GP experiments (exp_*.py): train a BEST_CONFIG variant
over several init seeds on the fixed seed-0 split, and score each the same way
the published model is scored -- flat group-conditional conformal correction
fit on val, then MAE / calibrated 80% coverage / width per TRUE price bucket on
test. Checkpoints and results live in training/experiments/ (gitignored)."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for train_model2

from mtg_price_prediction import calibration  # noqa: E402
from mtg_price_prediction.deep_gp import Z_80  # noqa: E402
from train_model2 import BEST_CONFIG, ROOT, build_model, load_data, run, _predict  # noqa: E402

EXP_DIR = ROOT / "training" / "experiments"
RESULTS_PATH = EXP_DIR / "results.jsonl"
SPLIT_SEED = 0
SEEDS = (0, 1, 2)


def metrics(val_mean, val_std, val_y, test_mean, test_std, test_y) -> dict:
    """Flat per-bucket conformal correction fit on val; per-TRUE-bucket test metrics."""
    corr = calibration.fit_corrections(val_mean, val_mean - Z_80 * val_std, val_mean + Z_80 * val_std, val_y)
    lo, hi = calibration.apply_corrections(test_mean, test_mean - Z_80 * test_std, test_mean + Z_80 * test_std, corr)
    out = {}
    true_bucket = calibration.bucket_of(test_y)
    for b, label in enumerate(calibration.PRICE_BUCKET_LABELS):
        m = true_bucket == b
        out[label] = {"mae": float(np.mean(np.abs(test_mean[m] - test_y[m]))),
                      "coverage": float(np.mean((test_y[m] >= lo[m]) & (test_y[m] <= hi[m]))),
                      "width": float(np.mean(hi[m] - lo[m])), "n": int(m.sum())}
    out["overall"] = {"mae": float(np.mean(np.abs(test_mean - test_y))),
                      "coverage": float(np.mean((test_y >= lo) & (test_y <= hi))),
                      "width": float(np.mean(hi - lo)), "n": len(test_y)}
    return out


def evaluate_checkpoint(data: dict, config, checkpoint_path: Path) -> dict:
    model, lik = build_model(data, config)
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model"])
    lik.load_state_dict(ckpt["likelihood"])
    val_mean, val_std = _predict(model, lik, data["val"], config.batch_size)
    test_mean, test_std = _predict(model, lik, data["test"], config.batch_size)
    return metrics(val_mean, val_std, data["val"][4].numpy(), test_mean, test_std, data["test"][4].numpy())


def train_variant(label: str, features_path: Path, seeds=SEEDS, **overrides) -> list[dict]:
    """BEST_CONFIG + overrides, one run per init seed, all on the seed-0 split."""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(seed=SPLIT_SEED, features_path=features_path)
    results = []
    for seed in seeds:
        ckpt = EXP_DIR / f"{label}_s{seed}.pt"
        config = replace(BEST_CONFIG, seed=seed, label=f"{label}_s{seed}", features_path=features_path,
                         checkpoint_path=ckpt, **overrides)
        run(data, config, verbose=False)
        m = evaluate_checkpoint(data, config, ckpt)
        print(f"  [{label} seed {seed}] MAE {m['overall']['mae']:.3f} | cov $10-100 {m['$10-$100']['coverage']:.3f}"
              f" $100+ {m['$100+']['coverage']:.3f} overall {m['overall']['coverage']:.3f}", flush=True)
        results.append(m)
    record(label, "deep_gp", results)
    return results


def record(label: str, model: str, results: list[dict]) -> None:
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps({"label": label, "model": model, "runs": results}) + "\n")


def summary_table(labels: list[str] | None = None) -> pd.DataFrame:
    """mean ± std over seeds, latest record per label."""
    latest = {}
    for line in RESULTS_PATH.read_text().splitlines():
        r = json.loads(line)
        latest[r["label"]] = r
    rows = []
    for label, r in latest.items():
        if labels and label not in labels:
            continue
        for bucket in [*calibration.PRICE_BUCKET_LABELS, "overall"]:
            for metric in ("mae", "coverage", "width"):
                vals = [run_[bucket][metric] for run_ in r["runs"]]
                rows.append({"label": label, "model": r["model"], "bucket": bucket, "metric": metric,
                             "mean": np.mean(vals), "std": np.std(vals), "n_runs": len(vals)})
    return pd.DataFrame(rows)


def print_comparison(labels: list[str]) -> None:
    t = summary_table(labels)
    t["cell"] = t.apply(lambda r: f"{r['mean']:.3f}" + (f"±{r['std']:.3f}" if r["n_runs"] > 1 else ""), axis=1)
    for metric in ("mae", "coverage"):
        piv = t[t.metric == metric].pivot(index="label", columns="bucket", values="cell")
        print(f"\n--- {metric} (test; mean±std over seeds) ---")
        print(piv[[*calibration.PRICE_BUCKET_LABELS, "overall"]].reindex(labels).to_string())
