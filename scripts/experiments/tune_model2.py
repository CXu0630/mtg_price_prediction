"""Randomized hyperparameter search for Model 2, same spirit as
scripts/tune_model1.py: score candidates by pinball loss weighted toward
the $10-100/$100+ buckets specifically (those are the whole reason Model 2
exists -- see SPECS.md S4.2's first-iteration results), not by an aggregate
metric that's dominated by the easy <$1 bucket.

Loads the data once (load_data() is architecture/hyperparameter-agnostic)
and reuses it across every candidate, since embedding joins + train-split
standardization are identical regardless of what's being tuned.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from train_model2 import Config, build_model, load_data, run, _pinball_loss, _predict

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_DIR = ROOT / "training" / "tune_candidates"

SEARCH_SPACE = {
    "joint_dim": [32, 48, 64, 96],
    "num_inducing": [500, 750, 1000],
    "lr": [1e-3, 3e-3, 1e-2],
}
N_CANDIDATES = 8
SEED = 0

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
BUCKET_WEIGHTS = [0.5, 1.0, 2.5, 2.5]  # matches tune_model1.py's weighting toward $10-100/$100+


def _weighted_score(model, likelihood, data) -> float:
    mean, std = _predict(model, likelihood, data["test"], 1024)
    y = data["test"][4].numpy()
    total, weight_sum = 0.0, 0.0
    for (lo, hi), w in zip(PRICE_BUCKETS, BUCKET_WEIGHTS):
        mask = (y >= lo) & (y < hi)
        if mask.sum() == 0:
            continue
        pinball = np.mean([
            _pinball_loss(y[mask], mean[mask] + std[mask] * z, q)
            for q, z in [(0.1, -1.2816), (0.5, 0.0), (0.9, 1.2816)]
        ])
        total += w * pinball
        weight_sum += w
    return total / weight_sum


def main() -> int:
    rng = random.Random(SEED)
    data = load_data(seed=SEED)
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(N_CANDIDATES):
        params = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        label = f"tune{i}_" + "_".join(f"{k}={v}" for k, v in params.items())
        config = replace(Config(), **params, label=label, checkpoint_path=CANDIDATES_DIR / f"{label}.pt")
        print(f"\n=== candidate {i+1}/{N_CANDIDATES}: {params} ===")

        metrics = run(data, config, verbose=True)

        model, likelihood = build_model(data, config)
        checkpoint = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        likelihood.load_state_dict(checkpoint["likelihood"])
        weighted_pinball = _weighted_score(model, likelihood, data)

        results.append({**params, "label": label, **metrics, "weighted_pinball": weighted_pinball})
        print(f"  -> weighted_pinball (favoring $10-100/$100+) = {weighted_pinball:.4f}")

    results.sort(key=lambda r: r["weighted_pinball"])
    print("\n=== ranked by weighted pinball (lower is better) ===")
    for r in results:
        print(f"  {r['weighted_pinball']:.4f}  {r['label']}  (test_coverage80={r['test_coverage80']*100:.1f}%, "
              f"test_mae={r['test_mae']:.4f})")

    best = results[0]
    print(f"\nbest candidate: {best['label']} -> {CANDIDATES_DIR / (best['label'] + '.pt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
