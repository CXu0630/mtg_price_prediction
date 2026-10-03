"""Ablations for the quantile-head Model 2 variant, same three variants
and discipline as scripts/check_model2_ablation_by_bucket.py (the GP
version): full (text+image+tabular), no_image, no_text, tabular_only --
trained on identical data/hyperparameters, compared bucketed rather than
by aggregate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for the shared train_model*/calibrate_* modules

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_model2_quantile import QuantileConfig, build_model, load_data, run, _predict

ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = ROOT / "training" / "ablation_checkpoints_quantile"

ABLATIONS = {
    "full": {"use_text": True, "use_image": True, "use_tabular": True},
    "no_image": {"use_text": True, "use_image": False, "use_tabular": True},
    "no_text": {"use_text": False, "use_image": True, "use_tabular": True},
    "tabular_only": {"use_text": False, "use_image": False, "use_tabular": True},
}

PRICE_BUCKETS = [(-np.inf, 0), (0, np.log(10)), (np.log(10), np.log(100)), (np.log(100), np.inf)]
PRICE_BUCKET_LABELS = ["<$1", "$1-$10", "$10-$100", "$100+"]
SEED = 0


def _bucketed_metrics(model, data, label: str) -> list[dict]:
    preds = _predict(model, data["test"], 1024)
    y = data["test"][4].numpy()
    q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]

    rows = []
    for (lo, hi), blabel in zip(PRICE_BUCKETS, PRICE_BUCKET_LABELS):
        mask = (y >= lo) & (y < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "variant": label, "bucket": blabel, "n": int(mask.sum()),
            "coverage_pct": float(np.mean((y[mask] >= q10[mask]) & (y[mask] <= q90[mask])) * 100),
            "mae_log": float(np.median(np.abs(q50[mask] - y[mask]))),  # median: robust to the rare runaway prediction
        })
    overall_cov = float(np.mean((y >= q10) & (y <= q90)) * 100)
    rows.append({"variant": label, "bucket": "OVERALL", "n": len(y), "coverage_pct": overall_cov,
                 "mae_log": float(np.median(np.abs(q50 - y)))})
    return rows


def main() -> int:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data(seed=SEED)

    all_rows = []
    for name, flags in ABLATIONS.items():
        config = replace(QuantileConfig(), oversample_high_price=True, **flags,
                          label=name, checkpoint_path=ABLATION_DIR / f"{name}.pt")
        print(f"\n=== training ablation: {name} ({flags}) ===")
        run(data, config, verbose=True)

        model = build_model(data, config)
        checkpoint = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        all_rows.extend(_bucketed_metrics(model, data, name))

    result = pd.DataFrame(all_rows)
    pivot_cov = result.pivot(index="bucket", columns="variant", values="coverage_pct")
    pivot_mae = result.pivot(index="bucket", columns="variant", values="mae_log")
    order = PRICE_BUCKET_LABELS + ["OVERALL"]
    print("\n=== coverage_pct by bucket x ablation (raw, uncalibrated) ===")
    print(pivot_cov.reindex(order)[list(ABLATIONS)].to_string())
    print("\n=== median mae_log by bucket x ablation ===")
    print(pivot_mae.reindex(order)[list(ABLATIONS)].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
