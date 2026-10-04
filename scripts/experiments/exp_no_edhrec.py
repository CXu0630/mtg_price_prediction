"""Experiment 1: how much do EDHREC rank + salt carry? Retrain both models
without them (permutation importance said EDHREC rank is the #1 driver for
both) -- this measures what *retraining* without it costs, which permutation
importance can't (other features partly compensate after a retrain).

    python scripts/experiments/exp_no_edhrec.py     # ~5 min CatBoost + ~6 GP runs
"""
from __future__ import annotations

import json

import pandas as pd

from _gp_common import EXP_DIR, RESULTS_PATH, ROOT, print_comparison, record, train_variant
import calibrate_model1

DROP = ("edhrec_rank", "edhrec_salt")
GP_FEATURES = ROOT / "training" / "features_model2.parquet"


def _catboost_record(label: str, test_metrics: dict) -> None:
    rename = {"mae_p50": "mae", "coverage_calibrated": "coverage", "width_calibrated": "width", "n": "n"}
    runs = [{b: {rename[k]: v for k, v in m.items() if k in rename} for b, m in test_metrics.items()}]
    record(label, "catboost", runs)


def baseline_gp_done() -> bool:
    return RESULTS_PATH.exists() and any(json.loads(l)["label"] == "gp_baseline" for l in RESULTS_PATH.read_text().splitlines())


def main() -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    print("== CatBoost ==", flush=True)
    best = json.loads((ROOT / "training" / "model1_calibration.json").read_text())["test_metrics"]
    _catboost_record("cb_baseline", best)
    _catboost_record("cb_no_edhrec", calibrate_model1.main(drop_features=DROP, save=False))

    print("\n== Deep GP ==", flush=True)
    if not baseline_gp_done():
        train_variant("gp_baseline", GP_FEATURES)
    df = pd.read_parquet(GP_FEATURES)
    no_edhrec = EXP_DIR / "features_model2_no_edhrec.parquet"
    df.drop(columns=[c for c in df.columns if c.split("__")[0] in DROP]).to_parquet(no_edhrec, index=False)
    train_variant("gp_no_edhrec", no_edhrec)

    print_comparison(["cb_baseline", "cb_no_edhrec", "gp_baseline", "gp_no_edhrec"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
