"""Packages the best Model 1 and Model 2 into self-contained, pickle-free
directories under models/ -- the exact layout published to Hugging Face by
scripts/push_models_to_hub.py and loaded by
`mtg_price_prediction.{CatBoostPredictor,DeepGPPredictor}.from_pretrained`.

    models/catboost/  model.cbm, config.json, README.md
    models/deep_gp/   model.safetensors, config.json, preprocessor.json,
                      pull_rate_defaults.parquet, README.md

Each also gets a copy of the inference code it needs
(mtg_price_prediction/), so the published repo is usable without this one.

Inputs (run these first):
  - training/model1.cbm + training/model1_calibration.json
      <- scripts/calibrate_model1.py
  - training/model2.pt  <- scripts/train_model2.py (BEST_CONFIG)

The Model 2 calibration corrections are refit here on the val split (same
flat group-conditional method as scripts/calibrate_model2.py), and the
exported directory is reloaded through DeepGPPredictor and checked against
the training-code predictions before anything is written as final.
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from train_model2 import MODEL_OUT_PATH as MODEL2_PATH, Config, build_model, load_data, _predict, _split
from mtg_price_prediction import calibration
from mtg_price_prediction.deep_gp import Z_80, DeepGPPredictor
from mtg_price_prediction.gp_features import MODEL_A_COLS, gp_feature_frame

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "models"
MODEL_CARDS_DIR = ROOT / "model_cards"
MODEL1_PATH = ROOT / "training" / "model1.cbm"
MODEL1_CALIBRATION_PATH = ROOT / "training" / "model1_calibration.json"
PULL_RATE_DEFAULTS_PATH = ROOT / "training" / "pull_rate_defaults.parquet"
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
MODEL_A_AS_OF = pd.Timestamp("2026-09-20")  # end of the price_avg window; age features are measured from here
PACKAGE_DIR = ROOT / "src" / "mtg_price_prediction"


def _bundle_code(out: Path, modules: list[str]) -> None:
    pkg = out / "mtg_price_prediction"
    pkg.mkdir(exist_ok=True)
    for name in ["__init__.py", "calibration.py", *modules]:
        shutil.copy(PACKAGE_DIR / name, pkg / name)


def export_catboost() -> None:
    out = OUT_DIR / "catboost"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL1_PATH, out / "model.cbm")
    config = json.loads(MODEL1_CALIBRATION_PATH.read_text())
    config = {"model_type": "catboost_multiquantile", "target": "log(price_avg_usd)",
              "price_buckets_log": calibration.PRICE_BUCKET_LABELS, **config}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copy(MODEL_CARDS_DIR / "catboost.md", out / "README.md")
    _bundle_code(out, ["catboost_model.py"])
    print(f"catboost -> {out}  (test overall: {config['test_metrics']['overall']})")


def _bucket_metrics(mean, lo, hi, y) -> dict:
    metrics = {}
    true_bucket = calibration.bucket_of(y)
    for b, label in enumerate(calibration.PRICE_BUCKET_LABELS):
        m = true_bucket == b
        metrics[label] = {"n": int(m.sum()), "mae": float(np.mean(np.abs(mean[m] - y[m]))),
                          "coverage_calibrated": float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))),
                          "width_calibrated": float(np.mean(hi[m] - lo[m]))}
    metrics["overall"] = {"n": len(y), "mae": float(np.mean(np.abs(mean - y))),
                          "coverage_calibrated": float(np.mean((y >= lo) & (y <= hi))),
                          "width_calibrated": float(np.mean(hi - lo))}
    return metrics


def export_deep_gp() -> None:
    checkpoint = torch.load(MODEL2_PATH, weights_only=False)
    config: Config = checkpoint["config"]
    assert config.likelihood_type == "gaussian", "DeepGPPredictor only supports the Gaussian likelihood"
    assert not (config.spectral_norm or config.residual_spectral_norm), "export assumes the plain MLP extractor"

    data = load_data(seed=config.seed, features_path=config.features_path)
    model, likelihood = build_model(data, config)
    model.load_state_dict(checkpoint["model"])
    likelihood.load_state_dict(checkpoint["likelihood"])

    val_mean, val_std = _predict(model, likelihood, data["val"], config.batch_size)
    val_y = data["val"][4].numpy()
    corrections = calibration.fit_corrections(val_mean, val_mean - Z_80 * val_std, val_mean + Z_80 * val_std, val_y)

    test_mean, test_std = _predict(model, likelihood, data["test"], config.batch_size)
    test_y = data["test"][4].numpy()
    lo, hi = calibration.apply_corrections(test_mean, test_mean - Z_80 * test_std, test_mean + Z_80 * test_std,
                                           corrections)
    metrics = _bucket_metrics(test_mean, lo, hi, test_y)

    out = OUT_DIR / "deep_gp"
    out.mkdir(parents=True, exist_ok=True)
    state = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    state |= {f"likelihood.{k}": v.contiguous() for k, v in likelihood.state_dict().items()}
    save_file(state, out / "model.safetensors")
    (out / "preprocessor.json").write_text(json.dumps(data["prep"]))
    training_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()
                       if k not in ("checkpoint_path", "features_path")}
    (out / "config.json").write_text(json.dumps({
        "model_type": "deep_kernel_svgp",
        "target": "log(price_avg_usd)",
        "architecture": {"joint_dim": config.joint_dim, "num_inducing": config.num_inducing,
                         "text_dim": data["text_dim"], "image_dim": data["image_dim"],
                         "use_text": config.use_text, "use_image": config.use_image,
                         "use_tabular": config.use_tabular,
                         "text_encoder": "sentence-transformers/all-mpnet-base-v2",
                         "image_encoder": "open_clip ViT-B-32-quickgelu (pretrained=openai), Scryfall art_crop"},
        "training_config": training_config,
        "price_buckets_log": calibration.PRICE_BUCKET_LABELS,
        "calibration_corrections": {str(b): c for b, c in corrections.items()},
        "test_metrics": metrics,
    }, indent=2))
    model_a_inputs = {}
    if any(c in data["prep"]["wide_cols"] for c in MODEL_A_COLS):
        # Set price level is built from the training split's prices only; ship that table so serving
        # computes it the same way (gp_features.model_a_features).
        rows = pd.read_parquet(config.features_path, columns=["__printing_key", "__group_oracle_id",
                                                             "__target_log_price_avg"])
        train_idx, _, _ = _split(rows["__group_oracle_id"].to_numpy(), config.seed)
        ds = pd.read_parquet(DATASET_PATH, columns=["printing_key", "set", "oracle_id"]).set_index("printing_key")
        tr = rows.iloc[train_idx]
        card_lp = (ds.loc[tr["__printing_key"].to_numpy()].reset_index(drop=True)
                   .assign(lp=tr["__target_log_price_avg"].to_numpy())
                   .groupby(["set", "oracle_id"], as_index=False)["lp"].max())
        card_lp.to_parquet(out / "set_prices.parquet", index=False)
        model_a_inputs = {"model_a_inputs": {"as_of": str(MODEL_A_AS_OF.date()), "set_prices": "set_prices.parquet"}}
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps({**json.loads(cfg_path.read_text()), **model_a_inputs}, indent=2))
    shutil.copy(PULL_RATE_DEFAULTS_PATH, out / "pull_rate_defaults.parquet")
    shutil.copy(MODEL_CARDS_DIR / "deep_gp.md", out / "README.md")
    _bundle_code(out, ["deep_gp.py", "gp_features.py"])

    # Parity check: reload through the public inference path on the test rows.
    test_rows = pd.read_parquet(config.features_path)
    _, _, test_idx = _split(test_rows["__group_oracle_id"].to_numpy(), config.seed)
    predictor = DeepGPPredictor.from_pretrained(str(out))
    reloaded = predictor.predict(test_rows.iloc[test_idx], data["test"][2].numpy(), data["test"][3].numpy())
    max_diff = float(np.max(np.abs(np.log(reloaded["price_mid"].to_numpy()) - test_mean)))
    assert max_diff < 1e-3, f"exported deep GP disagrees with training code (max |diff| = {max_diff})"
    if model_a_inputs:
        # Serving path: rebuild the test rows' features from raw dataset rows, as build_gp_static.py does.
        dataset = pd.read_parquet(DATASET_PATH)
        raw = dataset.set_index("printing_key").loc[test_rows.iloc[test_idx]["__printing_key"].to_numpy()].reset_index()
        feats = gp_feature_frame(raw, predictor.prep["wide_cols"], predictor.feature_context(dataset))
        served = predictor.predict(feats, data["test"][2].numpy(), data["test"][3].numpy())
        serve_diff = float(np.max(np.abs(np.log(served["price_mid"].to_numpy()) - test_mean)))
        assert serve_diff < 1e-3, f"serving features disagree with the training table (max |diff| = {serve_diff})"
        print(f"  serving-path parity max |diff| = {serve_diff:.2e}")
    print(f"deep_gp -> {out}  (parity max |diff| = {max_diff:.2e}; test: {metrics})")


def main(which: list[str]) -> int:
    if "catboost" in which:
        export_catboost()
    if "deep_gp" in which:
        export_deep_gp()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["catboost", "deep_gp"]))
