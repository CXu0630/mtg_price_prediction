"""Model 1 -- CatBoost MultiQuantile (p10/p50/p90) + per-bucket conformal
calibration (SPECS.md S4.1). Inference wrapper for the exported model
(scripts/export_models.py layout: model.cbm + config.json), which is also
what the published Hugging Face model loads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtg_price_prediction import calibration


class CatBoostPredictor:
    def __init__(self, model, config: dict):
        self.model, self.config = model, config
        self.corrections = {int(k): v for k, v in config["calibration_corrections"].items()}

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str) -> "CatBoostPredictor":
        from catboost import CatBoostRegressor

        path = Path(repo_id_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id_or_path))
        model = CatBoostRegressor()
        model.load_model(str(path / "model.cbm"))
        return cls(model, json.loads((path / "config.json").read_text()))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """`features`: rows shaped like training/features.parquet
        (scripts/prepare_training_data.py); extra columns are ignored.
        Returns USD price_low / price_mid / price_high (calibrated 80%
        interval + median)."""
        X = features[self.config["feature_cols"]].copy()
        for c in self.config["cat_features"]:
            X[c] = X[c].fillna("__missing__").astype(str)
        preds = self.model.predict(X)
        p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]
        lo, hi = calibration.apply_corrections(p50, p10, p90, self.corrections)
        return pd.DataFrame({"price_low": np.exp(lo), "price_mid": np.exp(p50), "price_high": np.exp(hi)},
                            index=features.index)
