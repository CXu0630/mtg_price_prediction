import json

import numpy as np
import pandas as pd
import pytest

from mtg_price_prediction import calibration
from mtg_price_prediction.deep_gp import CATEGORICAL_COLS, NEEDS_ISNA_FLAG, NUMERIC_TO_SCALE, fit_preprocessor, transform


def test_bucket_edges():
    log_prices = np.log([0.5, 1.0, 9.99, 10.0, 99.0, 100.0, 5000.0])
    assert calibration.bucket_of(log_prices).tolist() == [0, 1, 1, 2, 2, 3, 3]


def test_corrections_reach_nominal_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    median = np.zeros_like(y)  # all in bucket 1
    lo, hi = median - 0.5, median + 0.5  # deliberately too narrow
    corrections = calibration.fit_corrections(median, lo, hi, y)
    lo_cal, hi_cal = calibration.apply_corrections(median, lo, hi, corrections)
    coverage = np.mean((y >= lo_cal) & (y <= hi_cal))
    assert coverage == pytest.approx(0.8, abs=0.02)


def _toy_features(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.choice(["a", "b", None], n) for c in CATEGORICAL_COLS})
    for c in NUMERIC_TO_SCALE:
        df[c] = np.where(rng.random(n) < 0.2, np.nan, rng.normal(5, 2, n))
    df["play_rate_percent"] = rng.random(n)
    df["keywords__flying"] = rng.random(n) < 0.3
    df["__target_log_price_avg"] = rng.normal(0, 1, n)
    return df


def test_preprocessor_roundtrips_through_json_and_handles_unseen_rows():
    df = _toy_features(200)
    prep = json.loads(json.dumps(fit_preprocessor(df, np.arange(150))))

    cat_idx, wide = transform(df, prep)
    assert cat_idx.shape == (200, len(CATEGORICAL_COLS))
    assert wide.shape == (200, len(prep["wide_cols"]))
    assert not np.isnan(wide).any()
    assert all(f"{c}__isna" in prep["wide_cols"] for c in NEEDS_ISNA_FLAG)
    assert not any(c.startswith("__") for c in prep["wide_cols"])

    unseen = df.head(1).copy()
    unseen["artist"] = "someone new"
    assert transform(unseen, prep)[0][0, CATEGORICAL_COLS.index("artist")] == 0  # __unk__


def test_transform_rejects_missing_columns():
    df = _toy_features(50)
    prep = fit_preprocessor(df, np.arange(50))
    with pytest.raises(KeyError):
        transform(df.drop(columns=["conv_mana_cost"]), prep)
