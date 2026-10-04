"""Deep-GP outputs for every existing printing -- computed once (and whenever the
dataset changes), since they depend only on card attributes + frozen text/art
embeddings, never on prices. The nightly pipeline just joins this file to the day's
prices; recomputing embeddings nightly would need the 3.6 GB art corpus.

  fair_gp, fair_gp_lo, fair_gp_hi   calibrated 80% fair-value range (USD)
  comps                             the 25 nearest other cards in the GP's
                                    embedding (lengthscale-scaled), as printing_keys

Needs the local embeddings/ tables (scripts/embed_text.py, scripts/embed_images.py)
and the exported GP (scripts/export_models.py -> models/deep_gp/).

    python scripts/app/build_gp_static.py   # -> models/app/gp_static.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from mtg_price_prediction.deep_gp import DeepGPPredictor, transform
from mtg_price_prediction.gp_features import gp_feature_frame

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
GP_DIR = ROOT / "models" / "deep_gp"
EMB = ROOT / "embeddings"
OUT = ROOT / "models" / "app" / "gp_static.parquet"
N_COMPS, N_CANDIDATES = 25, 80  # score_all.py shows the top 5 distinct, priced cards


def stored_embeddings(ds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_index = pd.read_parquet(EMB / "text_row_index.parquet").set_index("printing_key")["text_id"]
    text = pd.read_parquet(EMB / "text_embeddings.parquet").set_index("text_id")["embedding"]
    t = np.stack(ds["printing_key"].map(row_index).map(text).to_numpy()).astype(np.float32)
    images = pd.read_parquet(EMB / "image_embeddings.parquet").set_index("illustration_id")["embedding"]
    zero = np.zeros(len(images.iloc[0]), np.float32)
    i = np.stack(ds["illustration_id"].map(lambda iid: images.get(iid, zero) if pd.notna(iid) else zero).to_numpy())
    return t, i.astype(np.float32)


def main() -> int:
    ds = pd.read_parquet(DATASET)
    gp = DeepGPPredictor.from_pretrained(str(GP_DIR))
    feats = gp_feature_frame(ds, gp.prep["wide_cols"])
    text, image = stored_embeddings(ds)
    pred = gp.predict(feats, text, image)

    cat, wide = transform(feats, gp.prep)
    with torch.no_grad():
        z = gp.model.feature_extractor(torch.from_numpy(cat), torch.from_numpy(wide), torch.from_numpy(text),
                                       torch.from_numpy(image)).numpy()
    z = z / gp.model.gp_layer.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    _, idx = NearestNeighbors(n_neighbors=N_CANDIDATES).fit(z).kneighbors(z)
    oracle = ds["oracle_id"].to_numpy()
    keys = ds["printing_key"].to_numpy()
    comps = [[keys[j] for j in row if oracle[j] != oracle[i]][:N_COMPS] for i, row in enumerate(idx)]

    out = pd.DataFrame({"printing_key": keys, "fair_gp": pred["price_mid"].to_numpy(),
                        "fair_gp_lo": pred["price_low"].to_numpy(), "fair_gp_hi": pred["price_high"].to_numpy(),
                        "comps": comps})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out):,} printings -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
