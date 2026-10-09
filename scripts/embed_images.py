"""CLIP image embeddings for the deep GP, over the art_crop
images fetched by scripts/fetch_card_art.py.

Model: CLIP ViT-B/32 (OpenAI's original checkpoint, via open_clip) --
a standard, well-tested default; frozen, no fine-tuning. One embedding per distinct `illustration_id`,
matching how the images themselves were fetched (dedup, not one per row).
"""
from __future__ import annotations

from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "card_art" / "manifest.parquet"
OUT_DIR = ROOT / "embeddings"

MODEL_NAME = "ViT-B-32-quickgelu"  # OpenAI's original checkpoint used QuickGELU,
PRETRAINED = "openai"               # not the newer default GELU -- must match exactly
BATCH_SIZE = 64


def main() -> int:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    manifest = pd.read_parquet(MANIFEST_PATH)
    manifest = manifest[manifest["status"].isin(["downloaded", "cached"])].reset_index(drop=True)
    print(f"embedding {len(manifest)} images on {device}")

    all_embeddings = []
    ids = manifest["illustration_id"].tolist()
    paths = manifest["local_path"].tolist()

    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            imgs = torch.stack([preprocess(Image.open(ROOT / p).convert("RGB")) for p in batch_paths]).to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())
            if (start // BATCH_SIZE) % 50 == 0:
                print(f"  {min(start + BATCH_SIZE, len(paths))}/{len(paths)}")

    import numpy as np
    embeddings = np.concatenate(all_embeddings, axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"illustration_id": ids})
    out["embedding"] = list(embeddings)
    out.to_parquet(OUT_DIR / "image_embeddings.parquet", index=False)

    print(f"model: {MODEL_NAME}/{PRETRAINED}, dim: {embeddings.shape[1]}")
    print(f"-> {OUT_DIR / 'image_embeddings.parquet'} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
