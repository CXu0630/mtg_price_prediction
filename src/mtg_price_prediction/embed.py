"""Text and art embeddings for cards the precomputed tables don't cover (custom cards),
matching scripts/embed_text.py and scripts/embed_images.py exactly:
  text  sentence-transformers/all-mpnet-base-v2 (768-d) over oracle text
  image CLIP ViT-B-32-quickgelu, pretrained=openai (512-d) over the art crop
Models load lazily on first use (a cold start of ~30-60 s on CPU)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
IMAGE_ARCH, IMAGE_PRETRAINED = "ViT-B-32-quickgelu", "openai"
TEXT_DIM, IMAGE_DIM = 768, 512


@lru_cache(maxsize=1)
def _text_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(TEXT_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _image_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(IMAGE_ARCH, pretrained=IMAGE_PRETRAINED, device="cpu")
    return model.eval(), preprocess


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, TEXT_DIM), np.float32)
    return _text_model().encode([t or "" for t in texts], batch_size=32, show_progress_bar=False,
                                normalize_embeddings=True).astype(np.float32)  # as scripts/embed_text.py


def embed_images(images: list) -> np.ndarray:
    """`images`: PIL images, or None for "no art" (a zero vector, as in training)."""
    import torch

    out = np.zeros((len(images), IMAGE_DIM), np.float32)
    present = [i for i, im in enumerate(images) if im is not None]
    if present:
        model, preprocess = _image_model()
        with torch.no_grad():
            batch = torch.stack([preprocess(images[i].convert("RGB")) for i in present])
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out[present] = feats.numpy()
    return out
