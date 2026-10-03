"""Downloads the published, already-joined training dataset
(CXu0630/mtg-card-market-dataset on HF) to mtg-card-market-dataset/dataset.parquet.

This is the quick start: it replaces fetch_mtg_card_prices.py ->
build_merged.py -> join_pull_rates.py -> join_price.py, which you only need
when rebuilding the dataset from its sources (the pull-rate join also needs
a local checkout of the separate mtg-print-distribution repo).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "CXu0630/mtg-card-market-dataset"
OUT_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"


def main() -> int:
    cached = hf_hub_download(REPO_ID, "dataset.parquet", repo_type="dataset")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(cached, OUT_PATH)
    print(f"{REPO_ID} -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
