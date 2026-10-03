"""Pulls pcwoods/mtg-card-prices (HF dataset) down to mtg-card-prices/raw/.

Uses the HF datasets-server /parquet endpoint to resolve the current parquet
file URLs rather than hardcoding them, so a re-run picks up upstream updates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

DATASET = "pcwoods/mtg-card-prices"
OUT_DIR = Path(__file__).resolve().parent.parent / "mtg-card-prices" / "raw"


def main() -> int:
    info = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": DATASET},
        timeout=30,
    )
    info.raise_for_status()
    files = info.json()["parquet_files"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        out_path = OUT_DIR / f"{f['config']}.parquet"
        print(f"{f['config']}: {f['size'] / 1e6:.1f} MB -> {out_path}", file=sys.stderr)
        with requests.get(f["url"], stream=True, timeout=180) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
