"""Fetches card art (Scryfall's `art_crop` -- illustration only, no border/
name/rules text, see conversation for why) for every distinct
`illustration_id` in mtg-card-market-dataset -- not one download per row,
since many printings reuse identical art (confirmed earlier: e.g. all 953
Forest printings share far fewer distinct illustrations).

URL source: Scryfall's `unique_artwork` bulk data file (one JSON object per
distinct illustration, refreshed via the bulk-data API rather than a
hardcoded URL, since these are timestamped/rotate) -- not one API call per
card, which would mean ~47k individual requests for metadata alone. 100%
of our illustration_ids were found covered when checked against it.

Output: card_art/images/{illustration_id}.jpg, plus card_art/manifest.parquet
(illustration_id, art_crop_url, local_path, status) recording what
happened for every one, so gaps are visible rather than silently missing.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
CARD_ART_DIR = ROOT / "card_art"
IMAGES_DIR = CARD_ART_DIR / "images"
MANIFEST_PATH = CARD_ART_DIR / "manifest.parquet"
BULK_JSONL_GZ = CARD_ART_DIR / "unique_artwork.jsonl.gz"

USER_AGENT = "mtg-price-prediction-research (contact: guangyu.xu0630@gmail.com)"
N_WORKERS = 8


def _fetch_bulk_data_uri() -> str:
    resp = requests.get("https://api.scryfall.com/bulk-data", headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    for item in resp.json()["data"]:
        if item["type"] == "unique_artwork":
            return item["jsonl_download_uri"]
    raise RuntimeError("unique_artwork bulk data entry not found")


def _download_bulk_jsonl(refresh: bool = False) -> None:
    if BULK_JSONL_GZ.exists() and not refresh:
        return
    uri = _fetch_bulk_data_uri()
    print(f"downloading {uri}", file=sys.stderr)
    resp = requests.get(uri, headers={"User-Agent": USER_AGENT}, timeout=180, stream=True)
    resp.raise_for_status()
    CARD_ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(BULK_JSONL_GZ, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)


def _illustration_to_art_crop_url() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(BULK_JSONL_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            iid = d.get("illustration_id")
            url = (d.get("image_uris") or {}).get("art_crop")
            if iid and url and iid not in mapping:
                mapping[iid] = url
    return mapping


def _download_one(illustration_id: str, url: str) -> tuple[str, str, str]:
    dest = IMAGES_DIR / f"{illustration_id}.jpg"
    if dest.exists():
        return illustration_id, str(dest.relative_to(ROOT)), "cached"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 404:
            return illustration_id, "", "not_found"
        resp.raise_for_status()
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".jpg.tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return illustration_id, str(dest.relative_to(ROOT)), "downloaded"
    except requests.exceptions.RequestException as e:
        return illustration_id, "", f"error: {e}"


def main(refresh_bulk: bool = False) -> int:
    _download_bulk_jsonl(refresh=refresh_bulk)
    illus_to_url = _illustration_to_art_crop_url()
    print(f"{len(illus_to_url)} distinct illustrations in Scryfall's bulk data", file=sys.stderr)

    df = pd.read_parquet(DATASET_PATH, columns=["illustration_id"])
    our_ids = sorted(df["illustration_id"].dropna().unique())
    print(f"{len(our_ids)} distinct illustration_ids to fetch", file=sys.stderr)

    todo = [(iid, illus_to_url[iid]) for iid in our_ids if iid in illus_to_url]
    missing = [iid for iid in our_ids if iid not in illus_to_url]
    if missing:
        print(f"  {len(missing)} illustration_ids not found in Scryfall bulk data, will be recorded as missing", file=sys.stderr)

    rows = [(iid, "", "missing_from_bulk_data") for iid in missing]
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_download_one, iid, url): iid for iid, url in todo}
        for fut in as_completed(futures):
            iid = futures[fut]
            url = illus_to_url[iid]
            _, local_path, status = fut.result()
            rows.append((iid, local_path, status))
            done += 1
            if done % 2000 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", file=sys.stderr)

    manifest = pd.DataFrame(rows, columns=["illustration_id", "local_path", "status"])
    manifest["art_crop_url"] = manifest["illustration_id"].map(illus_to_url)
    manifest.to_parquet(MANIFEST_PATH, index=False)
    print(manifest["status"].value_counts())
    print(f"manifest -> {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(refresh_bulk="--refresh" in sys.argv))
