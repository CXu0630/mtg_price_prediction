"""Downloads MTGJSON's AllPrices (rolling ~90 days, all providers, refreshed daily)
and flattens it to the same long format as mtg-card-prices/raw/price_history.parquet:

    scryfall_id, uuid, medium, provider, list_type, finish, currency, date, price

Only paper prices for printings in our dataset are kept (uuid -> scryfall_id
from mtg-card-market-dataset). Streamed with ijson -- the decompressed JSON is
several GB, never held in memory at once.

    python scripts/app/refresh_prices.py   # -> data/prices/price_history.parquet (+ data/prices/meta.json)
"""
from __future__ import annotations

import json
import lzma
import shutil
import sys
import urllib.request
from pathlib import Path

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "prices"
OUT_PATH = OUT_DIR / "price_history.parquet"
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
URL = "https://mtgjson.com/api/v5/AllPrices.json.xz"
META_URL = "https://mtgjson.com/api/v5/Meta.json"
HEADERS = {"User-Agent": "mtg-price-prediction/0.1 (academic research)"}
BATCH = 2_000_000


def _get(url: str, dest: Path) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=600) as r, dest.open("wb") as f:
        shutil.copyfileobj(r, f)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(META_URL, headers=HEADERS), timeout=60) as r:
        meta = json.load(r)["data"]
    raw = OUT_DIR / "AllPrices.json.xz"
    print(f"MTGJSON build {meta['version']} ({meta['date']}); downloading {URL}", flush=True)
    _get(URL, raw)

    ds = pd.read_parquet(DATASET, columns=["uuid", "scryfall_id"]).dropna().drop_duplicates("uuid")
    to_scryfall = dict(zip(ds["uuid"], ds["scryfall_id"]))

    schema = pa.schema([("scryfall_id", pa.string()), ("uuid", pa.string()), ("medium", pa.string()),
                        ("provider", pa.string()), ("list_type", pa.string()), ("finish", pa.string()),
                        ("currency", pa.string()), ("date", pa.timestamp("ns")), ("price", pa.float64())])
    tmp = OUT_PATH.with_suffix(".tmp.parquet")
    writer = pq.ParquetWriter(tmp, schema)
    cols = {name: [] for name in schema.names}
    n_rows = n_cards = 0

    def flush():
        nonlocal cols
        if cols["price"]:
            frame = pd.DataFrame(cols)
            frame["date"] = pd.to_datetime(frame["date"])
            writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
        cols = {name: [] for name in schema.names}

    with lzma.open(raw, "rb") as f:
        for uuid, mediums in ijson.kvitems(f, "data", use_float=True):
            sid = to_scryfall.get(uuid)
            if sid is None:
                continue
            n_cards += 1
            for provider, lists in (mediums.get("paper") or {}).items():
                currency = lists.get("currency")
                for list_type in ("retail", "buylist"):
                    for finish, series in (lists.get(list_type) or {}).items():
                        for date, price in series.items():
                            for k, v in (("scryfall_id", sid), ("uuid", uuid), ("medium", "paper"),
                                         ("provider", provider), ("list_type", list_type), ("finish", finish),
                                         ("currency", currency), ("date", date), ("price", price)):
                                cols[k].append(v)
                            n_rows += 1
            if len(cols["price"]) >= BATCH:
                flush()
    flush()
    writer.close()
    tmp.replace(OUT_PATH)
    raw.unlink()
    (OUT_DIR / "meta.json").write_text(json.dumps({"mtgjson_version": meta["version"], "mtgjson_date": meta["date"],
                                                   "rows": n_rows, "printings": n_cards}, indent=2))
    print(f"{n_rows:,} price rows for {n_cards:,} printings -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
