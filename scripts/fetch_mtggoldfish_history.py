"""Daily paper price history per (printing, finish) from MTGGoldfish, for
the temporal backtest (SPECS.md S6) -- MTGJSON/our current price data only
covers a rolling ~90 days, MTGGoldfish goes back to each card's release.

PRIVATE, ACADEMIC USE ONLY. MTGGoldfish's terms of use limit their content to
"personal, non-commercial use" and prohibit reproducing/publishing it.
Everything this writes lives in mtggoldfish/ (gitignored): never commit it,
upload it to HF, or publish a model trained on it without asking MTGGoldfish.

Polite by construction: one connection, >= REQUEST_INTERVAL_S between
requests, an honest User-Agent, exponential backoff + stop on repeated
429/5xx, and a resumable SQLite store so nothing is ever fetched twice.

    python scripts/fetch_mtggoldfish_history.py catalog   # ~340 set pages -> card_id mapping (~15 min)
    python scripts/fetch_mtggoldfish_history.py sample    # stratified-by-price sample (PRICE_BUCKETS), with weights
    python scripts/fetch_mtggoldfish_history.py history   # fetch the sample's histories, expensive first (~11 h)
    python scripts/fetch_mtggoldfish_history.py status
    python scripts/fetch_mtggoldfish_history.py export    # -> mtggoldfish/price_history.parquet

Two endpoints, both what the site's own pages load:
  - /sets/<Set>/All+Cards embeds JSON for every printing: MTGGoldfish's
    card_id (e.g. "Ragavan, Nimble Pilferer [MH2] (F)"), set code,
    collector number and finish -- joined to our dataset on
    (set, collector_number, finish).
  - /price_history_component?card_id=... returns the full daily paper price
    series for one card_id (one request per printing x finish).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "mtggoldfish"
DB_PATH = OUT_DIR / "history.sqlite"
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
BASE = "https://www.mtggoldfish.com"
USER_AGENT = "mtg-price-prediction/0.1 (academic non-commercial research; polite crawler, 1 req/1.5s)"
REQUEST_INTERVAL_S = 1.5
MAX_CONSECUTIVE_FAILURES = 8

# Our dataset's finish -> MTGGoldfish's. Specialty foils have their own
# finish on both sides under different spellings; anything unmapped (or
# missing on MTGGoldfish's side) falls back to plain "foil" when that is
# unambiguous -- see matched_printings().
FINISH_MAP = {
    "nonfoil": "regular", "foil": "foil", "etched": "foil_etched",
    "surgefoil": "surge_foil", "rainbowfoil": "rainbow_foil", "silverscroll": "silver_scroll_foil",
    "raisedfoil": "raised_foil", "facetfoil": "facet_foil", "galaxyfoil": "galaxy_foil",
    "confettifoil": "confetti_foil", "halofoil": "halo_foil", "textured": "textured_foil",
    "dazzlefoil": "dazzle_foil", "stepandcompleat": "step_and_compleat_foil", "cosmicfoil": "cosmic_foil",
}

CARD_JSON_RE = re.compile(
    r'\{"name":"(?:[^"\\]|\\.)+","set":"[^"]*","display_name":"(?:[^"\\]|\\.)*","rarity":"[^"]*",'
    r'"foil":(?:true|false),"card_num":[^,]*,"subset_id":"[^"]*","finish":"[^"]*"'
)
HISTORY_POINT_RE = re.compile(r"(\d{4}-\d{2}-\d{2}),\s*([\d.]+)")


class Client:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last_request = 0.0
        self.consecutive_failures = 0

    def get(self, path: str, params: dict | None = None, ajax: bool = False) -> str | None:
        headers = {"X-Requested-With": "XMLHttpRequest", "Accept": "text/javascript"} if ajax else {}
        backoff = 10.0
        for _ in range(4):
            wait = REQUEST_INTERVAL_S - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                r = self.session.get(BASE + path, params=params, headers=headers, timeout=60)
            except requests.RequestException as e:
                print(f"  network error: {e}", file=sys.stderr)
                r = None
            if r is not None and r.status_code == 200:
                self.consecutive_failures = 0
                return r.text
            if r is not None and r.status_code == 404:
                return None
            status = r.status_code if r is not None else "network"
            self.consecutive_failures += 1
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise SystemExit(f"stopping: {self.consecutive_failures} consecutive failures (last: {status}) "
                                 "-- being rate-limited or blocked; back off before resuming")
            print(f"  {status} on {path}; backing off {backoff:.0f}s", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 3
        return None


def connect() -> sqlite3.Connection:
    OUT_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sets (slug TEXT PRIMARY KEY, fetched_at TEXT, n_cards INTEGER);
        CREATE TABLE IF NOT EXISTS catalog (card_id TEXT PRIMARY KEY, set_code TEXT, card_num TEXT,
                                            finish TEXT, subset_id TEXT, display_name TEXT, rarity TEXT);
        CREATE TABLE IF NOT EXISTS fetched (card_id TEXT PRIMARY KEY, fetched_at TEXT, n_points INTEGER);
        CREATE TABLE IF NOT EXISTS history (card_id TEXT, date TEXT, price REAL, PRIMARY KEY (card_id, date));
    """)
    return db


def _fetch_set(client: Client, db: sqlite3.Connection, slug: str) -> int:
    text = client.get(f"/sets/{slug}/All+Cards")
    cards = []
    for obj in CARD_JSON_RE.findall(html.unescape(text or "")):
        c = json.loads(obj + "}")
        num = c["card_num"]
        num = str(int(num)) if isinstance(num, float) and num.is_integer() else str(num)
        cards.append((c["name"], c["set"].lower(), num, c["finish"], c["subset_id"], c["display_name"], c["rarity"]))
    db.executemany("INSERT OR IGNORE INTO catalog VALUES (?,?,?,?,?,?,?)", cards)
    db.execute("INSERT OR REPLACE INTO sets VALUES (?, datetime('now'), ?)", (slug, len(cards)))
    db.commit()
    return len(cards)


def cmd_catalog(client: Client, db: sqlite3.Connection) -> None:
    """Every set on MTGGoldfish's /sets index, then -- since the index omits
    many older/supplemental sets that still have pages -- a guessed slug
    (set name, spaces -> '+') for each of our dataset's sets whose code is
    still missing. Wrong guesses just 404 and are recorded as 0 cards."""
    page = client.get("/sets")
    slugs = sorted(set(re.findall(r'href="/sets/([^"/]+)"', html.unescape(page or ""))))
    ds_sets = pd.read_parquet(DATASET_PATH, columns=["set", "set_name"]).drop_duplicates("set")
    for pass_name in ("index", "dataset"):
        if pass_name == "dataset":
            have = {r[0] for r in db.execute("SELECT DISTINCT set_code FROM catalog")}
            missing = ds_sets[~ds_sets["set"].isin(have)]
            slugs = sorted({name.replace(" ", "+") for name in missing["set_name"]})
        done = {r[0] for r in db.execute("SELECT slug FROM sets")}
        todo = [s for s in slugs if s not in done]
        print(f"{pass_name} pass: {len(todo)} set pages to fetch", flush=True)
        for i, slug in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {slug}: {_fetch_set(client, db, slug)} printings", flush=True)


def _norm_name(name: pd.Series) -> pd.Series:
    return name.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


# Stratified sample by *current* price_avg: cheap buckets heavily
# downsampled, the expensive tail (where the models are weakest) taken in
# full. Each sampled printing carries weight = matched-in-bucket / sampled-
# in-bucket, so backtest metrics can be reweighted to the real population.
# Buckets are on today's price, so a card that crashed from $50 to $2 is
# sampled at the cheap rate -- the weights correct this in expectation.
PRICE_BUCKETS = [  # (label, lo, hi, quota; None = take all)
    ("<$1", 0.0, 1.0, 4000),
    ("$1-5", 1.0, 5.0, 4000),
    ("$5-10", 5.0, 10.0, 3000),
    ("$10-50", 10.0, 50.0, 5000),
    ("$50-100", 50.0, 100.0, None),
    ("$100-500", 100.0, 500.0, None),
    ("$500+", 500.0, float("inf"), None),
]


def _set_aliases(catalog: pd.DataFrame, ds: pd.DataFrame) -> dict[str, str]:
    """MTGGoldfish still files many older sets under legacy MTGO-style codes
    (7e, mm, od, uz, ...). Alias each unrecognized code to the dataset set
    whose card names overlap it in *both* directions (>= 90% of its names
    in ours, >= 60% of ours in it) -- the reverse check stops e.g. Mystery
    Booster (mb1) aliasing onto The List (plst), which contains most of
    mb1's names but is much larger."""
    ours = ds.groupby("set")["name_key"].agg(set)
    aliases = {}
    for code in set(catalog["set_code"]) - set(ours.index):
        names = set(catalog.loc[catalog["set_code"] == code, "name_key"])
        fwd = ours.map(lambda s: len(names & s) / len(names))
        best = fwd.idxmax()
        rev = len(names & ours[best]) / len(ours[best])
        if fwd[best] >= 0.9 and rev >= 0.6:
            aliases[code] = best
    return aliases


def matched_printings(db: sqlite3.Connection) -> pd.DataFrame:
    """Our priced dataset rows joined to MTGGoldfish card_ids, first match wins:
      1. (set, collector number, finish)
      2. (set, collector number, "foil")    -- specialty foil MTGGoldfish files as plain foil
      3. (set, card name, finish/"foil")    -- old sets have no collector numbers on MTGGoldfish
    Each step only uses keys unique on *both* sides, and a card_id is never
    assigned to two printings."""
    catalog = pd.read_sql("SELECT * FROM catalog", db)
    catalog["name_key"] = _norm_name(catalog["display_name"])
    ds = pd.read_parquet(DATASET_PATH, columns=["printing_key", "name", "set", "collector_number", "finish", "price_avg"])
    ds["name_key"] = _norm_name(ds["name"])
    catalog["set_code"] = catalog["set_code"].replace(_set_aliases(catalog, ds))
    ds = ds[ds["price_avg"].notna()].copy()
    ds["gf_finish"] = ds["finish"].map(FINISH_MAP).fillna("foil")
    ds["gf_foil"] = np.where(ds["finish"] == "nonfoil", "regular", "foil")

    steps = [(["set", "collector_number", "gf_finish"], ["set_code", "card_num", "finish"]),
             (["set", "collector_number", "gf_foil"], ["set_code", "card_num", "finish"]),
             (["set", "name_key", "gf_finish"], ["set_code", "name_key", "finish"]),
             (["set", "name_key", "gf_foil"], ["set_code", "name_key", "finish"])]
    matched, used_cards = [], set()
    remaining = ds
    for left, right in steps:
        cat = catalog[~catalog["card_id"].isin(used_cards)]
        cat = cat[~cat.duplicated(right, keep=False)]
        rem = remaining[~remaining.duplicated(left, keep=False)]
        m = rem.merge(cat[right + ["card_id"]], left_on=left, right_on=right, how="inner")
        m = m[~m.duplicated("card_id", keep=False)]
        matched.append(m[["printing_key", "card_id", "price_avg"]])
        used_cards |= set(m["card_id"])
        remaining = remaining[~remaining["printing_key"].isin(m["printing_key"])]
    return pd.concat(matched, ignore_index=True)


def cmd_sample(db: sqlite3.Connection, seed: int) -> None:
    """(Re)builds the `sample` table from PRICE_BUCKETS. Each bucket is a
    seeded random permutation truncated to its quota, so raising a quota
    later only *adds* printings -- nothing already fetched is wasted."""
    matched = matched_printings(db)
    n_total = pd.read_parquet(DATASET_PATH, columns=["price_avg"])["price_avg"].notna().sum()
    print(f"matched {len(matched)}/{n_total} priced dataset rows ({len(matched) / n_total:.1%}) to MTGGoldfish")
    rng = np.random.default_rng(seed)
    parts = []
    for label, lo, hi, quota in PRICE_BUCKETS:
        bucket = matched[(matched["price_avg"] >= lo) & (matched["price_avg"] < hi)]
        bucket = bucket.iloc[rng.permutation(len(bucket))]
        taken = bucket if quota is None else bucket.head(quota)
        parts.append(taken.assign(bucket=label, weight=len(bucket) / max(len(taken), 1)))
        print(f"  {label:9s} matched {len(bucket):6d}  sampled {len(taken):6d}  weight {len(bucket) / max(len(taken), 1):6.2f}")
    sample = pd.concat(parts)
    sample.to_sql("sample", db, if_exists="replace", index=False)
    print(f"{len(sample)} printings in sample (~{len(sample) * REQUEST_INTERVAL_S / 3600:.1f} h to fetch)")


def cmd_history(client: Client, db: sqlite3.Connection, limit: int | None) -> None:
    sample = pd.read_sql("SELECT card_id, price_avg FROM sample ORDER BY price_avg DESC", db)  # expensive first
    done = {r[0] for r in db.execute("SELECT card_id FROM fetched")}
    todo = [c for c in sample["card_id"] if c not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(sample)} printings in sample; {len(todo)} to fetch "
          f"(~{len(todo) * REQUEST_INTERVAL_S / 3600:.1f} h at {REQUEST_INTERVAL_S}s/request)", flush=True)
    t0 = time.monotonic()
    for i, card_id in enumerate(todo, 1):
        text = client.get("/price_history_component", ajax=True, params={
            "card_id": card_id, "selector": "#tab-paper", "type": "paper", "price_type": "card"})
        points = HISTORY_POINT_RE.findall(text or "")
        db.executemany("INSERT OR IGNORE INTO history VALUES (?,?,?)", [(card_id, d, float(p)) for d, p in points])
        db.execute("INSERT OR REPLACE INTO fetched VALUES (?, datetime('now'), ?)", (card_id, len(points)))
        if i % 25 == 0 or i == len(todo):
            db.commit()
            rate = i / (time.monotonic() - t0)
            print(f"[{i}/{len(todo)}] {card_id}: {len(points)} pts | ETA {(len(todo) - i) / rate / 3600:.1f} h", flush=True)
    db.commit()


def cmd_status(db: sqlite3.Connection) -> None:
    for name, q in [("sets fetched", "SELECT COUNT(*) FROM sets"), ("catalog card_ids", "SELECT COUNT(*) FROM catalog"),
                    ("histories fetched", "SELECT COUNT(*) FROM fetched"),
                    ("empty histories", "SELECT COUNT(*) FROM fetched WHERE n_points = 0"),
                    ("price points", "SELECT COUNT(*) FROM history"),
                    ("date range", "SELECT MIN(date) || ' -> ' || MAX(date) FROM history")]:
        print(f"{name:18s} {db.execute(q).fetchone()[0]}")
    if db.execute("SELECT name FROM sqlite_master WHERE name = 'sample'").fetchone():
        print(pd.read_sql("""SELECT s.bucket, COUNT(*) AS sampled, COUNT(f.card_id) AS fetched, MAX(s.weight) AS weight
                             FROM sample s LEFT JOIN fetched f USING (card_id) GROUP BY s.bucket""", db).to_string(index=False))


def cmd_export(db: sqlite3.Connection) -> None:
    """Long table keyed by our printing_key (joins straight onto the dataset),
    with each printing's sampling bucket + weight."""
    sample = pd.read_sql("SELECT printing_key, card_id, bucket, weight FROM sample", db)
    hist = pd.read_sql("SELECT * FROM history", db, parse_dates=["date"])
    out = sample.merge(hist, on="card_id")
    path = OUT_DIR / "price_history.parquet"
    out.to_parquet(path, index=False)
    print(f"{len(out)} rows, {out['printing_key'].nunique()} printings -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog")
    sub.add_parser("sample").add_argument("--seed", type=int, default=0)
    sub.add_parser("history").add_argument("--limit", type=int, default=None)
    sub.add_parser("status")
    sub.add_parser("export")
    args = parser.parse_args()

    db = connect()
    if args.cmd == "catalog":
        cmd_catalog(Client(), db)
    elif args.cmd == "sample":
        cmd_sample(db, args.seed)
    elif args.cmd == "history":
        cmd_history(Client(), db, args.limit)
    elif args.cmd == "status":
        cmd_status(db)
    else:
        cmd_export(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
