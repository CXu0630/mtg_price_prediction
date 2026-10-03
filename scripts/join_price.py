"""Joins card-level price onto mtg-card-market-dataset/dataset.parquet from
mtg-card-prices/raw/price_history.parquet.

Provider is hardcoded to tcgplayer -- checked all 4 paper providers first
(see session notes / mtg-card-market-dataset/README.md): manapool's series has real,
sane values mixed with multi-million-dollar spikes for ordinary cards
(confirmed on Mind Stone: $2.86-$6.41 on cardkingdom/cardmarket/tcgplayer,
$2.86-$3,537,205.22 on manapool for the identical card) -- a genuine
upstream data bug, not just an outlier to clip. cardkingdom and cardmarket
checked out clean too, but tcgplayer is what the rest of this project
already resolves prices through (mtg-print-distribution's own
tcgplayer.py), so sticking with one provider throughout.

price_history's `finish` vocabulary is coarser than mtg-card-market-dataset's
(normal/foil/etched only -- no specialty-foil names), matching what was
already found querying TCGplayer's own pricepoints API directly: it never
returns a finer printingType than Normal/Foil either. So every specialty
foil in mtg-card-market-dataset's `finish` (surgefoil, silverfoil, ...) maps to
price_history's plain "foil" here -- there's no finer price signal to
join against even though the treatment itself is more specific.

Output is price_min/price_max/price_avg per (uuid, finish) over
price_history's ~3-month window (2026-06-22 to 2026-09-20 as of this
pull) -- matching the shape SPECS.md already expected from the teammate's
per-card dataset, not a single point price (see SPECS.md §3 on why: a
point price goes stale, a min/max/avg range is closer to what "predict a
price range" needs anyway).

Also adds a compact 3-point trend curve (price_early/price_mid/price_late)
-- the window split into 3 equal *calendar* periods (not per-card relative
thirds, so "price_mid" means the same ~30 days for every card, comparable
across cards) with the mean price in each, rather than keeping the full
~90-day daily series in mtg-card-market-dataset. A single linear slope was considered
and rejected for now -- it collapses "spiked then came back down" into a
near-zero trend same as "flat the whole time"; three points keep that
shape for barely more cost.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
PRICE_HISTORY_PATH = ROOT / "mtg-card-prices" / "raw" / "price_history.parquet"

PROVIDER = "tcgplayer"


def _map_finish(finish: str) -> str:
    """mtg-card-market-dataset's finish -> price_history's coarser finish vocabulary."""
    if finish == "nonfoil":
        return "normal"
    if finish == "etched":
        return "etched"
    return "foil"  # covers "foil" itself and every specialty-foil name


PERIOD_COLS = ["price_early", "price_mid", "price_late"]


def _load_price_summary() -> pd.DataFrame:
    p = pd.read_parquet(PRICE_HISTORY_PATH)
    p = p[
        (p["medium"] == "paper")
        & (p["list_type"] == "retail")
        & (p["currency"] == "USD")
        & (p["provider"] == PROVIDER)
    ]

    summary = p.groupby(["uuid", "finish"])["price"].agg(
        price_min="min", price_max="max", price_avg="mean",
    ).reset_index()

    # 3-point trend curve: same 3 calendar periods for every card (not
    # per-card relative thirds), mean price within each.
    bin_edges = pd.date_range(p["date"].min(), p["date"].max(), periods=4)
    p["period"] = pd.cut(p["date"], bins=bin_edges, labels=PERIOD_COLS, include_lowest=True)
    periods = (
        p.groupby(["uuid", "finish", "period"], observed=True)["price"]
        .mean()
        .unstack("period")
        .reindex(columns=PERIOD_COLS)
        .reset_index()
    )
    summary = summary.merge(periods, on=["uuid", "finish"], how="left")

    return summary.rename(columns={"finish": "price_finish"})


OUTPUT_COLS = ["price_min", "price_max", "price_avg", *PERIOD_COLS]


def join() -> pd.DataFrame:
    merged = pd.read_parquet(MERGED_PATH)
    # Idempotent -- drop any output of a previous run before re-merging,
    # so re-running this script doesn't collide with its own prior columns.
    merged = merged.drop(columns=[c for c in OUTPUT_COLS if c in merged.columns])
    merged["price_finish"] = merged["finish"].apply(_map_finish)

    summary = _load_price_summary()
    result = merged.merge(summary, on=["uuid", "price_finish"], how="left")
    result = result.drop(columns=["price_finish"])

    n_matched = result["price_avg"].notna().sum()
    print(f"  {n_matched}/{len(result)} rows matched a {PROVIDER} price "
          f"({merged['uuid'].isin(summary['uuid']).sum()} distinct uuids had a match)")
    return result


def main() -> int:
    result = join()
    result.to_parquet(MERGED_PATH, index=False)
    print(f"{len(result)} rows x {len(result.columns)} cols -> {MERGED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
