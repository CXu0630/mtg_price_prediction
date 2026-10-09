"""Joins mtg-print-distribution's rarity/pull-rate data onto mtg-card-market-dataset/
dataset.parquet, using MTGJSON uuid membership (booster_rates.card_uuids)
rather than set+collector_number: two products from the same set often draw
from different sheets, so set+collector_number can't say which products a card
comes from.

Run scripts/build_merged.py first. Requires mtg-print-distribution's own
db to have been rebuilt with card_uuids populated (its db.py migration
does this automatically) and its HF export run with --with-prices:

    cd mtg-print-distribution
    .venv/bin/python3 -m mtg_print_distribution.export_hf \\
        --out-dir data/exports/hf_dataset --with-prices

A card can be pullable from more than one *type* of product (Draft vs.
Collector vs. Jumpstart Booster, ...) at once, each with its own rate and
price -- there's no way to collapse these into one number without knowing
real per-product unit sales, which nobody publishes. So
instead of one scarcity feature, this adds three columns per product
*subtype* actually observed in the data (booster_products.subtype, the
15-value grouping -- draft/play/set/collector/jumpstart/theme/topper/...,
not the 52-value raw pack_code, most of which are one-off per-set variants
that don't generalize): `{subtype}_product_id` (a reference back to the
exact mtg-print-distribution booster_products row used), `{subtype}
_rate_percent` (avg_card_rate_percent -- the odds of pulling this exact
card from that product), `{subtype}_price_usd` (that product's own sealed
price -- context, not the card's price).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
DIST_EXPORT_DIR = ROOT / "mtg-print-distribution" / "data" / "exports" / "hf_dataset"


def _load_long_pull_rates() -> pd.DataFrame:
    """One row per (uuid, subtype match) -- explodes booster_rates.card_uuids
    and joins in the product's subtype/id/price."""
    products = pd.read_parquet(DIST_EXPORT_DIR / "booster_products.parquet")
    rates = pd.read_parquet(DIST_EXPORT_DIR / "booster_rates.parquet")

    # Exclude the synthetic all-time-average "default" set -- not a real,
    # joinable product (see default_set.py in that repo).
    products = products[products["source"] != "computed_average"]

    rates = rates[rates["card_uuids"] != ""].copy()
    rates["uuid"] = rates["card_uuids"].str.split(";")
    long = rates.explode("uuid")[["uuid", "booster_product_id", "finish", "avg_card_rate_percent"]]

    long = long.merge(
        products[["id", "subtype", "price_usd"]],
        left_on="booster_product_id", right_on="id", how="inner",
    )

    # Rare real collision: e.g. LTR has both "Jumpstart" and "Jumpstart Vol
    # 2" boosters, both subtype=jumpstart -- a card can match both. Keep
    # the higher-rate match per (uuid, finish, subtype) rather than add
    # column-index complexity for one-off cases.
    n_before = len(long)
    long = long.sort_values("avg_card_rate_percent", ascending=False).drop_duplicates(
        subset=["uuid", "finish", "subtype"], keep="first"
    )
    n_collisions = n_before - len(long)
    if n_collisions:
        print(f"  {n_collisions} (uuid, finish, subtype) collisions resolved by keeping the higher rate")

    return long.rename(columns={
        "booster_product_id": "product_id",
        "avg_card_rate_percent": "rate_percent",
    })[["uuid", "finish", "subtype", "product_id", "rate_percent", "price_usd"]]


def _pivot_wide(long: pd.DataFrame) -> pd.DataFrame:
    wide = long.pivot(index=["uuid", "finish"], columns="subtype",
                       values=["product_id", "rate_percent", "price_usd"])
    wide.columns = [f"{subtype}_{field}" for field, subtype in wide.columns]
    return wide.reset_index()


def join() -> pd.DataFrame:
    merged = pd.read_parquet(MERGED_PATH)
    long = _load_long_pull_rates()
    wide = _pivot_wide(long)

    n_before_matched = merged["uuid"].isin(long["uuid"]).sum()
    result = merged.merge(wide, on=["uuid", "finish"], how="left")

    subtype_cols = [c for c in wide.columns if c.endswith("_rate_percent")]
    n_matched = result[subtype_cols].notna().any(axis=1).sum()
    print(f"  {n_matched}/{len(result)} rows matched at least one product "
          f"({n_before_matched} distinct uuids had a match)")
    for c in sorted(subtype_cols):
        subtype = c.removesuffix("_rate_percent")
        print(f"    {subtype}: {result[c].notna().sum()} rows")
    return result


def main() -> int:
    result = join()
    result.to_parquet(MERGED_PATH, index=False)
    print(f"{len(result)} rows x {len(result.columns)} cols -> {MERGED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
