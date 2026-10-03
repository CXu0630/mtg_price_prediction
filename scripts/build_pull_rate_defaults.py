"""Builds a rarity/finish -> expected booster pull-rate lookup table, so a
card that isn't (yet) printed in a known real product can still get a
per-subtype `{subtype}_rate_percent` feature -- a typical value for "a card
of this rarity/finish," in place of a real per-product number.

This is what makes pull-rate features usable as an inference-time input for
a hypothetical/unreleased card: rarity and finish are both things a card's
designer declares directly (or are visible on a real card), so this lookup
turns "we don't have this card's real pull rate" into "use the typical rate
for cards like it," rather than requiring the card to already exist in a
real product (SPECS.md 4.2 inference input contract).

Two-level fallback per subtype, since many (rarity, finish) combinations are
sparse or structurally impossible (e.g. secret_lair_rate_percent barely
exists outside its own product line):
  1. mean over (rarity, finish), if at least MIN_GROUP_N real observations
  2. mean over (rarity) alone, if at least MIN_GROUP_N real observations
  3. global mean over all real observations
  4. NaN, if the subtype has essentially no real data at all (e.g. a
     subtype that's a one-off product) -- left for the missingness flag to
     handle, not a fabricated number.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
OUT_PATH = ROOT / "training" / "pull_rate_defaults.parquet"

MIN_GROUP_N = 30


def build() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PATH)
    rate_cols = [c for c in df.columns if c.endswith("_rate_percent")]

    rarity_finish = df.groupby(["rarity", "finish"])[rate_cols].agg(["mean", "count"])
    rarity_only = df.groupby(["rarity"])[rate_cols].agg(["mean", "count"])
    global_mean = df[rate_cols].mean()

    combos = df[["rarity", "finish"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for rarity, finish in combos.itertuples(index=False):
        row = {"rarity": rarity, "finish": finish}
        for col in rate_cols:
            rf_mean, rf_n = rarity_finish.loc[(rarity, finish), (col, "mean")], rarity_finish.loc[(rarity, finish), (col, "count")]
            if rf_n >= MIN_GROUP_N:
                row[col] = rf_mean
                continue
            r_mean, r_n = rarity_only.loc[rarity, (col, "mean")], rarity_only.loc[rarity, (col, "count")]
            if r_n >= MIN_GROUP_N:
                row[col] = r_mean
                continue
            row[col] = global_mean[col] if pd.notna(global_mean[col]) else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    out = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    n_defaultable = out.drop(columns=["rarity", "finish"]).notna().sum().sum()
    n_total = out.drop(columns=["rarity", "finish"]).size
    print(f"{len(out)} (rarity, finish) combos -> {OUT_PATH} ({n_defaultable}/{n_total} subtype defaults filled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
