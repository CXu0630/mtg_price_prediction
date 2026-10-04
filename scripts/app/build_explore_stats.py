"""Summary statistics for the web app's Data Exploration page.

    python scripts/app/build_explore_stats.py   # -> web/src/content/explore.json

Reads the joined training dataset (mtg-card-market-dataset/dataset.parquet; fetch it with
scripts/fetch_market_dataset.py) and the permutation-importance table behind figures/feature_importance.png
(figures/data/feature_importance.csv). The output is small and committed, so the site needs no extra data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
IMPORTANCE = ROOT / "figures" / "data" / "feature_importance.csv"
OUT = ROOT / "web" / "src" / "content" / "explore.json"

BUCKETS = [(0, 1, "under $1"), (1, 10, "$1–10"), (10, 100, "$10–100"), (100, np.inf, "$100+")]
RARITY_ORDER = ["common", "uncommon", "rare", "mythic", "special", "bonus"]
FEATURE_LABELS = {  # as in scripts/make_figures.py
    "edhrec_rank": "EDHREC rank", "edhrec_salt": "EDHREC salt", "penny_rank": "Penny Dreadful rank",
    "num_paper_printings": "# paper printings", "set_type": "set type", "border_color": "border color",
    "security_stamp": "security stamp", "conv_mana_cost": "mana value", "has_image": "has art image",
    "card flags (reserved, promo, reprint, borderless, ...)": "reserved / promo flags",
    "P/T / loyalty / defense": "P/T, loyalty", "oracle_tags": "oracle tags", "art_tags": "art tags",
    "type_tags": "type tags", "promo_types": "promo types", "frame_effects": "frame effects",
}
TOP_FEATURES = 11  # same selection as figures/feature_importance.png: top 11 + both embeddings
ALWAYS_SHOW = ["text embedding", "image embedding"]


def r(x, nd=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def main():
    d = pd.read_parquet(DATASET, columns=["oracle_id", "set", "set_type", "rarity", "finish", "released_at",
                                          "price_avg", "price_early", "price_late", "name", "set_name",
                                          "reserved", "edhrec_rank"])
    p = d.dropna(subset=["price_avg"])
    price = p["price_avg"]

    # log-spaced histogram, $0.01 .. $100k, 4 bins per decade
    edges = np.logspace(-2, 5, 29)
    counts, _ = np.histogram(price.clip(edges[0], edges[-1] * 0.999), bins=edges)
    first, last = np.flatnonzero(counts)[[0, -1]]
    hist = [{"lo": r(edges[i], 4), "hi": r(edges[i + 1], 4), "n": int(counts[i])} for i in range(first, last + 1)]

    buckets = []
    for lo, hi, label in BUCKETS:
        m = (price >= lo) & (price < hi)
        buckets.append({"label": label, "n": int(m.sum()), "share": r(m.mean() * 100, 1),
                        "value_share": r(price[m].sum() / price.sum() * 100, 1)})

    def by(col, order=None, top=None):
        g = p.groupby(col)["price_avg"].agg(n="size", median="median", mean="mean").reset_index()
        g = g.sort_values("n", ascending=False)
        if order:
            g = g.set_index(col).reindex([o for o in order if o in set(g[col])]).reset_index()
        if top:
            g = g.head(top)
        return [{"label": str(row[col]), "n": int(row["n"]), "median": r(row["median"]), "mean": r(row["mean"])}
                for _, row in g.iterrows()]

    # 3-month trend: early-window vs late-window mean price, per printing
    t = p.dropna(subset=["price_early", "price_late"])
    t = t[(t["price_early"] >= 1)]
    move = np.log(t["price_late"] / t["price_early"])
    trend = {"n": int(len(t)), "up10": r((move > np.log(1.1)).mean() * 100, 1),
             "down10": r((move < -np.log(1.1)).mean() * 100, 1), "median_move": r((np.exp(move.median()) - 1) * 100, 1)}

    top = p.sort_values("price_avg", ascending=False).head(10)
    expensive = [{"name": row["name"], "set": row["set"], "set_name": row["set_name"], "finish": row["finish"],
                  "price": r(row["price_avg"])} for _, row in top.iterrows()]

    has_rank = p["edhrec_rank"].notna()
    rank_bins = pd.qcut(p.loc[has_rank, "edhrec_rank"], 5, labels=["top 20%", "20–40%", "40–60%", "60–80%", "bottom 20%"])
    edhrec = [{"label": str(k), "median": r(v)} for k, v in p.loc[has_rank].groupby(rank_bins, observed=True)["price_avg"].median().items()]

    imp = pd.read_csv(IMPORTANCE)
    ranked = imp.groupby("group")["% of base (all)"].max().sort_values(ascending=False)
    keep = list(ranked.head(TOP_FEATURES).index) + [g for g in ALWAYS_SHOW if g not in ranked.head(TOP_FEATURES).index]
    importance = []
    for g in sorted(keep, key=lambda g: -ranked[g]):
        e = {"label": FEATURE_LABELS.get(g, g)}
        for _, row in imp[imp["group"] == g].iterrows():
            e["catboost" if row["model"] == "CatBoost" else "gp"] = r(row["% of base (all)"], 1)
        importance.append(e)

    out = {
        "totals": {"rows": int(len(d)), "priced": int(len(p)), "cards": int(d["oracle_id"].nunique()),
                   "sets": int(d["set"].nunique()), "window": "Jun 22 – Sep 20, 2026",
                   "p10": r(price.quantile(0.1)), "median": r(price.median()), "p90": r(price.quantile(0.9)),
                   "max": r(price.max())},
        "histogram": hist,
        "buckets": buckets,
        "rarity": by("rarity", order=RARITY_ORDER),
        "finish": by("finish", top=8),
        "set_type": by("set_type", top=8),
        "edhrec": edhrec,
        "reserved": {"median_reserved": r(p.loc[p["reserved"] == True, "price_avg"].median()),  # noqa: E712
                     "median_other": r(p.loc[p["reserved"] != True, "price_avg"].median())},  # noqa: E712
        "trend": trend,
        "expensive": expensive,
        "importance": importance,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
