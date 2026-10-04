"""Presentation figures -> figures/*.png (16:9-friendly, 200 dpi, light surface).

    python scripts/make_figures.py

1. price_distribution.png  -- how card prices are distributed across all priced
   (printing, finish) rows, log-scaled, with p10 / median / p90 marked.
2. feature_importance.png  -- grouped permutation importance for both models
   (CatBoost, deep GP): % rise in held-out log-price MAE when that group is
   shuffled, for all cards and for $10+ cards. Reads
   figures/data/feature_importance.csv; run
   scripts/experiments/feature_importance.py first if it's missing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "figures"
DATASET_PATH = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
IMPORTANCE_PATH = OUT_DIR / "data" / "feature_importance.csv"

# Reference data-viz palette (light mode), validated as a 2-slot categorical set.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {"CatBoost": "#2a78d6", "Deep GP": "#eb6834"}
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 12, "text.color": TEXT, "text.parse_math": False,  # literal $ signs
    "axes.labelcolor": TEXT_2, "axes.edgecolor": GRID, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, fontsize=17, fontweight="bold", color=TEXT, va="top")
    fig.text(0.012, 0.915, subtitle, fontsize=11.5, color=TEXT_2, va="top")


def _dollars(x, _pos=None) -> str:
    return f"${x:,.2f}" if x < 1 else f"${x:,.0f}"


def price_distribution() -> Path:
    price = pd.read_parquet(DATASET_PATH, columns=["price_avg"])["price_avg"].dropna()
    price = price[price > 0]
    p10, p50, p90 = np.percentile(price, [10, 50, 90])

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.12)
    bins = np.logspace(np.log10(price.min()), np.log10(price.max()), 61)
    ax.hist(price, bins=bins, color=SERIES["CatBoost"], edgecolor=SURFACE, linewidth=1.2)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_xlabel("average price, Jun–Sep 2026 (log scale)")
    ax.set_ylabel("printings")

    # Headroom above the tallest bar so the marker labels never sit on the bars.
    counts, _ = np.histogram(price, bins=bins)
    top = counts.max() * 1.22
    ax.set_ylim(0, top)
    for value, label, style in [(p10, "p10", "--"), (p50, "median", "-"), (p90, "p90", "--")]:
        ax.axvline(value, ymax=0.97, color=TEXT_2, linewidth=1.2, linestyle=style)
        ax.text(value, top * 0.985, f" {label}\n ${value:,.2f}", color=TEXT, fontsize=11, va="top", ha="left")

    share_under_1 = (price < 1).mean()
    share_over_100 = (price >= 100).mean()
    _title(fig, "Most printings are bulk; value is in a long tail",
           f"n = {len(price):,} priced (printing, finish) rows · {share_under_1:.0%} under $1 · "
           f"{share_over_100:.1%} at $100+")
    path = OUT_DIR / "price_distribution.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


LABELS = {
    "edhrec_rank": "EDHREC rank", "edhrec_salt": "EDHREC salt", "penny_rank": "Penny Dreadful rank",
    "num_paper_printings": "# paper printings", "set_type": "set type", "border_color": "border color",
    "security_stamp": "security stamp", "conv_mana_cost": "mana value", "has_image": "has art image",
    "card flags (reserved, promo, reprint, borderless, ...)": "reserved / promo flags",
    "P/T / loyalty / defense": "P/T, loyalty", "oracle_tags": "oracle tags", "art_tags": "art tags",
    "type_tags": "type tags", "promo_types": "promo types", "frame_effects": "frame effects",
}
ALWAYS_SHOW = ["text embedding", "image embedding"]


def feature_importance(top_n: int = 11) -> Path:
    if not IMPORTANCE_PATH.exists():
        raise SystemExit(f"{IMPORTANCE_PATH} missing -- run scripts/experiments/feature_importance.py first")
    imp = pd.read_csv(IMPORTANCE_PATH)
    # Top groups by their larger effect across the two models (overall), plus the
    # embeddings, which are part of the story even when they rank low.
    ranked = imp.groupby("group")["% of base (all)"].max().sort_values(ascending=False)
    keep = list(ranked.head(top_n).index) + [g for g in ALWAYS_SHOW if g not in ranked.head(top_n).index]
    order = sorted(keep, key=lambda g: ranked[g])  # ascending -> largest drawn at the top
    panels = [("% of base (all)", "All cards"), ("% of base ($10+)", "$10+ cards")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.4), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.16, right=0.965, top=0.81, bottom=0.11, wspace=0.12)
    y = np.arange(len(order))
    h = 0.38
    for ax, (col, panel_title) in zip(axes, panels):
        for k, (model, color) in enumerate(SERIES.items()):
            offset = (0.5 - k) * h  # first series drawn on top, matching the legend order
            vals = imp[imp["model"] == model].set_index("group")[col].reindex(order)
            ax.barh(y + offset, vals.fillna(0), height=h, color=color, edgecolor=SURFACE, linewidth=1.5, label=model)
            for yi, v in zip(y, vals):
                if pd.isna(v):  # group the model doesn't have (CatBoost uses no embeddings)
                    ax.text(1.5, yi + offset, "not used", va="center", fontsize=9.5, color=TEXT_2)
        ax.set_title(panel_title, loc="left", color=TEXT)
        ax.set_yticks(y, [LABELS.get(g, g) for g in order])
        ax.xaxis.set_major_locator(MultipleLocator(40))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"+{v:.0f}%"))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_xlabel("rise in held-out error when shuffled")
    axes[1].legend(loc="lower right", frameon=False)
    _title(fig, "What drives the models: EDHREC rank first, then scarcity and printing type",
           "Permutation importance on each model's held-out test split (log-price MAE). "
           "Correlated groups share credit (e.g. rarity ↔ pull rates).")
    path = OUT_DIR / "feature_importance.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    for make in (price_distribution, feature_importance):
        print(f"-> {make()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
