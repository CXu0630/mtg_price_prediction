"""Exports Models A and B (and what their custom-card features read) to JSON, so the
frontend prices custom cards in the browser -- no server.

  web/public/data/models/model_a.json      point + sigma HGB trees, conformal q, feature vocab
  web/public/data/models/model_b.json      10/50/90% quantile HGB trees, CQR margin, category
                                           levels, live peer lookups (sibling / set moves)
  web/public/data/models/custom.json       custom-card defaults (demand ranks, pull rates, formats)
  web/public/data/models/reprints.json     per-card lookups for "reprint of" (rank, salt, tags,
                                           legalities, distinct release dates) -- loaded only when used
  web/src/model/parity.fixture.json        (--fixture) Python's predictions for a set of test
                                           cards, which the TypeScript port must reproduce

Trees are flattened into one node table per model; prediction mirrors scikit-learn's
_predict_one_from_raw_data exactly (NaN -> missing_go_to_left; categorical: raw bitset,
unknown categories treated as missing).

    python scripts/app/export_web_models.py [--fixture]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mtg_price_prediction import cards as cards_mod
from mtg_price_prediction import market
from mtg_price_prediction.fair_value import SIGMA_FLOOR

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
MODELS = ROOT / "models" / "app"
LIVE = ROOT / "data" / "app" / "live_rows.parquet"
OUT = ROOT / "web" / "public" / "data" / "models"
FIXTURE = ROOT / "web" / "src" / "model" / "parity.fixture.json"


FLOAT_MAX = float(np.finfo(np.float64).max)  # stands in for +-inf thresholds; hgb.ts maps it back
LEGAL_CODE = {"legal": "l", "not_legal": "n", "banned": "b", "restricted": "r"}


def _num(x):
    """JSON-safe number (NaN/inf -> None)."""
    x = float(x)
    return x if np.isfinite(x) else None


def export_hgb(est) -> dict:
    """One HistGradientBoostingRegressor -> flat node arrays + per-tree roots."""
    feature, threshold, left, right, missing_left, is_cat, bitset, value, roots = ([] for _ in range(9))
    bitsets: list[list[int]] = []
    for (pred,) in est._predictors:
        nodes, offset, b_offset = pred.nodes, len(feature), len(bitsets)
        roots.append(offset)
        bitsets.extend(pred.raw_left_cat_bitsets.astype(np.int64).tolist())
        for n in nodes:
            leaf = bool(n["is_leaf"])
            feature.append(-1 if leaf else int(n["feature_idx"]))
            t = 0.0 if leaf else float(n["num_threshold"])
            threshold.append(float(np.sign(t)) * FLOAT_MAX if np.isinf(t) else t)  # +-inf: JSON has no Infinity
            left.append(0 if leaf else offset + int(n["left"]))
            right.append(0 if leaf else offset + int(n["right"]))
            missing_left.append(int(n["missing_go_to_left"]))
            is_cat.append(int(n["is_categorical"]))
            bitset.append(b_offset + int(n["bitset_idx"]) if n["is_categorical"] else 0)
            value.append(float(n["value"]) if leaf else 0.0)
    known, f_idx = (None, None)
    if est.is_categorical_ is not None and est.is_categorical_.any():
        known_bitsets, f_idx_map = est._bin_mapper.make_known_categories_bitsets()
        known, f_idx = known_bitsets.astype(np.int64).tolist(), f_idx_map.astype(int).tolist()
    # With categorical features, sklearn fits an internal ColumnTransformer: categorical columns are
    # OrdinalEncoded (unknown -> NaN) and moved FIRST, then the numeric ones. Tree feature indices
    # refer to that transformed order, so the browser must apply the same transform.
    pre = None
    if getattr(est, "_preprocessor", None) is not None:
        t = {name: cols for name, _, cols in est._preprocessor.transformers_}
        pre = {"cat_idx": np.flatnonzero(t["encoder"]).tolist(), "num_idx": np.flatnonzero(t["numerical"]).tolist(),
               "categories": [c.astype(float).tolist() for c in est._preprocessor.named_transformers_["encoder"].categories_]}
    return {"pre": pre, "baseline": float(np.ravel(est._baseline_prediction)[0]), "roots": roots, "feature": feature,
            "threshold": threshold, "left": left, "right": right, "missing_left": missing_left, "is_cat": is_cat,
            "bitset": bitset, "value": value, "bitsets": bitsets, "known_bitsets": known, "f_idx_map": f_idx}


def export_model_a(a) -> dict:
    v = a.vocab
    return {
        "as_of": a.as_of, "q": a.q, "sigma_floor": SIGMA_FLOOR, "columns": a.columns,
        "point": export_hgb(a.model), "sigma": export_hgb(a.sigma_model),
        "vocab": {"lists": v["lists"], "cats": v["cats"], "rank_fill": v["rank_fill"],
                  "set_size": v["set_size"], "slot_size": v["slot_size"],
                  "set_price": {k: {s: _num(x) for s, x in d.items()} for k, d in v["set_price"].items()},
                  "rate_cols": v["rate_cols"],
                  "unknown_set": {k: {**{c: _num(x) for c, x in d.items() if c != "slot"},
                                      "slot": {r: _num(x) for r, x in d["slot"].items()}}
                                  for k, d in v["unknown_set"].items()}},
    }


def export_model_b(b, live: pd.DataFrame) -> dict:
    sib = live.groupby("oracle_id").agg(chg=("change_21d_pct", "mean"), lp=("log_price", "mean"))
    set_move = live.groupby(["set", "oracle_id"])["change_21d_pct"].mean().groupby("set").mean()
    return {
        "cutoff": b.cutoff, "q": b.q, "features": b.features, "cats": market.CATS, "cat_levels": b.cat_levels,
        "models": [export_hgb(m) for m in b.models],
        "sib": {o: [_num(r.chg), _num(r.lp)] for o, r in sib.iterrows()},
        "set_move": {s: _num(x) for s, x in set_move.items()},
        "constants": {"horizon": market.HORIZON, "new_release_days": market.NEW_RELEASE_DAYS,
                      "edhrec_worst": market.EDHREC_WORST},
    }


def export_cards(dataset: pd.DataFrame, a, as_of: pd.Timestamp) -> tuple[dict, dict]:
    """Lookups for custom cards: the reprinted card's first dataset row (what cards.custom_rows
    reads), its distinct release days (what print_history counts), and the defaults."""
    tags_vocab = set(a.vocab["lists"]["oracle_tags"])
    legal_cols = [c for c in dataset.columns if c.startswith("legalities.")]
    first = dataset.drop_duplicates("oracle_id")
    ev = dataset[["oracle_id", "released_at"]].drop_duplicates()
    ev = ev.assign(day=(pd.to_datetime(ev["released_at"]) - pd.Timestamp("1970-01-01")).dt.days)
    days = ev.groupby("oracle_id")["day"].apply(lambda s: sorted(int(d) for d in s.dropna()))
    n_print = dataset.groupby("oracle_id")["scryfall_id"].nunique()
    by_oracle = {}
    for d in first[["oracle_id", "edhrec_rank", "edhrec_salt", "oracle_tags", *legal_cols]].to_dict("records"):
        by_oracle[d["oracle_id"]] = {
            "rank": _num(d["edhrec_rank"]) if pd.notna(d["edhrec_rank"]) else None,
            "salt": _num(d["edhrec_salt"]) if pd.notna(d["edhrec_salt"]) else None,
            "tags": [t for t in (d["oracle_tags"] if d["oracle_tags"] is not None else []) if t in tags_vocab],
            "legal": "".join(LEGAL_CODE.get(d[c], "-") for c in legal_cols),
            "days": days.get(d["oracle_id"], []), "n": int(n_print.get(d["oracle_id"], 0)),
        }
    names = {}
    for name, oid in zip(dataset["name"], dataset["oracle_id"]):
        names.setdefault(name, oid)
    defaults = pd.read_parquet(cards_mod.PULL_RATE_DEFAULTS)
    rate_cols = sorted(c for c in dataset.columns if c.endswith("_rate_percent"))
    defaults_out = {
        "as_of_day": int((as_of - pd.Timestamp("1970-01-01")).days),
        "legal_formats": [c.split(".", 1)[1] for c in legal_cols],
        "legal_codes": {v: k for k, v in LEGAL_CODE.items()},
        "salt_median": float(dataset["edhrec_salt"].median()),
        "demand_rank": cards_mod.DEMAND_RANK,
        "booster_set_types": sorted(cards_mod.BOOSTER_SET_TYPES),
        "eternal": sorted(cards_mod.ETERNAL), "standard_era": sorted(cards_mod.STANDARD_ERA),
        "rate_cols": rate_cols,
        "pull_defaults": {f"{r['rarity']}|{r['finish']}": {c: _num(r[c]) for c in rate_cols if c in r}
                          for _, r in defaults.iterrows()},
    }
    return defaults_out, {"names": names, "oracles": by_oracle}


def write_fixture(dataset, as_of) -> None:
    """Python's own custom-card outputs for a varied set of cards -- the parity target."""
    from mtg_price_prediction.cards import CustomCard
    from mtg_price_prediction.fair_value import static_features, with_history
    from mtg_price_prediction.scoring import AppModels, score_custom

    models = AppModels.load(MODELS)
    live = pd.read_parquet(LIVE)
    rng = np.random.default_rng(0)
    real_names = dataset.drop_duplicates("oracle_id").sample(12, random_state=1)["name"].tolist()
    sets = dataset["set"].drop_duplicates().sample(6, random_state=2).tolist()
    cards = []
    for i in range(48):
        rarity = ["common", "uncommon", "rare", "mythic", "special"][i % 5]
        cards.append(CustomCard(
            name=f"Fixture {i}", rarity=rarity,
            type_line=["Instant", "Creature — Elf Druid", "Legendary Creature — Dragon", "Artifact",
                       "Legendary Planeswalker — Test", "Land", "Sorcery", "Enchantment — Aura"][i % 8],
            mana_cost=["{R}", "{1}{G}", "{3}{R}{R}", "{2}", "{2}{W}{U}", "", "{X}{B}{B}", "{2/W}{G/U}"][i % 8],
            oracle_text=["Deal 3 damage to any target.", "{T}: Add {G}.", "Flying, haste", "{T}: Add {C}{C}.",
                         "+1: Draw a card.", "", "Destroy target creature.", ""][i % 8],
            power=["2", None, "5", None, None, None, "*", None][i % 8],
            toughness=["2", None, "5", None, None, None, "1+*", None][i % 8],
            loyalty="4" if i % 8 == 4 else None,
            finish=["nonfoil", "foil", "etched"][i % 3],
            set_type=["expansion", "core", "masters", "commander", "funny", "draft_innovation"][i % 6],
            frame=["2015", "1997", "1993", "future"][i % 4],
            border_color=["black", "borderless", "white"][i % 3],
            frame_effects=[[], ["showcase"], ["extendedart", "legendary"]][i % 3],
            promo_types=[[], ["serialized"]][int(i % 7 == 0)],
            released_at=[None, "2026-11-14", "2020-01-01", "2026-09-25"][i % 4],
            demand=list(cards_mod.DEMAND_RANK)[i % 5],
            reprint_of=real_names[i // 4] if i % 4 == 3 else None,
            set=sets[i % 6] if i % 5 == 2 else None,
            quantity=1))
    out = score_custom(cards, dataset, models, live, as_of)
    rows = cards_mod.custom_rows(cards, dataset, as_of)
    hist = with_history(rows, dataset[["oracle_id", "released_at"]], as_of)
    xa = static_features(hist, models.a.vocab).reindex(columns=models.a.columns, fill_value=0.0)
    from dataclasses import asdict
    fixture = {"as_of": str(as_of.date()), "cards": [asdict(c) for c in cards],
               "expected": out[["fair", "fair_lo", "fair_hi", "pred_price", "price_lo", "price_hi", "signal"]]
               .to_dict("records"),
               "features_a": [[None if not np.isfinite(x) else float(x) for x in row] for row in xa.to_numpy()]}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, allow_nan=False))
    print(f"parity fixture: {len(cards)} cards -> {FIXTURE}")


def main() -> int:
    dataset = pd.read_parquet(DATASET)
    a, b = joblib.load(MODELS / "model_a.joblib"), joblib.load(MODELS / "model_b.joblib")
    live = pd.read_parquet(LIVE)
    as_of = pd.Timestamp(json.loads((MODELS / "manifest.json").read_text())["price_date"])
    OUT.mkdir(parents=True, exist_ok=True)
    custom, reprints = export_cards(dataset, a, as_of)
    for stale in OUT.glob("*.json"):
        stale.unlink()
    for name, obj in (("model_a", export_model_a(a)), ("model_b", export_model_b(b, live)),
                      ("custom", custom), ("reprints", reprints)):
        (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    size = sum(p.stat().st_size for p in OUT.glob("*.json")) / 1e6
    print(f"exported models -> {OUT} ({size:.1f} MB)")
    if "--fixture" in sys.argv:
        write_fixture(dataset, as_of)
    return 0


if __name__ == "__main__":
    sys.exit(main())
