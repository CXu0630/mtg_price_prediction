"""Models B and C -- 14-day TCGplayer move and 14-day Card Kingdom buylist move.

Rebuilt from the teammate's mtgpricepredictor (build_daily.py, price_move.py)
on our own data, with two deliberate differences (docs/APP_DESIGN.md):
  - no Model A inputs (gap, fair_log): their own ablation shows no loss, and it
    makes B/C independent of which Model A backend the app shows
  - no Manapool level (its prices are unreliable in our data); the CK buylist
    trust rule compares CK retail to TCGplayer only

Rows: one per (printing, day t), as-of t. Targets need t+14, so the newest 14
days are "live" rows -- scored, never trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON, LOOKBACK, BULK, SPIKE = 14, 21, 0.50, 2.0
LEVEL_CARRY_DAYS = 7
EDHREC_WORST = 40_000
NEW_RELEASE_DAYS = 30
CK_RETAIL_MAX_VS_TCG = 3.0          # CK retail > 3x TCGplayer = version mismatch / bad listing
UNCONFIRMED_JUMP = np.log(2)        # |jump_7d| above this: signals stay HOLD
ALPHA = 0.2
MARKETS = {"ck_retail": ("cardkingdom", "retail"), "ck_buylist": ("cardkingdom", "buylist"),
           "cardmarket": ("cardmarket", "retail")}
PH_FINISH = {"nonfoil": "normal", "etched": "etched"}  # everything else -> "foil"

B_NUMS = ["log_price", "change_21d_pct", "volatility_21d", "jump_7d", "releases_before_target",
          "printing_age_days", "card_age_days", "days_since_last_printing", "printings_to_date", "log_edhrec",
          "reprint", "promo", "reserved", "sib_chg", "sib_div", "sib_rel_level", "set_peer",
          "lvl_ck_retail", "lvl_ck_buylist", "lvl_cardmarket"]
C_EXTRA = ["bl_chg_7d", "bl_chg_21d", "bl_vs_ckr", "bl_days_same"]
CATS = ["release_status", "rarity", "finish_group", "set_type"]
CAT_LEVELS = {"release_status": ["preorder", "new", "established"],
              "finish_group": ["nonfoil", "foil", "etched", "special"]}


# --- Price matrices -------------------------------------------------------

def universe(dataset: pd.DataFrame) -> pd.DataFrame:
    """Dataset printings the market models can see: basic lands out, one row per
    (scryfall_id, MTGJSON finish) -- specialty foils share the plain-foil series."""
    ds = dataset[~dataset["type_line"].fillna("").str.contains("Basic Land")].copy()
    ds["ph_finish"] = ds["finish"].map(PH_FINISH).fillna("foil")
    ds = ds.drop_duplicates(["scryfall_id", "ph_finish"]).reset_index(drop=True)
    ds["released_at"] = pd.to_datetime(ds["released_at"])
    return ds


def wide_prices(price_history: Path, provider: str, list_type: str, ds: pd.DataFrame) -> pd.DataFrame:
    t = pq.read_table(price_history, filters=[("provider", "=", provider), ("list_type", "=", list_type),
                                             ("medium", "=", "paper")],
                      columns=["scryfall_id", "finish", "date", "price"]).to_pandas()
    w = t.pivot_table(index=["scryfall_id", "finish"], columns="date", values="price", aggfunc="mean")
    w = w.reindex(columns=pd.date_range(t["date"].min(), t["date"].max(), freq="D"))  # MTGJSON skips days
    w = w.reindex(pd.MultiIndex.from_frame(ds[["scryfall_id", "ph_finish"]], names=["scryfall_id", "finish"]))
    w.index = ds.index
    return w


def despike(w: pd.DataFrame, causal: bool) -> pd.DataFrame:
    """Drop days more than SPIKE x off their 5-day neighbourhood median, carry forward.
    causal: trailing median (features never look past t); else centered (targets)."""
    med = (w.T.shift(1).rolling(5, min_periods=3).median() if causal
           else w.T.rolling(5, center=True, min_periods=3).median()).T
    ratio = w / med
    return w.mask((ratio > SPIKE) | (ratio < 1 / SPIKE)).T.ffill().T


@dataclass
class Prices:
    tcg_feat: pd.DataFrame          # causal-cleaned TCGplayer, for features
    tcg_target: pd.DataFrame        # centered-cleaned TCGplayer, for targets
    levels: dict[str, pd.DataFrame]  # causal-cleaned other markets, carried forward <= 7 days
    ck_buylist_raw: pd.DataFrame    # as reported (C's target uses the day itself, not carried forward)
    ck_buylist_ffill: pd.DataFrame  # carried forward, for buylist momentum
    ck_buylist_last_change: pd.DataFrame  # day index (0..) of the buylist's latest change on or before each day

    @classmethod
    def load(cls, price_history: Path, ds: pd.DataFrame) -> "Prices":
        tcg = wide_prices(price_history, "tcgplayer", "retail", ds)
        levels = {m: despike(wide_prices(price_history, p, lt, ds), causal=True).T.ffill(limit=LEVEL_CARRY_DAYS).T
                  for m, (p, lt) in MARKETS.items()}
        cols = tcg.columns
        bl_raw = wide_prices(price_history, "cardkingdom", "buylist", ds).reindex(columns=cols)
        bl_ffill = bl_raw.T.ffill().T
        day_idx = np.broadcast_to(np.arange(len(cols), dtype=float), bl_ffill.shape)
        changed = bl_ffill.diff(axis=1).ne(0) & bl_ffill.notna()
        last_change = pd.DataFrame(np.where(changed, day_idx, np.nan), index=bl_ffill.index, columns=cols).T.ffill().T
        return cls(despike(tcg, True), despike(tcg, False), {m: w.reindex(columns=cols) for m, w in levels.items()},
                   bl_raw, bl_ffill, last_change)


# --- Daily rows -----------------------------------------------------------

def peer_features(day: pd.DataFrame) -> pd.DataFrame:
    """sib_chg / sib_div / sib_rel_level over the card's other printings, set_peer over the set's other cards."""
    chg, lp = day["change_21d_pct"], day["log_price"]
    pchg = chg.where(day["price"] >= BULK)
    day = day.assign(_pchg=pchg)
    g = day.groupby("oracle_id")
    s, n = g["_pchg"].transform("sum"), g["_pchg"].transform("count")
    own = pchg.notna()
    out = pd.DataFrame(index=day.index)
    out["sib_chg"] = (s - pchg.fillna(0)) / (n - own).replace(0, np.nan)
    out["sib_div"] = chg - out["sib_chg"]
    ls, ln = g["log_price"].transform("sum"), g["log_price"].transform("count")
    out["sib_rel_level"] = lp - (ls - lp) / (ln - 1).replace(0, np.nan)
    card = day[own].groupby(["set", "oracle_id"])["_pchg"].mean().rename("card_chg").reset_index()
    tot = card.groupby("set")["card_chg"].agg(["sum", "count"])
    card = card.join(tot, on="set")
    card["set_peer"] = (card["sum"] - card["card_chg"]) / (card["count"] - 1).replace(0, np.nan)
    out["set_peer"] = day[["set", "oracle_id"]].merge(card[["set", "oracle_id", "set_peer"]], how="left",
                                                      on=["set", "oracle_id"])["set_peer"].to_numpy()
    return out


def print_history(ds: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    ev = ds[["oracle_id", "released_at"]].drop_duplicates()
    ev = ev[ev["released_at"] <= t]
    agg = ev.groupby("oracle_id")["released_at"].agg(["min", "max", "size"]).reindex(ds["oracle_id"])
    return pd.DataFrame({
        "printing_age_days": (t - ds["released_at"]).dt.days.to_numpy(float),
        "card_age_days": (t - agg["min"]).dt.days.to_numpy(float),
        "days_since_last_printing": (t - agg["max"]).dt.days.to_numpy(float),
        "printings_to_date": agg["size"].fillna(0).to_numpy(float),
    }, index=ds.index)


def static_frame(ds: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "printing_key": ds["printing_key"], "oracle_id": ds["oracle_id"], "set": ds["set"],
        "log_edhrec": np.log(ds["edhrec_rank"].fillna(EDHREC_WORST).astype(float)),
        "reprint": ds["reprint"].astype(float), "promo": ds["promo"].astype(float),
        "reserved": ds["reserved"].astype(float), "rarity": ds["rarity"], "set_type": ds["set_type"],
        "finish_group": ds["finish"].where(ds["finish"].isin(["nonfoil", "foil", "etched"]), "special"),
    })


def day_rows(ds: pd.DataFrame, px: Prices, static: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Every priced printing's as-of-t features, B target (ret) and C inputs/target where they exist."""
    f = px.tcg_feat
    p = f[t]
    start = f[t - pd.Timedelta(days=LOOKBACK)]
    window = f.loc[:, t - pd.Timedelta(days=LOOKBACK):t]
    prior7 = f.loc[:, t - pd.Timedelta(days=7):t - pd.Timedelta(days=1)]
    day = static.copy()
    day["snapshot_date"] = t
    day["price"] = p
    day["log_price"] = np.log(p)
    day["change_21d_pct"] = (100 * (p / start - 1)).clip(-90, 300)
    day["volatility_21d"] = (window / window.shift(axis=1) - 1).std(axis=1).clip(0, 0.5)
    day["jump_7d"] = np.log(p / prior7.median(axis=1))
    day = day.join(peer_features(day))
    for m, w in px.levels.items():
        day[f"lvl_{m}"] = np.log(w[t] / p).clip(-3, 3)
    day = day.join(print_history(ds, t))
    rel = ds["released_at"]
    day["release_status"] = np.select([rel > t, rel > t - pd.Timedelta(days=NEW_RELEASE_DAYS)],
                                      ["preorder", "new"], "established")
    day["releases_before_target"] = ((rel > t) & (rel <= t + pd.Timedelta(days=HORIZON))).astype(float)

    # Card Kingdom: clean buylist (CK lists a retail price that day, buylist <= retail, retail sane vs TCG).
    ck_r, bl = px.levels["ck_retail"][t], px.ck_buylist_raw[t]
    trusted = bl.notna() & ck_r.notna() & (bl <= ck_r) & (ck_r <= CK_RETAIL_MAX_VS_TCG * p)
    day["ck_retail"], day["ck_buylist"] = ck_r, bl.where(trusted)
    day["cardmarket_eur"] = px.levels["cardmarket"][t]
    for d in (7, 21):
        past_t = t - pd.Timedelta(days=d)
        past = px.ck_buylist_ffill[past_t] if past_t in px.ck_buylist_ffill else np.nan
        day[f"bl_chg_{d}d"] = np.log(day["ck_buylist"] / past)
    day["bl_vs_ckr"] = np.log(day["ck_buylist"] / ck_r)
    t_idx = px.ck_buylist_ffill.columns.get_loc(t)
    day["bl_days_same"] = (t_idx - px.ck_buylist_last_change[t]).where(day["ck_buylist"].notna())

    target_t = t + pd.Timedelta(days=HORIZON)
    if target_t in px.tcg_target:
        day["ret"] = np.log(px.tcg_target[target_t] / px.tcg_target[t])
        bl_future = px.ck_buylist_raw[target_t]
        day["ret_buylist"] = np.log(bl_future / day["ck_buylist"])
    else:
        day["ret"] = day["ret_buylist"] = np.nan
    return day[day["price"] >= BULK]


def build_rows(ds: pd.DataFrame, px: Prices, days=None) -> pd.DataFrame:
    """All snapshot days with a full lookback (targets NaN for the newest HORIZON days)."""
    dates = px.tcg_feat.columns
    first = dates[0] + pd.Timedelta(days=LOOKBACK)
    days = [d for d in dates if d >= first] if days is None else days
    static = static_frame(ds)
    return pd.concat([day_rows(ds, px, static, t) for t in days], ignore_index=True)


# --- Models -----------------------------------------------------------------

def encode_cats(df: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for c in CATS:
        out[c] = pd.Categorical(out[c], categories=levels[c]).codes.astype(float)
        out.loc[out[c] < 0, c] = np.nan
    return out


def conformal_q(scores: np.ndarray, alpha: float = ALPHA) -> float:
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, len(scores)) - 1])


@dataclass
class MoveModel:
    """Three quantile HGBs (10/50/90%) + CQR margin -- one per target (B: 'ret', C: 'ret_buylist')."""
    target: str
    features: list[str]
    cat_levels: dict[str, list[str]]
    models: list = field(default_factory=list)
    q: float = 0.0
    cutoff: str = ""

    @property
    def columns(self) -> list[str]:
        return self.features + CATS

    def _X(self, rows: pd.DataFrame) -> pd.DataFrame:
        return encode_cats(rows, self.cat_levels)[self.columns]

    def fit(self, rows: pd.DataFrame, max_train_rows: int = 600_000, seed: int = 0) -> "MoveModel":
        known = rows[rows[self.target].notna() & np.isfinite(rows[self.target])]
        last = known["snapshot_date"].max()
        cal = known[known["snapshot_date"] > last - pd.Timedelta(days=7)]
        train = known[known["snapshot_date"] <= last - pd.Timedelta(days=7)]
        train = train.sample(min(max_train_rows, len(train)), random_state=seed)
        cat_mask = [c in CATS for c in self.columns]
        self.models = [HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                                     min_samples_leaf=100, categorical_features=cat_mask,
                                                     random_state=seed).fit(self._X(train), train[self.target])
                       for q in (0.1, 0.5, 0.9)]
        lo, _, hi = self._raw(cal)
        self.q = conformal_q(np.maximum(lo - cal[self.target], cal[self.target] - hi))
        self.cutoff = str(last.date())
        return self

    def _raw(self, rows):
        lo, med, hi = (m.predict(self._X(rows)) for m in self.models)
        return np.minimum(lo, med), med, np.maximum(hi, med)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Log-move median + calibrated 80% range."""
        lo, med, hi = self._raw(rows)
        # The CQR margin can be negative (raw quantiles over-covered on calibration): re-clip around the median.
        lo, hi = np.minimum(lo - self.q, med), np.maximum(hi + self.q, med)
        return pd.DataFrame({"move_med": med, "move_lo": lo, "move_hi": hi}, index=rows.index)


def cat_levels_from(rows: pd.DataFrame) -> dict[str, list[str]]:
    levels = dict(CAT_LEVELS)
    for c in ("rarity", "set_type"):
        levels[c] = sorted(rows[c].dropna().unique().tolist())
    return levels


def model_b(rows) -> MoveModel:
    return MoveModel("ret", B_NUMS, cat_levels_from(rows))


def model_c(rows) -> MoveModel:
    return MoveModel("ret_buylist", B_NUMS + C_EXTRA, cat_levels_from(rows))


# --- Custom cards: a synthetic day-0 row ------------------------------------

def custom_day0_rows(custom: pd.DataFrame, fair_price: np.ndarray, live: pd.DataFrame, t: pd.Timestamp,
                     events: pd.DataFrame) -> pd.DataFrame:
    """Model B rows for custom cards. One day of data -- the price Model A predicts:
    no momentum, no other markets. Peers come from the live rows of real cards:
    siblings from `reprint_of_oracle` (its real printings' moves), set peers from `set`."""
    rows = static_frame(custom.assign(printing_key=custom["printing_key"]))
    rows["snapshot_date"] = t
    rows["price"] = fair_price
    rows["log_price"] = np.log(fair_price)
    for c in ("change_21d_pct", "volatility_21d", "jump_7d"):
        rows[c] = 0.0
    sib = live.groupby("oracle_id").agg(sib_chg=("change_21d_pct", "mean"), sib_lp=("log_price", "mean"))
    oracle = custom["reprint_of_oracle"]
    rows["sib_chg"] = oracle.map(sib["sib_chg"]).to_numpy()
    rows["sib_div"] = -rows["sib_chg"]
    rows["sib_rel_level"] = rows["log_price"] - oracle.map(sib["sib_lp"]).to_numpy()
    set_move = live.groupby(["set", "oracle_id"])["change_21d_pct"].mean().groupby("set").mean()
    rows["set_peer"] = custom["set"].map(set_move).to_numpy()
    for m in MARKETS:
        rows[f"lvl_{m}"] = np.nan
    # Printing history as of t: the real card's printings (`events`: dataset oracle_id, released_at) + this one.
    c = custom.assign(released_at=pd.to_datetime(custom["released_at"]),
                      oracle_id=custom["reprint_of_oracle"].fillna(custom["oracle_id"]))
    hist = print_history(pd.concat([c[["oracle_id", "released_at"]], events[["oracle_id", "released_at"]]],
                                   ignore_index=True), t).iloc[:len(c)]
    rows = rows.join(hist.set_index(rows.index))
    rel = pd.to_datetime(custom["released_at"])
    rows["release_status"] = np.select([rel > t, rel > t - pd.Timedelta(days=NEW_RELEASE_DAYS)],
                                       ["preorder", "new"], "established")
    rows["releases_before_target"] = ((rel > t) & (rel <= t + pd.Timedelta(days=HORIZON))).astype(float)
    return rows
