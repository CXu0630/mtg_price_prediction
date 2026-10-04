"""Comps' moves for the Model B rebuild: each printing's nearest *other cards*
(same-card printings excluded -- sib_* already covers those), and per snapshot
day their similarity-weighted 21-day move and the printing's price relative to
theirs, as of that day only.

Two neighbour spaces:
  gp   the published deep GP's joint embedding (training/model2.pt), divided by
       the RBF kernel's learned lengthscales -- the GP's own notion of similar
  raw  control: the same standardized wide/tabular features the GP is fed,
       PCA'd to 48 dims -- no learned similarity, no price target in the loop

Caveat (gp only): the embedding was fit to window-average prices that include
days after some snapshots, so neighbourhoods were shaped partly by later price
*levels*. The comps' *moves* are strictly as-of; the raw control bounds how
much the learned space adds.

    python scripts/model_b/comps.py   # -> training/model_b/comps_{gp,raw}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, for train_model2
from build_daily import BULK, LOOKBACK, OUT_DIR, despike, printings, wide_prices  # noqa: E402
from mtg_price_prediction.deep_gp import transform  # noqa: E402
from train_model2 import IMAGE_EMB_PATH, MODEL_OUT_PATH, TEXT_EMB_PATH, build_model, load_data  # noqa: E402

K, K_CANDIDATES = 20, 80


def embeddings() -> tuple[pd.Series, dict[str, np.ndarray]]:
    ckpt = torch.load(MODEL_OUT_PATH, weights_only=False)
    config = ckpt["config"]
    data = load_data(seed=config.seed, features_path=config.features_path)
    model, _ = build_model(data, config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    feats = pd.read_parquet(config.features_path)
    cat, wide = transform(feats, data["prep"])
    text, image = np.load(TEXT_EMB_PATH), np.load(IMAGE_EMB_PATH)
    with torch.no_grad():
        z = torch.cat([model.feature_extractor(torch.from_numpy(cat[s:s + 8192]), torch.from_numpy(wide[s:s + 8192]),
                                               torch.from_numpy(text[s:s + 8192]), torch.from_numpy(image[s:s + 8192]))
                       for s in range(0, len(feats), 8192)]).numpy()
    lengthscale = model.gp_layer.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
    spaces = {"gp": z / lengthscale, "raw": PCA(n_components=48, random_state=0).fit_transform(wide)}
    return feats["__printing_key"], spaces


def neighbours(space: np.ndarray, oracle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """K nearest rows with a different oracle_id, and RBF weights exp(-d^2 / 2h^2),
    h = median neighbour distance (so the two spaces are weighted comparably)."""
    dist, idx = NearestNeighbors(n_neighbors=K_CANDIDATES).fit(space).kneighbors(space)
    other = oracle[idx] != oracle[:, None]
    order = np.argsort(~other, axis=1, kind="stable")[:, :K]  # first K other-card candidates, distance order kept
    idx, dist, valid = (np.take_along_axis(a, order, axis=1) for a in (idx, dist, other))
    h = np.median(dist[valid])
    w = np.exp(-0.5 * (dist / h) ** 2) * valid
    print(f"  {valid.all(axis=1).mean():.1%} of printings have {K} other-card neighbours; h = {h:.3f}", flush=True)
    return idx, w


def comps_features(nbr: np.ndarray, w: np.ndarray, price: pd.DataFrame, days, keys: pd.Series) -> pd.DataFrame:
    frames = []
    for t in days:
        p = price[t].to_numpy()
        start = price[t - pd.Timedelta(days=LOOKBACK)].to_numpy()
        chg = np.clip(100 * (p / start - 1), -90, 300)
        chg[~(p >= BULK)] = np.nan
        lp = np.log(p)
        out = {"printing_key": keys.to_numpy(), "snapshot_date": t}
        for name, v in (("comps_chg", chg), ("comps_level", lp)):
            vals = v[nbr]
            ww = np.where(np.isnan(vals), 0.0, w)
            out[name] = np.where(ww.sum(1) > 0, np.nansum(ww * np.nan_to_num(vals), 1) / np.maximum(ww.sum(1), 1e-12), np.nan)
            if name == "comps_chg":
                out["comps_n"] = (ww > 0).sum(1)
        f = pd.DataFrame(out)
        f["comps_rel_level"] = lp - f.pop("comps_level")
        frames.append(f[p >= BULK])
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    daily_days = pd.read_parquet(OUT_DIR / "daily.parquet", columns=["snapshot_date"])["snapshot_date"].unique()
    keys, spaces = embeddings()
    ds = printings().set_index("printing_key")
    keep = keys.isin(ds.index).to_numpy()  # basic lands / duplicate foil rows dropped in the daily universe
    keys = keys[keep].reset_index(drop=True)
    ds = ds.loc[keys].reset_index()
    price = despike(wide_prices("tcgplayer", "retail", ds), causal=True)
    for name, space in spaces.items():
        print(f"== {name} space ==", flush=True)
        nbr, w = neighbours(space[keep], ds["oracle_id"].to_numpy())
        out = comps_features(nbr, w, price, sorted(daily_days), keys)
        path = OUT_DIR / f"comps_{name}.parquet"
        out.to_parquet(path, index=False)
        print(f"  {len(out):,} rows -> {path}; comps_chg coverage {out['comps_chg'].notna().mean():.1%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
