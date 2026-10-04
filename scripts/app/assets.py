"""Moves the app's artifacts between this machine / CI / the API Space and Hugging Face.

  fetch    download what the pipeline or API needs into the repo layout:
             mtg-card-market-dataset/dataset.parquet   <- CXu0630/mtg-card-market-dataset
             models/deep_gp/                           <- CXu0630/mtg-price-deep-gp
             models/app/, data/app/                    <- CXu0630/mtg-price-app-assets (if published)
  publish  upload models/app/ (A/B/C, manifest, gp_static) + data/app/live_rows.parquet
           to CXu0630/mtg-price-app-assets (nightly, after train_models + score_all)

    python scripts/app/assets.py fetch [--no-app]     # --no-app: skip the app models (CI retrains them)
    python scripts/app/assets.py publish              # needs a write token (HF_TOKEN / hf auth login)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO = "CXu0630/mtg-card-market-dataset"
GP_REPO = "CXu0630/mtg-price-deep-gp"
ASSETS_REPO = "CXu0630/mtg-price-app-assets"
APP_FILES = {  # path in the assets repo -> local path
    "models/app/model_a.joblib": ROOT / "models/app/model_a.joblib",
    "models/app/model_b.joblib": ROOT / "models/app/model_b.joblib",
    "models/app/model_c.joblib": ROOT / "models/app/model_c.joblib",
    "models/app/manifest.json": ROOT / "models/app/manifest.json",
    "models/app/gp_static.parquet": ROOT / "models/app/gp_static.parquet",
    "data/app/live_rows.parquet": ROOT / "data/app/live_rows.parquet",
}


def fetch(with_app_models: bool) -> None:
    dataset = ROOT / "mtg-card-market-dataset" / "dataset.parquet"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(DATASET_REPO, "dataset.parquet", repo_type="dataset"), dataset)
    snapshot_download(GP_REPO, local_dir=ROOT / "models" / "deep_gp")
    wanted = list(APP_FILES) if with_app_models else ["models/app/gp_static.parquet"]
    for remote in wanted:
        local = APP_FILES[remote]
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(ASSETS_REPO, remote, repo_type="dataset"), local)
    print(f"fetched dataset, deep GP and {len(wanted)} app asset(s) into {ROOT}")


def publish() -> None:
    api = HfApi()
    api.create_repo(ASSETS_REPO, repo_type="dataset", private=False, exist_ok=True)
    missing = [str(p) for p in APP_FILES.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing} -- run train_models.py / build_gp_static.py first")
    readme = ROOT / "model_cards" / "app_assets.md"
    ops = [(str(local), remote) for remote, local in APP_FILES.items()] + [(str(readme), "README.md")]
    from huggingface_hub import CommitOperationAdd
    api.create_commit(ASSETS_REPO, repo_type="dataset", commit_message="Update app assets",
                      operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in ops])
    print(f"published {len(ops)} files -> https://huggingface.co/datasets/{ASSETS_REPO}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        fetch(with_app_models="--no-app" not in sys.argv)
    elif cmd == "publish":
        publish()
    else:
        raise SystemExit(__doc__)
