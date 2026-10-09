"""Uploads the exported deep GP (scripts/export_models.py -> models/deep_gp/) to Hugging Face
as the public model repo CXu0630/mtg-price-deep-gp. Needs `hf auth login` with write access.

    python scripts/push_models_to_hub.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
REPOS = {
    "deep_gp": "CXu0630/mtg-price-deep-gp",
}


def main(which: list[str]) -> int:
    api = HfApi()
    for name in which:
        folder = ROOT / "models" / name
        if not (folder / "README.md").exists():
            raise SystemExit(f"{folder} missing -- run scripts/export_models.py first")
        repo_id = REPOS[name]
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        api.update_repo_settings(repo_id, private=False)
        api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=f"Upload {name} model",
                          ignore_patterns=["**/__pycache__/**", "*.pyc", ".DS_Store"])
        print(f"{name} -> https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or list(REPOS)))
