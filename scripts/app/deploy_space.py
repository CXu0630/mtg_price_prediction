"""Pushes the custom-card API to a Hugging Face Docker Space: the Dockerfile at the
Space root, plus the source it builds from. The Space then pulls the nightly assets
(scripts/app/assets.py fetch) whenever it starts.

    python scripts/app/deploy_space.py CXu0630/mtg-price-api [--private]
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parents[2]
FILES = ["pyproject.toml", "README.md", "scripts/app/assets.py", *[
    str(p.relative_to(ROOT)) for d in ("src/mtg_price_prediction", "api", "requirements")
    for p in (ROOT / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]]
SPACE_README = """---
title: MTG Price API
emoji: 🃏
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

Custom-card pricing API for the [MTG price-range app](https://github.com/CXu0630/mtg_price_prediction)
(`api/app.py`). `GET /health`, `POST /predict`. Unofficial fan project; Magic: The Gathering is a trademark of
Wizards of the Coast.
"""


def main(space_id: str, private: bool) -> int:
    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)) for f in FILES if f != "README.md"]
    ops += [CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=str(ROOT / "api" / "Dockerfile")),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=SPACE_README.encode())]
    api.create_commit(space_id, repo_type="space", operations=ops, commit_message="Deploy API")
    print(f"deployed {len(ops)} files -> https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], "--private" in sys.argv))
