"""Presentation GIFs from recorded site shots (record.py): play once, then hold on the final state.

    python ad/make_gifs.py   # -> ad/gifs/01_grid_load.gif ... 07_card_details.gif

Each GIF runs from `start` until the page has settled, then on to the moment (within `window`) when the
spinning showcase card on the left is most face-on, so the last frame looks finished.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

AD = Path(__file__).resolve().parent
OUT = AD / "gifs"
SIZE, FPS = (1280, 720), 20

GIFS = [  # (file, shot, start, settled by, latest end) -- one story, each GIF starting where the last one ends
    ("01_grid_load.gif", "load", 0.0, 2.5, 4.4),
    ("02_add_card_search.gif", "gif_add_search", 0.0, 2.9, 3.4),
    ("03_add_card_submit.gif", "gif_add_submit", 0.0, 1.9, 2.6),
    ("04_custom_card_fill.gif", "gif_custom_fill", 0.0, 5.3, 5.6),
    ("05_custom_card_submit.gif", "gif_custom_submit", 0.0, 2.1, 2.8),
    ("06_predict.gif", "gif_predict", 0.0, 2.6, 3.4),
    ("07_card_details.gif", "gif_details", 0.0, 2.3, 3.6),
]


def frames(shot):
    m = json.loads((AD / "build" / shot / "frames.json").read_text())
    return np.array(m["timestamps"]) - m["t0_wall"]


def at(shot, ts, t):
    i = max(int(np.searchsorted(ts, t, side="right")) - 1, 0)
    return Image.open(AD / "build" / shot / f"{i:05d}.jpg").convert("RGB")


def card_width(img):
    """How wide the showcase card looks: columns in the left third that aren't page background."""
    a = np.asarray(img.convert("L").crop((0, img.height // 3, img.width // 3, 2 * img.height // 3)), float)
    return int((a.max(axis=0) > 40).sum())


def make(name, shot, start, settled, latest):
    ts = frames(shot)
    cands = np.arange(settled, latest + 1e-9, 1 / FPS)
    end = max(cands, key=lambda t: card_width(at(shot, ts, t)))
    n = int(round((end - start) * FPS)) + 1
    OUT.mkdir(exist_ok=True)
    out = OUT / name
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE[0]}x{SIZE[1]}",
                           "-r", str(FPS), "-i", "-", "-vf",
                           "split[a][b];[a]palettegen=stats_mode=full:max_colors=256[p];[b][p]paletteuse=dither=sierra2_4a",
                           "-loop", "-1", str(out)], stdin=subprocess.PIPE)  # -loop -1: play once
    for k in range(n):
        ff.stdin.write(at(shot, ts, start + k / FPS).resize(SIZE, Image.Resampling.LANCZOS).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{name}: {end - start:.2f}s, {n} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    for g in GIFS:
        make(*g)
