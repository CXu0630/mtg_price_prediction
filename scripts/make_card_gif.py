"""Animated GIF of a Magic card turning on a vertical axis (front, then the card back), for slides.

    python scripts/make_card_gif.py                                   # Sol Ring (C21), dark background
    python scripts/make_card_gif.py --bg transparent                  # transparent background
    python scripts/make_card_gif.py --id <scryfall_id> --out figures/my_card.gif --seconds 3

Card images come from Scryfall's PNG scans (transparent rounded corners). Each frame is a true perspective
projection of the card, drawn at 2x and downsampled for smooth edges.
"""
import argparse
import io
import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACK = "https://backs.scryfall.io/png/0/a/0aeebaf5-8c7d-4636-9e82-8c27447861f7.png"
SOL_RING = "4cbc6901-6a4a-4d0a-83ea-7eefa3b35021"  # Sol Ring (C21)


def fetch(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "mtg-price-prediction/0.1"})
    with urllib.request.urlopen(req) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGBA")


def front_url(scryfall_id: str) -> str:
    return f"https://cards.scryfall.io/png/front/{scryfall_id[0]}/{scryfall_id[1]}/{scryfall_id}.png"


def perspective_coeffs(dst, src):
    """Coefficients for Image.transform(PERSPECTIVE): maps each output point in `dst` to the input point in `src`."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    return np.linalg.solve(np.array(a, float), np.array(src, float).reshape(8))


def frame(front: Image.Image, back: Image.Image, deg: float, size: tuple[int, int], card_h: int,
          distance: float) -> Image.Image:
    w_out, h_out = size
    cw, ch = card_h * front.width / front.height, card_h
    t = math.radians(deg)
    cx, cy = w_out / 2, h_out / 2
    corners = []  # project the card's 4 corners (TL, TR, BR, BL) after turning it about the vertical axis
    for x, y in [(-cw / 2, -ch / 2), (cw / 2, -ch / 2), (cw / 2, ch / 2), (-cw / 2, ch / 2)]:
        xr, z = x * math.cos(t), x * math.sin(t)
        s = distance / (distance + z)
        corners.append((cx + xr * s, cy + y * s))
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    if abs(math.cos(t)) < 1e-3:
        return out  # edge-on
    showing_back = math.cos(t) < 0
    img = back if showing_back else front
    w, h = img.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    if showing_back:  # seen from behind: swap left and right so the back isn't mirrored
        src = [(w, 0), (0, 0), (0, h), (w, h)]
    coeffs = perspective_coeffs(corners, src)
    return img.transform(size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default=SOL_RING, help="Scryfall id of the card (default: Sol Ring, C21)")
    ap.add_argument("--bg", default="#0a0a0a", help="background colour, or 'transparent'")
    ap.add_argument("--seconds", type=float, default=3.6, help="time for one full turn")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--height", type=int, default=500, help="output height in px")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    front, back = fetch(front_url(a.id)), fetch(BACK)
    ss = 2  # supersample
    h = a.height
    w = round(h * 0.82)
    size = (w * ss, h * ss)
    card_h = round(h * 0.8) * ss
    n = round(a.seconds * a.fps)
    transparent = a.bg == "transparent"
    frames = []
    for i in range(n):
        f = frame(front, back, 360 * i / n, size, card_h, distance=card_h * 3.2).resize((w, h), Image.Resampling.LANCZOS)
        if transparent:
            alpha = f.getchannel("A").point(lambda v: 255 if v > 127 else 0)  # GIF transparency is on/off
            f = f.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=255)
            f.paste(255, mask=Image.eval(alpha, lambda v: 255 - v))
            f.info["transparency"] = 255
        else:
            bg = Image.new("RGBA", (w, h), a.bg)
            f = Image.alpha_composite(bg, f).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        frames.append(f)

    out = a.out or ROOT / "figures" / f"rotating_card{'_transparent' if transparent else ''}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"transparency": 255, "disposal": 2} if transparent else {}
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / a.fps), loop=0,
                   optimize=False, **extra)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB, {n} frames)")


if __name__ == "__main__":
    main()
