"""Compose the ad: recorded site shots + a synthetic Sol Ring intro and logo end card, cheesy zooms, pops, flashes
and text callouts, timed to the voiceover (vo_timing.json), mixed with generated sound effects.

    python ad/compose.py                 # -> ad/mtg_price_intelligence_ad.mp4
    python ad/compose.py --music bed.mp3 # optional background track, ducked under the voice
    python ad/compose.py --preview 9.5 14  # only render a window (fast check)
"""
import argparse
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

AD = Path(__file__).resolve().parent
ROOT = AD.parent
sys.path.insert(0, str(ROOT / "scripts"))
from make_card_gif import BACK, SOL_RING, fetch, front_url, frame as card_frame  # noqa: E402

W, H, FPS = 2560, 1440, 30
K = W / 1920  # layout below is written in 1920x1080 units and scaled by K
BG = (10, 10, 10)
PAUSES = (20.83, 21.67, 27.75)  # half-second breaks inserted in the voiceover: after "list,", "predict,", "hypotheticals."
PAUSE = 0.5
END = 32.4 + PAUSE * len(PAUSES)
VO = next(AD.glob("ElevenLabs_*.mp3"))
SERIF = str(AD / "fonts" / "InstrumentSerif-Regular.ttf")
SANS = str(AD / "fonts" / "InstrumentSans-Variable.ttf")
MONO = str(AD / "fonts" / "IBMPlexMono-Medium.ttf")


# ---------------------------------------------------------------- helpers
def ease(x):  # ease-in-out cubic on [0, 1]
    x = min(max(x, 0.0), 1.0)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def keyed(keys, t):
    """Interpolate (time, value...) keyframes with ease-in-out."""
    if t <= keys[0][0]:
        return keys[0][1:]
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if t <= t1:
            k = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1
            return tuple(x + (y - x) * k for x, y in zip(a, b))
    return keys[-1][1:]


def zoom(img: Image.Image, scale, cx, cy):
    if scale <= 1.0001:
        return img
    cx, cy = cx * K, cy * K
    w, h = W / scale, H / scale
    x0 = min(max(cx - w / 2, 0), W - w)
    y0 = min(max(cy - h / 2, 0), H - h)
    return img.resize((W, H), Image.Resampling.BICUBIC, box=(x0, y0, x0 + w, y0 + h))


def font(path, size, weight=None):
    f = ImageFont.truetype(path, size)
    if weight:
        try:
            f.set_variation_by_axes([100, weight])  # [wdth, wght]
        except Exception:
            pass
    return f


@lru_cache(maxsize=None)
def text_sprite(text, kind="title", size=110):
    """Pre-rendered RGBA text: 'title' (serif, white), 'chip' (white pill, black sans), 'mono' (small caps label)."""
    size = round(size * K)
    if kind == "title":
        f = font(SERIF, size)
        l, t, r, b = f.getbbox(text)
        pad = size // 3
        img = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).text((pad - l, pad - t), text, font=f, fill=(0, 0, 0, 220))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(size / 8)))
        d.text((pad - l, pad - t), text, font=f, fill=(242, 242, 242, 255))
        return img
    if kind == "chip":
        f = font(SANS, size, 600)
        l, t, r, b = f.getbbox(text)
        px, py = int(size * 0.75), int(size * 0.45)
        img = Image.new("RGBA", (r - l + 2 * px, b - t + 2 * py + 6), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, img.width - 1, img.height - 7), radius=int(size * 0.5), fill=(242, 242, 242, 255))
        d.text((px - l, py - t), text, font=f, fill=(10, 10, 10, 255))
        return img
    f = font(MONO, size)
    spaced = " ".join(text.upper())
    l, t, r, b = f.getbbox(spaced)
    img = Image.new("RGBA", (r - l + 8, b - t + 8), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((4 - l, 4 - t), spaced, font=f, fill=(163, 163, 163, 255))
    return img


def pop(frame: Image.Image, sprite: Image.Image, cx, cy, t, start, end=None, rot=0.0):
    """Paste a sprite with a springy pop-in (0.6 -> 1.12 -> 1.0) and a quick fade-out at `end`."""
    if t < start or (end is not None and t > end + 0.2):
        return
    k = t - start
    s = 0.6 + 0.52 * ease_out(k / 0.16) if k < 0.16 else 1.12 - 0.12 * ease_out((k - 0.16) / 0.18)
    a = min(1.0, k / 0.08)
    if end is not None and t > end:
        a *= 1 - (t - end) / 0.2
        s *= 1 + 0.1 * (t - end) / 0.2
    cx, cy = cx * K, cy * K
    sp = sprite.resize((max(1, int(sprite.width * s)), max(1, int(sprite.height * s))), Image.Resampling.BICUBIC)
    if rot:
        sp = sp.rotate(rot, resample=Image.Resampling.BICUBIC, expand=True)
    if a < 1:
        sp.putalpha(sp.getchannel("A").point(lambda v: int(v * a)))
    frame.alpha_composite(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)))


def flash(frame: Image.Image, t, at, strength=0.75, dur=0.18):
    if at <= t < at + dur:
        a = strength * (1 - (t - at) / dur)
        frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(255 * a))))


# ---------------------------------------------------------------- sources
class Shot:
    def __init__(self, name):
        m = json.loads((AD / "build" / name / "frames.json").read_text())
        self.dir = AD / "build" / name
        self.ts = np.array(m["timestamps"]) - m["t0_wall"]

    @lru_cache(maxsize=8)
    def _load(self, i):
        return Image.open(self.dir / f"{i:05d}.jpg").convert("RGBA")

    def at(self, t):
        i = int(np.searchsorted(self.ts, t, side="right")) - 1
        return self._load(max(i, 0)).copy()


FRONT = BACKIMG = LOGO = TREASURE_IMG = None
TREASURE_CUT = 24.40  # on "treasure" (23.46 in the read, +1.0 s of pauses)
CUSTOM_AT = 26.21  # "Even new cards ..." (25.21 + 1.0)
OUTRO_AT = 29.48  # "MTG Price Intelligence." (27.98 + 1.5)
TREASURE_ID = "46a2c363-c879-4c36-b08a-dd57e3da19ae"  # Treasure token (TLTR #12), the site's prediction card


def card_angle(t):
    """Sol Ring intro: a slow turn with two fast spins, on 'worth?' (1.82) and 'cards?' (4.32)."""
    a = 160 + 55 * t
    for t0, turns in ((1.70, 2), (4.20, 2)):
        if t > t0:
            a += 360 * turns * ease_out((t - t0) / 0.9)
    return a


def intro(t):
    img = Image.new("RGBA", (W, H), BG + (255,))
    scale_in = 0.55 + 0.45 * ease_out(t / 0.6)  # zooms up out of the dark
    card_h = int(820 * K * scale_in * (1 + 0.04 * math.sin(t * 2)))
    f = card_frame(FRONT, BACKIMG, card_angle(t), (W, H), card_h, distance=card_h * 3.2)
    img.alpha_composite(f)
    pop(img, text_sprite("What's it", "title", 120), 470, 420, t, 0.40, 2.30)
    pop(img, text_sprite("really worth?", "title", 120), 470, 560, t, 1.40, 2.30, rot=4)
    pop(img, text_sprite("Buy.", "title", 150), 1470, 380, t, 3.55, 5.05, rot=-5)
    pop(img, text_sprite("Sell.", "title", 150), 1470, 540, t, 4.05, 5.05, rot=3)
    pop(img, text_sprite("Profit?", "title", 150), 1470, 700, t, 4.45, 5.05, rot=-4)
    return img


def coin_sprite(r):
    """A gold coin face (rim, face, highlight), drawn at 2x and reduced."""
    big = Image.new("RGBA", (4 * r, 4 * r), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    d.ellipse((0, 0, 4 * r - 1, 4 * r - 1), fill=(150, 104, 24, 255))
    d.ellipse((r // 2, r // 2, 4 * r - r // 2, 4 * r - r // 2), fill=(232, 182, 62, 255))
    d.ellipse((r, r, 3 * r, 3 * r), outline=(176, 126, 30, 255), width=max(2, r // 4))
    d.ellipse((int(1.1 * r), int(0.9 * r), int(1.9 * r), int(1.5 * r)), fill=(255, 236, 160, 200))
    return big.resize((2 * r, 2 * r), Image.Resampling.LANCZOS)


COIN = None
COINS = None


def coins(t, front):
    """Falling coins that flip as they fall; `front` picks the half drawn in front of the card."""
    global COIN, COINS
    if COINS is None:
        COIN = {r: coin_sprite(r) for r in (18, 24, 30, 38)}
        rng = np.random.default_rng(3)
        COINS = [dict(x=rng.uniform(-0.05, 1.05) * W, r=int(rng.choice([18, 24, 30, 38])), delay=rng.uniform(0, 1.3),
                      vy=rng.uniform(900, 1500) * K, spin=rng.uniform(6, 14), ph=rng.uniform(0, 6.3),
                      drift=rng.uniform(-80, 80) * K, front=bool(rng.random() < 0.55)) for _ in range(90)]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for c in COINS:
        if c["front"] != front or t < c["delay"]:
            continue
        k = t - c["delay"]
        y = -60 * K + c["vy"] * k + 300 * K * k * k
        if y > H + 80:
            continue
        x = c["x"] + c["drift"] * k
        sp = COIN[c["r"]]
        w = max(2, int(sp.width * K * abs(math.cos(c["ph"] + c["spin"] * k))))  # flipping edge-on and back
        img = sp.resize((w, int(sp.height * K)), Image.Resampling.BILINEAR)
        layer.alpha_composite(img, (int(x - w / 2), int(y)))
    return layer


def treasure(t):
    """The Treasure token turning like the intro's Sol Ring (enters on a fast spin), in a shower of coins."""
    img = Image.new("RGBA", (W, H), BG + (255,))
    img.alpha_composite(coins(t, front=False))
    card_h = int(820 * K)
    a = 720 * ease_out(t / 0.9) - 720 + 25 * t  # two fast turns that land face-up, then a gentle drift
    img.alpha_composite(card_frame(TREASURE_IMG, BACKIMG, a, (W, H), card_h, distance=card_h * 3.2))
    img.alpha_composite(coins(t, front=True))
    pop(img, text_sprite("Treasure!", "title", 170), 1460, 300, t, 0.10, None, rot=-6)
    return img


def outro(t):  # t from OUTRO_AT
    img = Image.new("RGBA", (W, H), BG + (255,))
    k = ease_out(t / 0.5)
    lg = LOGO.resize((int(220 * K * (0.7 + 0.3 * k)),) * 2, Image.Resampling.LANCZOS)
    lg.putalpha(lg.getchannel("A").point(lambda v: int(v * k)))
    img.alpha_composite(lg, (W // 2 - lg.width // 2, int(300 * K) - lg.height // 2))
    pop(img, text_sprite("MTG Price Intelligence", "title", 150), 960, 560, t, 0.10)
    pop(img, text_sprite("Know what your cards can buy you.", "chip", 46), 960, 760, t, 1.95)
    pop(img, text_sprite("cxu0630.github.io/mtg_price_prediction", "mono", 22), 960, 880, t, 2.6)
    return img


SHOTS = {}


def live(name, src_t, keys, overlays=()):
    """A recorded shot at its own time `src_t`, zoomed per `keys` [(t, scale, cx, cy)], with overlay callbacks."""
    img = SHOTS[name].at(src_t)
    s, cx, cy = keyed(keys, src_t)
    img = zoom(img, s, cx, cy)
    for f in overlays:
        f(img, src_t)
    return img


def render(t):
    if t < 5.20:
        img = intro(t)
    elif t < 9.72:  # title turns over on "MTG" (5.24): the intro's solo spin ends 0.9 s after load
        u = t - 5.20 + 1.03  # the title first shows 1.07 s into the recording (app mount + 0.9 s solo spin)
        img = live("load", u, [(1.03, 1.8, 1245, 150), (1.5, 1.8, 1245, 150), (2.8, 1.0, 960, 540)], [
            lambda im, u: pop(im, text_sprite("Powered by AI", "chip", 54), 1560, 940, u, 1.03 + 3.50, 1.03 + 4.45, rot=-3),
        ])
    elif t < 15.06:  # Predict on "forecast" (9.92); summary turns over; the Treasure spins on "gold" (13.98)
        u = t - 9.72
        img = live("predict", u, [(0.0, 1.0, 960, 540), (1.4, 1.0, 960, 540), (2.0, 1.7, 1330, 165), (3.4, 1.7, 1330, 165),
                                  (4.1, 1.0, 960, 540)], [  # then back out to the whole view
            lambda im, u: pop(im, text_sprite("14-day forecast", "chip", 54), 1300, 640, u, 2.1, 3.4, rot=-3),
        ])
    elif t < 19.83:  # a card's details; Deep GP on "Deep Kernel" (17.64) -- no zoom from here on
        u = t - 15.06
        img = live("focus", u, [(0.0, 1.0, 960, 540)], [
            lambda im, u: pop(im, text_sprite("Gradient-boosted trees", "chip", 46), 1560, 250, u, 1.20, 2.5, rot=-3),
            lambda im, u: pop(im, text_sprite("Deep kernel Gaussian process", "chip", 46), 1520, 330, u, 2.58, 4.6, rot=2),
        ])
    elif t < TREASURE_CUT:  # add Sol Ring, press Predict (21.16), every card turns over to its price
        u = t - 19.83
        img = live("add", u, [(0.0, 1.0, 960, 540)], [
            lambda im, u: pop(im, text_sprite("PREDICT", "chip", 64), 1600, 420, u, 1.83, 2.8, rot=-4),
        ])
    elif t < CUSTOM_AT:  # "... into treasure for your trove": a spinning Treasure token in a rain of coins
        img = treasure(t - TREASURE_CUT)
    elif t < OUTRO_AT:  # "Even new cards or your own hypotheticals": the open form fills itself in
        u = t - CUSTOM_AT
        img = live("custom", u, [(0.0, 1.0, 960, 540)], [
            lambda im, u: pop(im, text_sprite("New cards", "chip", 54), 640, 840, u, 0.15, 1.30, rot=-3),
            lambda im, u: pop(im, text_sprite("Your own hypotheticals", "chip", 54), 640, 840, u, 1.39, 2.70, rot=3),
        ])
    else:
        img = outro(t - OUTRO_AT)
    for cut in (5.20, 9.72, 15.06, 19.83, TREASURE_CUT, CUSTOM_AT, OUTRO_AT):  # a white pop on every cut
        flash(img, t, cut)
    flash(img, t, 21.66, 0.35, 0.12)  # "predict"
    if t > END - 0.5:  # fade out
        img.alpha_composite(Image.new("RGBA", (W, H), BG + (int(255 * (t - END + 0.5) / 0.5),)))
    return img.convert("RGB")


# ---------------------------------------------------------------- audio
SR = 44100


def sfx_track(n):
    rng = np.random.default_rng(7)
    out = np.zeros(n, np.float32)

    def add(sig, at, gain):
        i = int(at * SR)
        j = min(n, i + len(sig))
        if i < n:
            out[i:j] += gain * sig[: j - i]

    def whoosh(dur=0.42):
        m = int(dur * SR)
        noise = rng.standard_normal(m)
        spec = np.fft.rfft(noise)
        f = np.fft.rfftfreq(m, 1 / SR)
        spec *= np.exp(-((np.log(f + 1) - np.log(1800)) ** 2) / 1.2)
        s = np.fft.irfft(spec, m)
        env = np.sin(np.linspace(0, np.pi, m)) ** 2
        return (s * env / (np.abs(s).max() + 1e-9)).astype(np.float32)

    def blip(f0=950, f1=320, dur=0.09):
        m = int(dur * SR)
        tt = np.arange(m) / SR
        ph = 2 * np.pi * np.cumsum(np.linspace(f0, f1, m)) / SR
        return (np.sin(ph) * np.exp(-tt / 0.025)).astype(np.float32)

    def clink(f0, dur=0.35):
        m = int(dur * SR)
        tt = np.arange(m) / SR
        partials = sum(np.sin(2 * np.pi * f0 * r * tt + rng.uniform(0, 6)) * g for r, g in ((1, 1), (2.76, 0.5), (5.4, 0.25)))
        return (partials * np.exp(-tt / 0.06)).astype(np.float32)

    def boom(dur=0.9):
        m = int(dur * SR)
        tt = np.arange(m) / SR
        return (np.sin(2 * np.pi * (55 - 20 * tt) * tt) * np.exp(-tt / 0.25)).astype(np.float32)

    for at in (5.20, 9.72, 15.06, 19.83, TREASURE_CUT, CUSTOM_AT):
        add(whoosh(), at - 0.18, 0.28)
    for at in (1.70, 4.20):  # card spins
        add(whoosh(0.7), at, 0.22)
    for at in (0.40, 1.40, 3.55, 4.05, 4.45, 9.72 + 2.1, 15.06 + 1.2, 15.06 + 2.58, 19.83 + 1.83, TREASURE_CUT + 0.08,
               CUSTOM_AT + 0.15, CUSTOM_AT + 1.39, OUTRO_AT + 0.1, OUTRO_AT + 1.95):
        add(blip(), at, 0.32)
    add(boom(), OUTRO_AT, 0.5)
    for k in range(26):  # coins landing: bright metallic clinks, scattered through the treasure shot
        at = TREASURE_CUT + 0.15 + rng.uniform(0, 1.6)
        add(clink(rng.uniform(2600, 4200)), at, rng.uniform(0.05, 0.12))
    return out


def audio(path_out, music=None):
    def decode(p):
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32)

    n = int(END * SR)
    mix = np.zeros(n, np.float32)
    vo = decode(VO)
    for at in sorted(PAUSES, reverse=True):  # splice in the pauses (latest first, so earlier positions hold)
        i = int(at * SR)
        vo = np.concatenate([vo[:i], np.zeros(int(PAUSE * SR), np.float32), vo[i:]])
    vo = vo[:n]
    mix[: len(vo)] += vo
    mix += sfx_track(n)
    if music:
        m = np.resize(decode(music), n)
        env = np.abs(np.pad(vo, (0, n - len(vo))))  # duck the music under the voice
        k = int(0.25 * SR)
        env = np.convolve(env, np.ones(k) / k, mode="same")
        duck = 1 - 0.6 * np.clip(env / (env.max() + 1e-9) * 4, 0, 1)
        mix += 0.16 * m * duck
    fade = int(0.5 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", str(path_out)],
                   input=mix.astype(np.float32).tobytes(), check=True)


# ---------------------------------------------------------------- main
def main():
    global FRONT, BACKIMG, LOGO, TREASURE_IMG
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", type=Path)
    ap.add_argument("--preview", nargs=2, type=float, help="render only this window, in seconds")
    ap.add_argument("--out", type=Path, default=AD / "mtg_price_intelligence_ad.mp4")
    a = ap.parse_args()

    FRONT, BACKIMG = fetch(front_url(SOL_RING)), fetch(BACK)
    TREASURE_IMG = fetch(front_url(TREASURE_ID))
    LOGO = Image.open(AD / "build" / "logo.png").convert("RGBA")
    for name in ("load", "predict", "focus", "add", "custom"):
        SHOTS[name] = Shot(name)

    t0, t1 = a.preview or (0.0, END)
    wav = AD / "build" / "mix.wav"
    audio(wav, a.music)
    out = a.out if not a.preview else AD / "build" / "preview.mp4"
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-ss", str(t0), "-t", str(t1 - t0), "-i", str(wav),
                            "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    n = int(round((t1 - t0) * FPS))
    for i in range(n):
        enc.stdin.write(render(t0 + i / FPS).tobytes())
        if i % 60 == 0:
            print(f"  {t0 + i / FPS:5.1f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
