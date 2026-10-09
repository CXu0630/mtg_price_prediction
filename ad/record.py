"""Record the ad's live-site shots from the local preview (http://localhost:4173) with headless Chrome.

    python ad/record.py            # every shot -> ad/build/<shot>/NNNNN.jpg + frames.json (timestamps)
    python ad/record.py predict    # just one

Each shot loads the app with a seeded card list, runs scripted clicks at fixed times, and captures a real-time
screencast (Chrome sends a frame whenever the page repaints; compose.py resamples to a constant 30 fps).
Action times are relative to the shot's start and are chosen to land on words in vo_timing.json.
"""
import base64
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import websocket

AD = Path(__file__).resolve().parent
BUILD = AD / "build"
APP = "http://localhost:4173/"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 2560, 1440          # output pixels
CSS_W, CSS_H, DPR = 2560, 1440, 1  # a 2560x1440 (QHD) window, rendered 1:1

KEY = {  # printing keys (scryfall_id:finish), all with live forecasts
    "ragavan": ("a9738cda-adb1-47fb-9f4c-ecd930228c4d:nonfoil", "mh2", "Ragavan, Nimble Pilferer"),
    "ring": ("10a4e202-b564-4ac8-8bdf-a2a563662563:nonfoil", "ltr", "The One Ring"),
    "nazgul": ("8c20df77-2a4d-44a6-9389-125c4e5f5e40:nonfoil", "ltr", "Nazgûl"),
    "sheoldred": ("8df6603a-38c1-4d18-8b84-6211e9a7cc09:nonfoil", "dmu", "Sheoldred, the Apocalypse"),
    "watcher": ("f584f9a4-4ef1-463e-8aa3-a21b4f820e2a:nonfoil", "ltr", "The Watcher in the Water"),
    "jace": ("0e606072-a3aa-4300-ba90-ec92a721fa76:nonfoil", "wwk", "Jace, the Mind Sculptor"),
}
CUSTOM = {"name": "Gilded Wurm", "rarity": "mythic", "type_line": "Creature — Wurm", "mana_cost": "{4}{G}{G}",
          "oracle_text": "Trample\nWhen Gilded Wurm enters, create three Treasure tokens.", "power": "7", "toughness": "7",
          "finish": "foil", "set_type": "expansion", "frame": "2015", "border_color": "borderless",
          "frame_effects": ["showcase"], "demand": "popular"}


def items(*names):
    out = []
    for i, n in enumerate(names):
        k, s, name = KEY[n]
        out.append({"id": f"ad{i}", "kind": "existing", "printing_key": k, "set": s, "name": name, "quantity": 1})
    return out


class CDP:
    """Minimal Chrome DevTools client: calls block on their reply; screencast frames are acked and kept."""

    def __init__(self, url):
        self.ws = websocket.create_connection(url, suppress_origin=True, max_size=None)
        self.n, self.replies, self.frames, self.recording = 0, {}, [], False
        self.lock = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            try:
                m = json.loads(self.ws.recv())
            except Exception:
                return
            if m.get("method") == "Page.screencastFrame":
                p = m["params"]
                if self.recording:
                    self.frames.append((p["metadata"]["timestamp"], p["data"]))
                self.send("Page.screencastFrameAck", sessionId=p["sessionId"])
            elif "id" in m:
                with self.lock:
                    self.replies[m["id"]] = m
                    self.lock.notify_all()

    def send(self, method, **params):
        with self.lock:
            self.n += 1
            i = self.n
        self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        return i

    def call(self, method, **params):
        i = self.send(method, **params)
        with self.lock:
            while i not in self.replies:
                self.lock.wait(10)
            return self.replies.pop(i).get("result", {})

    def js(self, expr):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True).get("result", {}).get("value")


def click(sel_js):  # JS that clicks the element found by an expression
    return f"(()=>{{const e={sel_js}; if(e) e.click(); return !!e}})()"


def button(text):
    return f"[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==={json.dumps(text)})"


def type_into(sel, text):
    return (f"(()=>{{const i=document.querySelector({json.dumps(sel)}); const set=Object.getOwnPropertyDescriptor("
            f"HTMLInputElement.prototype,'value').set; set.call(i,{json.dumps(text)}); i.dispatchEvent(new Event('input',"
            f"{{bubbles:true}})); return true}})()")


def fill(kind, index, text, t_start, t_end):
    """Actions that type `text` into the custom form's `index`-th input (or textarea) between two times."""
    proto = "HTMLTextAreaElement" if kind == "textarea" else "HTMLInputElement"
    sel = "textarea" if kind == "textarea" else "input:not([type=checkbox]):not([type=date])"
    steps = max(1, len(text))
    out = []
    for k in range(1, steps + 1):
        part = text[:k]
        js = (f"(()=>{{const i=document.querySelectorAll('.custom-sheet {sel}')[{index}]; if(!i) return false; "
              f"Object.getOwnPropertyDescriptor({proto}.prototype,'value').set.call(i,{json.dumps(part)}); "
              f"i.dispatchEvent(new Event('input',{{bubbles:true}})); return true}})()")
        out.append((t_start + (t_end - t_start) * k / steps, js))
    return out


def typing(sel, text, t_start, t_end):
    """Actions that type `text` into an input one character at a time."""
    return [(t_start + (t_end - t_start) * k / len(text), type_into(sel, text[:k])) for k in range(1, len(text) + 1)]


SOL_RING = {"id": "ad6", "kind": "existing", "printing_key": "4cbc6901-6a4a-4d0a-83ea-7eefa3b35021:nonfoil", "set": "c21",
            "name": "Sol Ring", "quantity": 1}
WURM = {"id": "ad7", "kind": "custom", "quantity": 1, "card": {  # exactly what the form produces in the custom GIFs
    **CUSTOM, "oracle_text": "Trample. When Gilded Wurm enters, create three Treasure tokens.", "rarity": "mythic",
    "finish": "nonfoil", "border_color": "black", "frame_effects": ["showcase"], "demand": "playable"}}
SIX = ("ragavan", "ring", "nazgul", "sheoldred", "watcher", "jace")


def search_fill(t0, fast=False):
    """Open the search side, type Sol Ring, pick the Commander 2021 printing (its only treatment fills in)."""
    k = 0.0 if fast else 1.0
    return [
        (t0, click("document.querySelector('.add-slot')")),
        *(typing("#card-search", "Sol Ring", t0 + 0.6, t0 + 1.4) if not fast else [(t0 + 0.02, type_into("#card-search", "Sol Ring"))]),
        (t0 + 1.7 * k + 0.03, click("document.querySelectorAll('.panel-search .dropdown-button')[0]")),
        (t0 + 2.3 * k + 0.04, click("[...document.querySelectorAll('.panel-search .dropdown-list button')].find(b=>b.textContent.includes('(C21)'))")),
    ]


def custom_fill(t0, fast=False):
    """From the search side: open the custom form, then fill it in."""
    if fast:  # set-up only: each field gets its whole value in one step (one keystroke per frame would truncate)
        return [(t0, click(button("+ Custom card"))), fill("input", 0, "Gilded Wurm", t0, t0 + 1.2)[-1],
                fill("input", 1, "{4}{G}{G}", t0, t0 + 1.21)[-1], fill("input", 2, "Creature — Wurm", t0, t0 + 1.22)[-1],
                fill("textarea", 0, WURM["card"]["oracle_text"], t0, t0 + 1.23)[-1], fill("input", 3, "7", t0, t0 + 1.24)[-1],
                fill("input", 4, "7", t0, t0 + 1.25)[-1],
                (t0 + 1.26, click("document.querySelectorAll('.custom-sheet .dropdown-button')[0]")),
                (t0 + 1.30, click("[...document.querySelectorAll('.custom-sheet .dropdown-list button')].find(b=>b.textContent.trim()==='mythic')")),
                (t0 + 1.32, click("document.querySelectorAll('.custom-sheet .chip input')[0]"))]
    return [
        (t0, click(button("+ Custom card"))),
        *fill("input", 0, "Gilded Wurm", t0 + 1.3, t0 + 1.8),
        *fill("input", 1, "{4}{G}{G}", t0 + 1.9, t0 + 2.2),
        *fill("input", 2, "Creature — Wurm", t0 + 2.3, t0 + 2.7),
        *fill("textarea", 0, WURM["card"]["oracle_text"], t0 + 2.8, t0 + 3.8),
        *fill("input", 3, "7", t0 + 3.9, t0 + 3.91),
        *fill("input", 4, "7", t0 + 4.05, t0 + 4.06),
        (t0 + 4.25, click("document.querySelectorAll('.custom-sheet .dropdown-button')[0]")),
        (t0 + 4.55, click("[...document.querySelectorAll('.custom-sheet .dropdown-list button')].find(b=>b.textContent.trim()==='mythic')")),
        (t0 + 4.8, click("document.querySelectorAll('.custom-sheet .chip input')[0]")),
    ]


# shot name -> (seed list, settle-before-recording seconds or 'load', duration, [(t, js), ...])
SHOTS = {
    # 5.20 title turns over on "MTG" (5.24); the intro's solo spin is 0.9 s, so the recording starts at the load
    "load": (items("ragavan", "ring", "nazgul", "sheoldred", "watcher", "jace"), "load", 6.0, []),
    # 9.72 "We forecast ..." -> Predict on "forecast" (9.92)
    "predict": (items("ragavan", "ring", "nazgul", "sheoldred", "watcher", "jace"), 3.5, 5.6, [
        (0.20, click("document.querySelector('button.predict')")),
        (3.95, click("document.querySelector('.card3d')")),  # fast spin on "gold" (13.98)
    ]),
    # 15.06 "Switch up your prediction ..." -> open a card's details, switch to Deep GP on "Deep Kernel" (17.64)
    "focus": (items("ring", "ragavan", "nazgul", "sheoldred", "watcher", "jace"), 3.5, 5.0, [
        (-1.6, click("document.querySelector('button.predict')")),
        (0.10, click("document.querySelector('.price-face')")),
        (2.55, click(button("Deep GP"))),
    ]),
    # 19.83 "Just add your list, [pause] press predict, [pause] and watch every card turn into treasure ..."
    "add": (items("ragavan", "ring", "nazgul", "sheoldred", "watcher"), 3.5, 5.6, [
        (0.05, click("document.querySelector('.add-slot')")),
        (0.45, type_into("#card-search", "Sol R")),
        (0.62, type_into("#card-search", "Sol Rin")),
        (0.85, click("document.querySelector('.hits button')")),
        (1.05, click("document.querySelector('.add-card')")),
        (1.83, click("document.querySelector('button.predict')")),  # on "predict" (21.66 after the first pause)
    ]),
    # 26.21 "Even new cards or your own hypotheticals." -> the form is already open; it fills itself in
    "custom": (items("ragavan", "ring", "nazgul"), 5.5, 3.4, [  # long settle: the load intro ignores clicks
        (-2.60, click("document.querySelector('.add-slot')")),
        (-2.25, click(button("+ Custom card"))),  # page turned and the left card settled before the shot starts
        *fill("input", 0, "Gilded Wurm", 0.10, 0.40),
        *fill("input", 1, "{4}{G}{G}", 0.42, 0.60),
        *fill("input", 2, "Creature — Wurm", 0.62, 0.87),
        *fill("textarea", 0, "Trample. When Gilded Wurm enters, create three Treasure tokens.", 0.89, 1.60),
        *fill("input", 3, "7", 1.65, 1.66),
        *fill("input", 4, "7", 1.72, 1.73),
        (1.85, click("[...document.querySelectorAll('.custom-sheet .dropdown-button')][0]")),
        (1.98, click("[...document.querySelectorAll('.custom-sheet .dropdown-list button')].find(b=>b.textContent.trim()==='mythic')")),
        (2.10, click("[...document.querySelectorAll('.custom-sheet .chip input')][0]")),
    ]),
    # presentation GIFs (make_gifs.py), one continuous story: each starts where the previous one ends
    "gif_add_search": (items(*SIX), 4.5, 3.4, search_fill(0.3)),
    "gif_add_submit": (items(*SIX), 6.0, 2.6, [
        *[(t - 3.0, js) for t, js in search_fill(0.0, fast=True)],
        (0.30, click("document.querySelector('.add-card')")),
    ]),
    "gif_custom_fill": (items(*SIX) + [SOL_RING], 4.5, 5.6, [
        (0.30, click("document.querySelector('.add-slot')")),
        *custom_fill(1.0),
    ]),
    "gif_custom_submit": (items(*SIX) + [SOL_RING], 7.0, 2.8, [
        (-4.0, click("document.querySelector('.add-slot')")),
        *[(t - 3.6, js) for t, js in custom_fill(0.0, fast=True)],
        (0.30, click(button("Add to list"))),
    ]),
    "gif_predict": (items(*SIX) + [SOL_RING, WURM], 3.5, 3.4, [
        (0.40, click("document.querySelector('button.predict')")),
    ]),
    "gif_details": (items(*SIX) + [SOL_RING, WURM], 4.5, 3.6, [
        (-2.0, click("document.querySelector('button.predict')")),
        (0.40, click("document.querySelectorAll('.price-face')[1]")),
    ]),
}


def record(name):
    seed, settle, dur, actions = SHOTS[name]
    out = BUILD / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    prof = BUILD / f".chrome_{name}"
    shutil.rmtree(prof, ignore_errors=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=9334", f"--force-device-scale-factor={DPR:.6f}",
                             f"--user-data-dir={prof}", f"--window-size={CSS_W},{CSS_H}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9334/json"))
                break
            except Exception:
                time.sleep(0.1)
        c = CDP([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"])
        c.call("Page.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=CSS_W, height=CSS_H, deviceScaleFactor=DPR, mobile=False)
        c.call("Page.navigate", url=APP)  # same origin first, so localStorage can be seeded
        time.sleep(1.0)
        c.js(f"localStorage.setItem('mtg-price-list-v1', {json.dumps(json.dumps(seed))}); true")
        c.call("Page.startScreencast", format="jpeg", quality=92, maxWidth=W, maxHeight=H, everyNthFrame=1)
        pre = sorted(a for a in actions if a[0] < 0)  # set-up actions before the shot (e.g. Predict so prices are up)
        c.call("Page.navigate", url=APP)
        if settle == "load":
            t0 = time.time()
            c.recording = True
        else:
            lead = -pre[0][0] if pre else 0.0
            time.sleep(max(settle - lead, 0))
            t_pre = time.time() + lead  # the moment the shot starts
            for t, js in pre:
                while time.time() < t_pre + t:
                    time.sleep(0.005)
                c.js(js)
            while time.time() < t_pre:
                time.sleep(0.005)
            t0 = time.time()
            c.recording = True
        for t, js in sorted(a for a in actions if a[0] >= 0):
            while time.time() - t0 < t:
                time.sleep(0.005)
            ok = c.js(js)
            print(f"  {name} +{t:.2f}s {'ok' if ok is not False else 'MISSED'}")
        while time.time() - t0 < dur:
            time.sleep(0.01)
        c.recording = False
        c.call("Page.stopScreencast")
        frames = sorted(c.frames)  # timestamps are wall-clock seconds, comparable with t0
        meta = []
        for i, (ts, data) in enumerate(frames):
            (out / f"{i:05d}.jpg").write_bytes(base64.b64decode(data))
            meta.append(ts)
        (out / "frames.json").write_text(json.dumps({"t0_wall": t0, "timestamps": meta, "duration": dur}))
        print(f"{name}: {len(frames)} frames over {dur:.1f}s")
    finally:
        proc.terminate()
        time.sleep(0.5)
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or SHOTS):
        record(n)
