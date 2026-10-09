"""Contact sheet of a recorded shot at given times: python ad/sheet.py <shot> t1 t2 ..."""
import json, sys
from PIL import Image
shot, times = sys.argv[1], [float(x) for x in sys.argv[2:]]
m = json.load(open(f"ad/build/{shot}/frames.json")); ts = m["timestamps"]; t0 = m["t0_wall"]
def at(t):
    i = max([k for k, x in enumerate(ts) if x - t0 <= t] or [0]); return Image.open(f"ad/build/{shot}/{i:05d}.jpg")
ims = [at(t).resize((640, 360)) for t in times]
cols = 3; rows = (len(ims) + cols - 1) // cols
c = Image.new("RGB", (640 * cols, 360 * rows)); [c.paste(im, ((i % cols) * 640, (i // cols) * 360)) for i, im in enumerate(ims)]
c.save(f"ad/build/_sheet_{shot}.png")
