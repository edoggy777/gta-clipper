#!/usr/bin/env python3
"""petals.py - the ending for mashup.py --hearts (keep this file in the same folder as mashup.py).

About a hundred small hearts, the same red metal heart as in the bullet transition, drift down from the top
of the screen like petals from a tree, rocking and swaying, and settle until they cover the whole frame.
mashup.py lays this over the last seconds of the video while the picture fades to black."""
import struct, subprocess, zlib
from pathlib import Path
import numpy as np
import bullet

COUNT = 200      # about how many hearts (the exact number is whatever fills an even grid)
SIZE = 1.5       # heart width compared with its own patch of screen (bigger = more overlap, fuller cover)
SWAY = 0.6       # how far they drift from side to side while falling (0 = straight down)
ROCK = 28        # degrees they rock back and forth, like petals
SEED = 3         # change for a different random arrangement
FACE_TURN = 0    # degrees the falling hearts are turned to the side (0 = facing straight out)
FACE_LEAN = 0    # degrees they lean back (0 = upright). After changing either, delete art/heart_front.png


def _load(path):
    """Reads a PNG written by bullet.py. Returns None for any other kind of PNG."""
    d = Path(path).read_bytes()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, head = 8, b"", None
    while pos < len(d):
        n, tag = struct.unpack(">I", d[pos:pos + 4])[0], d[pos + 4:pos + 8]
        if tag == b"IHDR":
            head = struct.unpack(">IIBBBBB", d[pos + 8:pos + 21])
        elif tag == b"IDAT":
            idat += d[pos + 8:pos + 8 + n]
        pos += 12 + n
    if not head or head[2:] != (8, 6, 0, 0, 0):
        return None
    w, h = head[0], head[1]
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 4 * w)
    if raw[:, 0].any():
        return None
    return raw[:, 1:].reshape(h, w, 4).astype(np.float32) / 255


def _sample(img, ys, xs):
    """Picks colours from img at fractional positions (smoothly); outside the picture is see-through."""
    h, w = img.shape[:2]
    ok = (ys > -1) & (ys < h) & (xs > -1) & (xs < w)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    fy, fx = (ys - y0)[..., None], (xs - x0)[..., None]
    pad = np.zeros((h + 2, w + 2, 4), np.float32)
    pad[1:-1, 1:-1] = img
    y0, x0 = np.clip(y0 + 1, 0, h), np.clip(x0 + 1, 0, w)
    out = (pad[y0, x0] * (1 - fy) * (1 - fx) + pad[y0, x0 + 1] * (1 - fy) * fx
           + pad[y0 + 1, x0] * fy * (1 - fx) + pad[y0 + 1, x0 + 1] * fy * fx)
    return out * ok[..., None]


def _shrink(img, width):
    """Cuts the picture down to the heart itself and scales it to the given width. Colours are stored
    already multiplied by how solid they are, which keeps the edges clean."""
    a = img[..., 3]
    rows, cols = np.where(a.max(axis=1) > 0.02)[0], np.where(a.max(axis=0) > 0.02)[0]
    img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()
    img[..., :3] *= img[..., 3:4]
    b = max(1, int(img.shape[1] / width))              # first average whole blocks, then fine-tune
    if b > 1:
        h, w = img.shape[0] // b * b, img.shape[1] // b * b
        img = img[:h, :w].reshape(h // b, b, w // b, b, 4).mean(axis=(1, 3))
    f = img.shape[1] / width
    ys, xs = np.meshgrid((np.arange(int(img.shape[0] / f)) + 0.5) * f - 0.5, (np.arange(int(width)) + 0.5) * f - 0.5,
                         indexing="ij")
    return _sample(img, ys, xs)


def _spin(img, deg):
    """The picture turned by deg degrees, on a square big enough that nothing is cut off."""
    h, w = img.shape[:2]
    side = int(np.ceil(np.hypot(h, w))) + 2
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    yy, xx = np.meshgrid(np.arange(side) - (side - 1) / 2, np.arange(side) - (side - 1) / 2, indexing="ij")
    return _sample(img, c * yy - s * xx + (h - 1) / 2, s * yy + c * xx + (w - 1) / 2)


def _front_heart():
    """The same red metal heart as the bullet scene, but facing the front. Drawn once, kept as art/heart_front.png."""
    f = bullet.ART / "heart_front.png"
    img = _load(f) if f.exists() else None
    if img is None:
        keep = getattr(bullet, "HEART_TURN", 0), getattr(bullet, "HEART_LEAN", 0)
        bullet.HEART_TURN, bullet.HEART_LEAN = FACE_TURN, FACE_LEAN
        try:
            whole = bullet.make_heart()
        finally:
            bullet.HEART_TURN, bullet.HEART_LEAN = keep
        bullet.ART.mkdir(parents=True, exist_ok=True)
        bullet._save(f, whole)
        img = np.asarray(whole, np.float32)
    return img


def make(W, H, fps, seconds, out, ff="ffmpeg"):
    """Writes a see-through video (out, a .mov) of the hearts falling and settling over `seconds` seconds."""
    heart = _front_heart()
    rng = np.random.RandomState(SEED)
    cols = max(2, int(round(np.sqrt(COUNT * W / H))))
    rows = int(np.ceil(COUNT / cols))
    cw, ch = W / cols, H / rows
    scales = (0.85, 1.0, 1.2)
    base = [_shrink(heart, SIZE * cw * k) for k in scales]
    turned = {}                                         # (which size, angle) -> turned picture, made when first needed
    speed = (H + 2.5 * cw) / (0.5 * seconds)            # every heart falls at the same speed, like petals
    hearts = []
    for r in range(rows):
        for c in range(cols):
            tx = (c + 0.5 + rng.uniform(-0.25, 0.25)) * cw
            ty = (r + 0.5 + rng.uniform(-0.25, 0.25)) * ch
            land = seconds * (0.50 + 0.40 * (1 - (r + 0.5) / rows) + rng.uniform(0, 0.06))   # bottom rows settle first
            fall = (ty + 1.5 * cw) / speed
            hearts.append(dict(tx=tx, ty=ty, t0=land - fall, fall=fall, k=rng.randint(len(scales)),
                               rest=rng.uniform(-30, 30), p1=rng.uniform(0, 1), p2=rng.uniform(0, 1),
                               f1=rng.uniform(0.45, 0.8), f2=rng.uniform(0.6, 1.1)))
    rng.shuffle(hearts)                                 # so they overlap each other in a random order
    n = int(round(seconds * fps))
    enc = subprocess.Popen([str(ff), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                            "-r", str(fps), "-i", "-", "-c:v", "qtrle", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / fps
        frame = np.zeros((H, W, 4), np.float32)
        for h in hearts:
            p = (t - h["t0"]) / h["fall"]
            if p <= 0:
                continue
            p = min(p, 1.0)
            loose = 1 - p                               # 1 while it is high up, 0 once it has settled
            x = h["tx"] + SWAY * cw * np.sin(2 * np.pi * (h["f1"] * t + h["p1"])) * loose
            y = -1.5 * cw + (h["ty"] + 1.5 * cw) * p
            ang = h["rest"] + ROCK * np.sin(2 * np.pi * (h["f2"] * t + h["p2"])) * (0.3 + 0.7 * loose)
            key = (h["k"], int(round(ang / 5)) * 5)
            if key not in turned:
                turned[key] = _spin(base[key[0]], key[1])
            s = turned[key]
            sh, sw = s.shape[:2]
            x0, y0 = int(round(x - sw / 2)), int(round(y - sh / 2))
            a, b, c2, d = max(x0, 0), max(y0, 0), min(x0 + sw, W), min(y0 + sh, H)
            if a >= c2 or b >= d:
                continue
            part = s[b - y0:d - y0, a - x0:c2 - x0]
            frame[b:d, a:c2] = part + frame[b:d, a:c2] * (1 - part[..., 3:4])
        frame[..., :3] /= np.maximum(frame[..., 3:4], 1e-4)
        enc.stdin.write((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8).tobytes())
        print(f"\r  drawing the falling hearts {i + 1}/{n}", end="", flush=True)
    print()
    enc.stdin.close()
    if enc.wait() or not Path(out).exists():
        raise SystemExit("could not write the falling-hearts video")
    return len(hearts)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "hearts_test.mov"
    print(f"{make(1080, 1920, 30, 3.0, out)} hearts -> {out}")
