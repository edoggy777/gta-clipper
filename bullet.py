#!/usr/bin/env python3
"""bullet.py - the bullet transition for mashup.py (keep this file in the same folder as mashup.py).

The old shot drops to black. A shiny bullet flies in from the far left and drags the next shot onto the
screen behind it. It hits a metal heart on the right, bounces off, and the heart falls out of frame.
With solo=True there is no heart and the bullet just flies across the whole screen.

The bullet, heart, spark and streak pictures are drawn by this file the first time they are needed and
saved in the folder "art" next to it. Replace any of those PNG files with your own to change the look.
Run  python3 bullet.py  to (re)make the pictures."""
import struct, zlib
from pathlib import Path
import numpy as np

ART = Path(__file__).resolve().parent / "art"
BANNER_FILL = "bars"   # what the banner shows: "bars" = TV colour bars, "scene" = the next shot
SCENE_FADE = 0.10      # with "bars": seconds the next shot takes to fade in once the flash is over
HIT_FLASH = 1.22     # seconds the gold outline flash lasts when the bullet hits the heart (0 = none)
RIM = 0.014          # thickness of the gold outline, as a share of the frame's short side
BLACK = 0.12      # seconds the old shot takes to drop to black when the bullet appears
FLAP = 0.135      # how far the banner swings up and down, as a share of the frame height (0 = no flapping)
RIPPLE = 0.40     # how wavy the banner's edges are (0 = straight edges, a stiff triangle)
BANNER = 0.32     # how wide the triangle opens: the share of the frame height it reaches one frame-width behind its tip
HEART_TURN = 40   # degrees the heart's front is swung toward the viewer's left (0 = facing straight out)
HEART_LEAN = 42   # degrees the heart leans back, top away from the viewer (0 = upright)


# ---------------------------------------------------------------- pictures
def _save(path, rgba):
    """Writes a float RGBA picture (values 0 to 1) as a PNG file."""
    a = (np.clip(rgba, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    chunk = lambda tag, data: (struct.pack(">I", len(data)) + tag + data
                               + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _half(img):
    """Shrinks a picture to half size (it is drawn double size first, for smooth edges)."""
    a = img[..., 3:4]
    rgb = (img[..., :3] * a).reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    a = a.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, 1).mean(axis=(1, 3))
    return np.concatenate([rgb / np.maximum(a, 1e-6), a], axis=2)


def _metal(nx, ny, nz, tint):
    """Colour of polished metal facing direction (nx, ny, nz): it mirrors a bright sky, a dark horizon
    and a warm floor, which is what makes chrome look like chrome. ny points up, nz toward the viewer."""
    rx, ry = 2 * nz * nx, 2 * nz * ny                       # where the mirror is looking
    sky = np.clip(ry, 0, 1)[..., None] ** 0.6
    col = np.array([1.0, 1.0, 1.0]) + (np.array([0.50, 0.58, 0.72]) - 1.0) * sky
    floor = np.array([0.08, 0.07, 0.06]) + np.array([0.74, 0.66, 0.58]) * np.clip(-ry, 0, 1)[..., None] ** 0.5
    col = np.where((ry < 0)[..., None], floor, col)
    col = col * (1 - 0.8 * np.exp(-(ry / 0.05) ** 2))[..., None]          # dark line at the horizon
    col = col * (0.9 + 0.1 * rx[..., None]) * np.array(tint)
    edge = (0.5 * (1 - np.clip(nz, 0, 1)) ** 3)[..., None]                 # edges catch more light
    col = col * (1 - edge) + edge
    light = np.array([-0.35, 0.55, 0.75]); half = light / np.linalg.norm(light) + np.array([0, 0, 1.0])
    half /= np.linalg.norm(half)
    shine = np.clip(nx * half[0] + ny * half[1] + nz * half[2], 0, 1) ** 120
    return np.clip(col + shine[..., None], 0, 1)


def make_heart(n=512, tint=(1.0, 0.13, 0.17)):
    m = 2 * n
    v = (np.arange(m) + 0.5) / m * 2 - 1
    sx, sy = np.meshgrid(v * 1.9, -v * 1.9 + 0.12)          # the heart fills about 2/3 of the picture, so it can spin
    # turn the heart: first lean it back, then swing its front toward the viewer's left
    T, L = np.radians(HEART_TURN), np.radians(HEART_LEAN)
    lean = np.array([[1, 0, 0], [0, np.cos(L), np.sin(L)], [0, -np.sin(L), np.cos(L)]])
    turn = np.array([[np.cos(T), 0, -np.sin(T)], [0, 1, 0], [np.sin(T), 0, np.cos(T)]])
    R = turn @ lean                                          # heart's own directions -> directions on screen

    def at(d):
        """Position inside the heart's own space of the point d units toward the viewer, for every pixel."""
        return (R[0, 0] * sx + R[1, 0] * sy + R[2, 0] * d, R[0, 1] * sx + R[1, 1] * sy + R[2, 1] * d,
                R[0, 2] * sx + R[1, 2] * sy + R[2, 2] * d)

    def shape(d):                                            # a real 3-D heart: inside where this is below 0
        x, y, z = at(d)
        return (x * x + 2.25 * z * z + y * y - 1) ** 3 - x * x * y ** 3 - 0.1125 * z * z * y ** 3

    # look from the front toward the back and note where each pixel first meets the heart
    steps = np.linspace(1.4, -1.4, 57)
    hit = np.zeros(sx.shape, bool)
    lo, hi = np.zeros_like(sx), np.zeros_like(sx)            # lo = just inside the heart, hi = just outside
    for d0, d1 in zip(steps[:-1], steps[1:]):
        new = (shape(d1) < 0) & ~hit
        lo, hi = np.where(new, d1, lo), np.where(new, d0, hi)
        hit |= new
    for _ in range(18):
        mid = (lo + hi) / 2
        deeper = shape(mid) < 0
        lo, hi = np.where(deeper, mid, lo), np.where(deeper, hi, mid)
    x, y, z = at((lo + hi) / 2)
    g = x * x + 2.25 * z * z + y * y - 1
    fx = 6 * x * g * g - 2 * x * y ** 3                      # which way the surface faces, in the heart's own space
    fy = 6 * y * g * g - 3 * x * x * y * y - 0.3375 * z * z * y * y
    fz = 13.5 * z * g * g - 0.225 * z * y ** 3
    nx = R[0, 0] * fx + R[0, 1] * fy + R[0, 2] * fz          # ... and as seen on screen
    ny = R[1, 0] * fx + R[1, 1] * fy + R[1, 2] * fz
    nz = R[2, 0] * fx + R[2, 1] * fy + R[2, 2] * fz
    ln = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
    rgb = _metal(nx / ln, ny / ln, np.clip(nz / ln, 0, 1), tint)
    return _half(np.concatenate([rgb, hit.astype(float)[..., None]], axis=2))


def make_bullet(n=512, tint=(1.0, 0.72, 0.38)):
    m = 2 * n
    px, py = np.meshgrid(np.arange(m) + 0.5, np.arange(m) + 0.5)
    length, rad = 0.8 * m, 0.8 * m / 6.4                     # points right; fills 80% of the picture, so it can spin
    s = (np.arange(m) + 0.5 - 0.1 * m) / length              # 0 at the tail, 1 at the tip
    u = np.clip((s - 0.5) / 0.5, 0, 1)
    prof = rad * np.where(s < 0.06, 0.85 + 0.15 * np.clip(s, 0, 1) / 0.06, (1 - u ** 1.6) ** 0.8)
    prof = np.where((s < 0) | (s > 1), 0, prof)
    slope = np.clip(np.gradient(prof), -4, 4)[None, :]
    r = np.maximum(prof[None, :], 1e-6)
    dy = -(py - m / 2) / r
    inside = (np.abs(dy) < 1) & (prof[None, :] > 0.5)
    dyc = np.clip(dy, -1, 1)
    ln = np.sqrt(1 + slope ** 2)
    rgb = _metal(-slope / ln + 0 * dyc, dyc / ln, np.sqrt(1 - dyc ** 2) / ln, tint)
    groove = ((s > 0.30) & (s < 0.33))[None, :, None]        # the ring pressed into the jacket
    rgb = np.where(groove, rgb * 0.55, rgb)
    return _half(np.concatenate([rgb, inside.astype(float)[..., None]], axis=2))


def make_spark(n=256):
    v = (np.arange(n) + 0.5) / n * 2 - 1
    x, y = np.meshgrid(v, v)
    glow = (np.exp(-np.abs(x) * 3.5) * np.exp(-(y * 22) ** 2) + np.exp(-np.abs(y) * 3.5) * np.exp(-(x * 22) ** 2)
            + 0.9 * np.exp(-(np.hypot(x, y) * 5) ** 2))
    rgb = np.ones((n, n, 3)) * np.array([1.0, 0.95, 0.75])
    return np.concatenate([rgb, np.clip(glow, 0, 1)[..., None]], axis=2)


def make_streak(w=512, h=48):
    x, y = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h * 2 - 1)
    rgb = np.ones((h, w, 3)) * np.array([1.0, 0.97, 0.90])
    return np.concatenate([rgb, (0.85 * x ** 2.2 * np.exp(-(y / 0.35) ** 2))[..., None]], axis=2)


def make_vortex(phase, ppl=180):
    """The air the bullet cuts through: a corkscrew of three strands that widens behind it, plus the
    V-shaped shock wave off its tip. phase (0 to 1) turns the corkscrew; 8 phases in a row make it spin.
    The picture is 4.6 bullet lengths long and the bullet sits near its right end."""
    w, h = int(4.6 * ppl), int(1.6 * ppl)
    x, y = np.meshgrid((np.arange(w) + 0.5) / ppl, (np.arange(h) + 0.5) / ppl - 0.8)   # measured in bullet lengths
    back = np.clip(3.2 - x, 0, None)                          # distance behind the bullet's tail
    wide = 0.05 + 0.15 * back ** 0.85                         # the corkscrew opens up as it trails away
    thick = 0.022 + 0.018 * back
    fade = np.clip(1 - back / 3.2, 0, 1) ** 1.3 * np.clip(back / 0.15, 0, 1)
    a = np.zeros_like(x)
    for k in range(3):
        ang = 2 * np.pi * (back / 0.95 - phase + k / 3)
        near = 0.4 + 0.6 * (0.5 + 0.5 * np.cos(ang))          # the side of the strand nearest the viewer is brighter
        a += near * np.exp(-((y - wide * np.sin(ang)) / thick) ** 2)
    a = a * fade + 0.30 * fade * np.exp(-(y / (0.04 + 0.05 * back)) ** 2)
    tip = np.clip(4.2 - x, 0, None)                           # distance behind the bullet's tip
    shock = (0.55 * np.exp(-tip / 1.3) * (x < 4.2) * np.clip((0.8 - np.abs(y)) / 0.25, 0, 1)
             * np.exp(-((np.abs(y) - 0.42 * tip) / (0.012 + 0.012 * tip)) ** 2))
    rgb = np.ones((h, w, 3)) * np.array([0.86, 0.94, 1.0])
    return np.concatenate([rgb, np.clip(0.9 * a + shock, 0, 1)[..., None]], axis=2)


def crack(img, side):
    """One half of the heart, broken along a jagged line down its middle. side: -1 = left half, 1 = right half."""
    h, w = img.shape[:2]
    a = img[..., 3]
    cols = np.arange(w)[None, :]
    mid = (a * cols).sum() / max(a.sum(), 1)                  # the heart's own middle (it may be turned)
    knots = np.arange(0, h + 36, 36)
    rng = np.random.RandomState(7)
    zig = rng.uniform(0.012, 0.038, len(knots)) * w * np.where(np.arange(len(knots)) % 2, 1, -1)
    line = (mid + np.interp(np.arange(h), knots, zig))[:, None]
    out = img.copy()
    out[..., 3] = a * ((cols < line - 1.5) if side < 0 else (cols > line + 1.5))
    return out


def rim(img, tint=(1.0, 0.72, 0.38)):
    """A glowing outline of a picture's shape in the bullet's metal colour: near-white on the edge itself,
    gold in the glow around it."""
    a = (img[..., 3] > 0.5).astype(float)

    def blur(x, r):
        k = np.ones(2 * r + 1) / (2 * r + 1)
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, x)
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, x)
    edge = np.clip(1 - np.abs(blur(a, 4) - 0.5) / 0.5, 0, 1) ** 1.5   # 1 right on the boundary, fading either side
    glow = np.clip(4 * blur(blur(edge, 6), 6), 0, 1)
    rgb = np.array(tint) * (1 - edge[..., None]) + np.array([1.0, 0.95, 0.80]) * edge[..., None]
    return np.concatenate([rgb, np.clip(edge + 0.7 * glow, 0, 1)[..., None]], axis=2)


def art(remake=False):
    """Returns the four picture files, drawing any that are missing."""
    ART.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, make in (("bullet", make_bullet), ("streak", make_streak), ("heart", make_heart), ("spark", make_spark)):
        files[name] = ART / f"{name}.png"
        if remake or not files[name].exists():
            _save(files[name], make())
    halves = ("heart_l", "heart_r", "heart_l_rim", "heart_r_rim")
    for name in halves:
        files[name] = ART / f"{name}.png"
    if remake or not all(files[name].exists() for name in halves):
        whole = make_heart()                                  # the two broken halves are cut from the same drawing
        for name, side in (("heart_l", -1), ("heart_r", 1)):
            half = crack(whole, side)
            _save(files[name], half)
            _save(files[name + "_rim"], rim(half))            # ... and each gets an outline in the bullet's gold
    files["vortex"] = ART / "vortex_%02d.png"                 # 8 pictures, played in a loop so the air spins
    for i in range(8):
        f = ART / f"vortex_{i:02d}.png"
        if remake or not f.exists():
            _save(f, make_vortex(i / 8))
    return files


# ---------------------------------------------------------------- which cuts
def choose(segs, fps, every_on, every, at, seconds, solo):
    """Marks the cuts that get the bullet: about every Nth cut, and the cut nearest each --bullet-at second."""
    forced = set()
    for t in (at if len(segs) > 1 else []):
        near = min(range(1, len(segs)), key=lambda j: abs(segs[j]["t0"] - t))
        forced.add(near)
        print(f"--bullet-at {t:g}: using the cut at {segs[near]['t0']:.2f}s")
    since, n = 0, 0
    for k in range(1, len(segs)):
        since += 1
        before = segs[k - 1]["frames"] / fps
        auto = every_on and since >= max(1, every) and before >= 1.2 and segs[k]["act"] != "explosion"
        if k in forced and before < 0.8:
            print(f"  the shot before {segs[k]['t0']:.2f}s is too short for the bullet, skipped")
        if auto or (k in forced and before >= 0.8):
            segs[k]["bullet"], segs[k]["solo"] = min(seconds, 0.6 * before), solo
            segs[k].pop("pip", None)                          # one transition per cut: the bullet wins
            segs[k]["tin"], segs[k]["punch"] = "cut", 0.0
            since, n = 0, n + 1
    print(f"bullet transition on {n} of {len(segs) - 1} cuts")


def choose_from(segs, fps, at, seconds, solo):
    """--bullet-from: the sequence starts on the cut nearest each of those seconds and plays over the
    beginning of the shot that follows (which appears once the flash is over)."""
    for t in (at if len(segs) > 1 else []):
        k = min(range(1, len(segs)), key=lambda j: abs(segs[j]["t0"] - t))
        own = segs[k]["frames"] / fps
        if own < 1.0:
            print(f"--bullet-from {t:g}: the shot at {segs[k]['t0']:.2f}s is too short for the bullet, skipped")
            continue
        segs[k]["bullet_in"], segs[k]["solo"] = min(seconds, 0.85 * own), solo
        for j in (k, k + 1):                                  # no other transition into or out of this shot
            if j < len(segs):
                segs[j].pop("pip", None)
                segs[j].pop("bullet", None)
        segs[k]["tin"], segs[k]["punch"] = "cut", 0.0
        note = "" if seconds <= 0.85 * own else f" (shortened to {0.85 * own:.1f}s to fit that {own:.1f}s shot)"
        print(f"--bullet-from {t:g}: starts on the cut at {segs[k]['t0']:.2f}s{note}")


# ---------------------------------------------------------------- ffmpeg graph
def graph(pv, W, H, fps, frames, n, solo, lead_in=False):
    """Returns (extra ffmpeg inputs, filter graph). The graph takes [m] (this clip, finished) and input 1
    (the next shot, through the filter chain pv) and gives [v]. n = frames the transition lasts;
    it ends exactly on the cut."""
    files = art()
    P, T0 = n / fps, (0.0 if lead_in else (frames - n) / fps)   # lead_in: it opens the clip instead of closing it
    base = min(W, H)
    even = lambda v: 2 * max(1, round(v / 2))
    L = 0.20 * base                               # bullet length on screen
    Sb = even(L / 0.8)                            # its picture is square with room to spin
    hw = 0.30 * base                              # heart width on screen
    Sh = even(hw * 1.67)
    yh, xh = 0.5 * H, W - 0.62 * hw               # where the heart hangs: far right, just inside the edge
    cx0 = -1.5 * L                                # bullet centre at the start: a full bullet length off screen
    if solo:
        Tf = P                                    # it flies the whole time and leaves on the right
        v = (W + 1.6 * L - cx0) / (0.75 * P)       # gone by three quarters of the way, so the banner can finish opening
    else:
        Tf = 0.45 * P                             # flight, then the bounce and the fall
        v = (xh - 0.46 * hw - L / 2 - cx0) / Tf
    Tr = max(P - Tf, 1 / fps)
    cx1 = cx0 + v * Tf                            # bullet centre when it hits
    xc = cx1 + L / 2                              # where its tip touches the heart
    xt = cx1 - L / 2                              # where its tail is then: the next shot has been dragged this far
    vy = (yh + Sb) / (0.7 * Tr)                   # bounce: up and back, off the top of the frame
    drop = (H - yh + Sh / 2 + 10) / (0.85 * Tr) ** 2

    def ex(u):
        """All the moving numbers as ffmpeg expressions of u = seconds since the transition began."""
        w = f"({u}-{Tf:.4f})"
        fly = f"lt({u},{Tf:.4f})"
        return dict(
            fly=fly,
            cx=f"if({fly},{cx0:.2f}+{v:.2f}*{u},{cx1:.2f}-{0.45 * v:.2f}*{w})",
            cy=f"if({fly},{yh:.2f},{yh:.2f}-{vy:.2f}*{w})",
            spin=f"if({fly},0,-14*{w})",
            hx=f"if({fly},{xh:.2f},{xh:.2f}+{0.015 * W / Tr:.2f}*{w})",
            hy=f"if({fly},{yh:.2f},{yh:.2f}+{drop:.2f}*{w}*{w})",
            tilt=f"if({fly},0,{1.1 / Tr:.3f}*{w})",
            # X = right-hand edge of the next shot: it rides on the bullet's tail, then slides the rest of the way
            # X = the point of the banner: it rides on the bullet's tail, then carries on past the right edge
            X=(f"max(0,{cx0 - L / 2:.2f}+{v:.2f}*{u})" if solo else
               f"if({fly},max(0,{cx0 - L / 2:.2f}+{v:.2f}*{u}),"
               f"{xt:.2f}+{1.1 * W - xt:.2f}*(1-pow(1-clip({w}/{0.6 * Tr:.4f},0,1),2)))"),
            # q = 0 while the banner is a narrow triangle, 1 once it has opened out to cover the whole frame
            q=(f"clip(({u}-{0.45 * P:.4f})/{0.4 * P:.4f},0,1)" if solo else f"clip({w}/{0.6 * Tr:.4f},0,1)"))
    loc, g = ex("t"), ex(f"(t-{T0:.4f})")         # before / after the pieces are moved to the end of the clip
    shift = f"setpts=PTS+{T0:.4f}/TB"
    img = lambda name: ["-loop", "1", "-framerate", str(fps), "-t", f"{P + 0.2:.3f}", "-i", str(files[name])]
    more = img("bullet") + img("vortex") + ([] if solo else img("heart") + img("spark") + img("heart_l") + img("heart_r") + img("heart_l_rim") + img("heart_r_rim"))
    SW, SHt = even(4.6 * L), even(1.6 * L)        # the vortex picture; the bullet sits 3.7 lengths from its left end
    gq = ex("T")                                  # the same moving numbers again, for the banner-shape filter
    mw, mh = even(W / 4), even(H / 4)             # the shape is worked out small and scaled up: quick, and soft-edged
    dist = f"({gq['X']}-X*{W / mw:.4f})"          # how far a point is behind the banner's tip
    # a point shows the next shot when it is behind the tip and close enough to the banner's (flapping) centre line
    shape = (f"0*st(0,{dist})+255*gt(ld(0),0)*lt(abs(Y*{H / mh:.4f}-{yh:.2f}"
             f"-{FLAP * H:.2f}*sin(ld(0)/{0.095 * base:.2f}-11*T)*min(1,ld(0)/{2 * L:.2f})),"
             f"{BANNER * H / W:.4f}*(1+16*pow({gq['q']},2))*ld(0)*(1+{RIPPLE}*sin(ld(0)/{0.056 * base:.2f}-14*T)))")
    if solo or HIT_FLASH <= 0:
        rimfc = "[b1a]null[b1];"
    else:   # the outline of the banner, drawn the same way as the banner itself, shown in gold from the hit on
        off = (f"abs(Y*{H / mh:.4f}-{yh:.2f}"
               f"-{FLAP * H:.2f}*sin(ld(0)/{0.095 * base:.2f}-11*T)*min(1,ld(0)/{2 * L:.2f}))")
        half = f"{BANNER * H / W:.4f}*(1+16*pow({gq['q']},2))*ld(0)*(1+{RIPPLE}*sin(ld(0)/{0.056 * base:.2f}-14*T))"
        edge = f"0*st(0,{dist})+255*gt(ld(0),0)*lt(abs({off}-{half}),{RIM * base:.2f})"
        rimfc = (f"color=c=0xFFDC96:s={W}x{H}:r={fps}:d={P + 0.5:.4f},trim=end_frame={n},format=yuv420p[rc];"
                 f"color=black:s={mw}x{mh}:r={fps}:d={P + 0.5:.4f},trim=end_frame={n},format=gray,geq=lum='{edge}',"
                 f"scale={W}:{H}:flags=bilinear[rk];"
                 f"[rc][rk]alphamerge,fade=t=out:st={Tf:.4f}:d={HIT_FLASH:.4f}:alpha=1,{shift}[rim];"
                 f"[b1a][rim]overlay=enable='between(t,{T0 + Tf - 0.04:.4f},{T0 + Tf + HIT_FLASH:.4f})':"
                 f"eof_action=pass[b1];")
    if BANNER_FILL == "scene" and not lead_in:    # the banner is made of the next shot, as before
        fill, scenefc = f"[1:v]{pv}[nx0];", ""
    else:                                         # the banner is made of TV colour bars ...
        fill = (f"smptebars=s={W}x{2 * int(H * 0.75)}:r={fps}:d={P + 0.5:.4f},crop={W}:{H}:0:0,trim=end_frame={n},"
                f"format=yuv420p[nx0];")
        # ... and the next shot itself fades in over it (under the heart and bullet) when the flash is over
        ta = 0.85 * P if solo else min(Tf + max(HIT_FLASH, 0.0), P - 2 / fps)
        rimfc = rimfc.replace("[b1];", "[b1r];")
        scenefc = (f"[1:v]{pv},format=yuva420p,fade=t=in:st={ta:.4f}:d={SCENE_FADE:.3f}:alpha=1,{shift}[nsc];"
                   f"[b1r][nsc]overlay=enable='gte(t,{T0 + ta:.4f})':eof_action=repeat[b1];")
        if lead_in:   # here [m] is the new shot itself: the black, the bars and the outline lie on top of it and clear at ta
            ta = min(ta, P - SCENE_FADE - 2 / fps)
            scenefc = (f"[b1r]format=yuva420p,fade=t=out:st={ta:.4f}:d={SCENE_FADE:.3f}:alpha=1[cov];"
                       f"[m][cov]overlay=eof_action=pass[b1];")
    stage = (f"color=black:s={W}x{H}:r={fps}:d={P + 0.5:.4f},trim=end_frame={n},format=yuv420p[mb];" if lead_in
             else f"[m]fade=t=out:st={T0:.4f}:d={BLACK:.3f}[mb];")   # what is behind the banner: black / the old shot going black
    fc = (f"{fill}color=black:s={W}x{H}:r={fps}:d={P + 0.5:.4f}[bk];"
          f"[bk][nx0]overlay=x='min({loc['X']},{W})-{W}':y=0:shortest=1,format=yuv420p[nxs];"   # next shot slides with the bullet
          f"color=black:s={mw}x{mh}:r={fps}:d={P + 0.5:.4f},trim=end_frame={n},format=gray,geq=lum='{shape}',"
          f"scale={W}:{H}:flags=bilinear[mk];"                                      # ... and is cut to the banner shape
          f"[nxs][mk]alphamerge,{shift}[nxa];"
          f"{stage}[mb][nxa]overlay=eof_action=repeat[b1a];{rimfc}{scenefc}"   # old shot drops to black
          f"[3:v]format=rgba,scale={SW}:{SHt},{shift}[st];"
          f"[b1][st]overlay=x='{g['cx']}-{3.7 * L:.2f}':y={yh - SHt / 2:.2f}:"
          f"enable='{g['fly']}':eof_action=pass[b2];")
    last = "b2"
    if not solo:
        hit = T0 + Tf                                         # the moment of the hit, on the clip's own clock
        wl, wg = f"max(0,t-{Tf:.4f})", f"(t-{hit:.4f})"       # seconds since the hit
        hx0, hy0 = xh - Sh / 2, yh - Sh / 2
        # after the hit there are two halves: the left one drops back and tips over, the right one is knocked on
        lx, ly = f"{hx0:.2f}-{0.10 * W / Tr:.2f}*{wg}", f"{hy0:.2f}+{drop:.2f}*{wg}*{wg}"
        rx = f"{hx0:.2f}+{0.05 * W / Tr:.2f}*{wg}"
        ry = f"{hy0:.2f}-{0.06 * H / Tr:.2f}*{wg}+{1.2 * drop:.2f}*{wg}*{wg}"
        lrot, rrot = f"rotate=a='-1.5*{wl}/{Tr:.4f}':c=black@0", f"rotate=a='1.7*{wl}/{Tr:.4f}':c=black@0"
        after = f"enable='gte(t,{hit:.4f})':eof_action=pass"
        fc += (f"[4:v]format=rgba,scale={Sh}:{Sh},fade=t=in:st=0:d=0.10:alpha=1,{shift}[ht];"
               f"[b2][ht]overlay=x={hx0:.2f}:y={hy0:.2f}:enable='lt(t,{hit:.4f})':eof_action=pass[h0];"   # whole, until the hit
               f"[6:v]format=rgba,scale={Sh}:{Sh},{lrot},{shift}[hl];[h0][hl]overlay=x='{lx}':y='{ly}':{after}[h1];"
               f"[7:v]format=rgba,scale={Sh}:{Sh},{rrot},{shift}[hr];[h1][hr]overlay=x='{rx}':y='{ry}':{after}[h2];")
        if HIT_FLASH > 0:   # gold outlines on both halves, brightest at the hit and fading out
            lit = f"enable='between(t,{hit:.4f},{hit + HIT_FLASH:.4f})':eof_action=pass"
            dim = f"fade=t=out:st={Tf:.4f}:d={HIT_FLASH:.4f}:alpha=1"
            fc += (f"[8:v]format=rgba,scale={Sh}:{Sh},{dim},{lrot},{shift}[el];[h2][el]overlay=x='{lx}':y='{ly}':{lit}[h3];"
                   f"[9:v]format=rgba,scale={Sh}:{Sh},{dim},{rrot},{shift}[er];[h3][er]overlay=x='{rx}':y='{ry}':{lit}[b3];")
        else:
            fc += "[h2]null[b3];"
        last = "b3"
    fc += (f"[2:v]format=rgba,scale={Sb}:{Sb},rotate=a='{loc['spin']}':c=black@0,{shift}[bl];"
           f"[{last}][bl]overlay=x='{g['cx']}-{Sb / 2}':y='{g['cy']}-{Sb / 2}':eof_action=pass[b4];")
    if solo:
        return more, fc + "[b4]format=yuv420p[v]"
    burst = min(max(HIT_FLASH, 3 / fps), 0.15)    # a quick spark at the point of the hit, in bullet gold
    sp = even(1.0 * hw)
    fc += (f"[5:v]format=rgba,scale={sp}:{sp},colorchannelmixer=gg=0.80:bb=0.42,"
           f"fade=t=out:st={Tf:.4f}:d={burst:.4f}:alpha=1,{shift}[sp];"
           f"[b4][sp]overlay=x={xc - sp / 2:.2f}:y={yh - sp / 2:.2f}:"
           f"enable='between(t,{T0 + Tf:.4f},{T0 + Tf + burst:.4f})':eof_action=pass,format=yuv420p[v]")
    return more, fc


if __name__ == "__main__":
    for name, f in art(remake=True).items():
        print(f"made {f}")
