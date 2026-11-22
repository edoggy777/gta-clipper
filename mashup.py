#!/usr/bin/env python3
"""mashup.py - cut several videos into one edit that is paced to a song.

Story arc: intro shot -> rising tension -> chase + explosions -> falling tension.
Cuts land on the song's beats: long shots where the song is quiet, fast cuts where it is loud.
Quiet breaks in the middle of the song get a calm "cutscene" shot.

Usage:
  python3 mashup.py --song song.mp3 video1.mp4 video2.mp4 ...
Options:
  --length 60   seconds of the song to use (default 60)
  --start 30    where in the song to begin (default: picked automatically around the drop)
  --wide        make a 1920x1080 video instead of vertical 1080x1920
  --out FILE    output file (default ~/clipper/outbox/mashup_<song>.mp4)
  --ramp 0.4    action clips start fast and settle (speed ramp)
  --endcard F   picture or short video shown after the mashup
  --review      with --versions: make a blind comparison page
  --edit FILE   render a saved edit list (a .edit.txt file) exactly: its clips, in its order
  --pip         the next shot previews in a small grey box, then grows to fill the frame on the cut
  --bullet      a metal bullet flies across, the next shot follows it in; it bounces off a heart, which falls
Every finished video is checked (black frames, frozen picture, loudness, silence); --no-check turns that off.
"""
import argparse, bisect, random, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import json
import numpy as np
try:
    import eases, check, review
except ImportError as err:
    sys.exit(f"{err}. Put eases.py, check.py and review.py in the same folder as mashup.py.")

try:
    import bullet
except ImportError as err:
    sys.exit(f"{err}. Put bullet.py in the same folder as mashup.py.")

try:
    import petals
except ImportError as err:
    sys.exit(f"{err}. Put petals.py in the same folder as mashup.py.")

HOME = Path.home()
FF = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else shutil.which("ffmpeg")
FP = "/usr/bin/ffprobe" if Path("/usr/bin/ffprobe").exists() else shutil.which("ffprobe")
VLM = HOME / "clipper/vlm_score.py"
WORK = HOME / "clipper/work"
SR, HOP, NFFT, FPS = 22050, 512, 1024, 30
MOVE = 8
ZOOM = 0.0      # set by --zoom: each clip slowly zooms in by this much (0.04 = 4%)
FADE = 0.15     # with --zoom, each clip also fades in over this many seconds
MATCH = 8.0                                 # how strongly clips must fit the song (0 = off)
HIT_ACTS = ("explosion", "guns", "chase")   # shots whose big moment is lined up with the beat
PACE = 1.0       # 2 = cut twice as often
SPREAD = 6.0     # how strongly clips are shared out between the source videos (0 = off)
HIST = HOME / "clipper/used_clips.txt"   # footage used by earlier runs
MAXSHOT = 6.0    # longest single shot, in seconds
DARK = 0.0       # how strongly dark footage is avoided (0 = off)
GRADE = ""       # colour filter put on every clip
STACK = False    # whole picture in the middle, blurred copy above and below
TEXT = ""        # caption shown in the middle of the video
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HIT_EASE = "snapSettle"   # curve the zoom "hit" settles with (a name from eases.py, or "old")
RAMP = 0.0       # set by --ramp: action clips start fast and settle (0 = constant speed)
LUFS, TRUE_PEAK = -14, -2   # loudness target and peak ceiling for the finished video
PIP = 0.0        # set by --pip: seconds the next shot is previewed in a small grey box (0 = off)
PIP_SIZE = 0.3   # width of that box, as a share of the frame
PIP_GROW = 0.25  # seconds the box takes to grow to the full frame; it lands exactly on the cut
PIP_CORNER = ""  # qtl, qtr, qbl, qbr = middle of that quarter of the frame; tl, tr, bl, br = in that corner ("" = qtl)
PIP_BARS = 0.13  # seconds of TV colour bars flashed in the box before the preview shows (0 = off)
HEARTS = 0.0     # set by --hearts: seconds of small hearts falling over the end of the video (0 = off)
HEARTS_BG = "bars"   # behind the falling hearts: "bars" = TV colour bars that fade to black, "video" = the video itself
HEARTS_MID = 0.0  # set by --hearts-at: the cut (in seconds) that falling hearts in the middle finish on (0 = off)
HEARTS_LEN = 3.0  # set by --hearts-time: how long the hearts fall
GLITCH = []       # set by --glitch-at: the cuts (in seconds) a glitch starts on
GLITCH_IMG = None   # set by --glitch-image: picture that flickers in during the glitch
GLITCH_LEN = 0.5    # set by --glitch-time: seconds a glitch lasts
GLITCH_HOLD = 0.0   # set by --glitch-hold: seconds the picture stays, steady, after the glitch
GLITCH_SIZE = 0.6   # set by --glitch-size: width of the picture, as a share of the frame
_SIZE = {}


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, **kw)


def probe(path, entry, stream=None):
    cmd = [FP, "-v", "error"] + (["-select_streams", stream] if stream else [])
    out = run(cmd + ["-show_entries", entry, "-of", "csv=p=0", path], text=True).stdout
    return out.strip().splitlines()[0].strip(",") if out.strip() else ""


# ---------------------------------------------------------------- song analysis
def load_audio(path):
    raw = run([FF, "-v", "error", "-i", path, "-ac", "1", "-ar", SR, "-f", "f32le", "-"]).stdout
    y = np.frombuffer(raw, dtype=np.float32)
    if y.size < SR * 5:
        sys.exit(f"could not read the song (or it is under 5 seconds): {path}")
    return y


def analyse(y):
    """Returns onset strength and loudness, one value per frame (about 43 frames per second)."""
    n = 1 + (len(y) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    mag = np.log1p(10 * np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1)))
    flux = np.maximum(0, np.diff(mag, axis=0)).sum(axis=1)
    onset = np.concatenate([[0], flux])
    onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    return onset, rms


def smooth(a, width):
    width = max(1, int(width))
    return np.convolve(a, np.ones(width) / width, mode="same")


def find_beats(onset):
    """Tempo from autocorrelation, then dynamic-programming beat tracking. Returns beat frames."""
    fps = SR / HOP
    best, period = -1e18, int(fps / 2)
    for lag in range(int(fps * 60 / 180), int(fps * 60 / 70) + 1):
        bpm = 60 * fps / lag
        s = np.dot(onset[:-lag], onset[lag:]) / (len(onset) - lag)
        s *= np.exp(-0.5 * np.log2(bpm / 120) ** 2)
        if s > best:
            best, period = s, lag
    window = np.arange(-2 * period, -period // 2)
    penalty = -100 * np.log(-window / period) ** 2
    score, back = onset.copy(), np.full(len(onset), -1)
    for i in range(len(onset)):
        prev = i + window
        ok = prev >= 0
        if ok.any():
            cand = penalty[ok] + score[prev[ok]]
            j = int(np.argmax(cand))
            score[i] = onset[i] + cand[j]
            back[i] = prev[ok][j]
    i = len(onset) - 2 * period + int(np.argmax(score[-2 * period:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1]), 60 * fps / period


def plan_song(song, length, start, tail=0.0):
    """Returns (start, length, segments). Each segment: dict(t0, t1, energy, hit)."""
    y = load_audio(song)
    onset, rms = analyse(y)
    fps = SR / HOP
    dur = len(y) / SR
    beats, bpm = find_beats(onset)
    bt = beats / fps
    loud = smooth(rms, fps * 2)
    length = min(length, max(5.0, dur - tail))   # tail: seconds of song kept for the end card
    if start is None:  # put the first loud section about a third of the way in
        drop = np.argmax(smooth(rms, fps * 4) >= 0.7 * smooth(rms, fps * 4).max()) / fps
        start = max(0.0, drop - 0.35 * length)
        if len(bt):
            start = float(bt[np.argmin(np.abs(bt - start))])
    start = max(0.0, min(start, dur - length - tail))
    keep = (bt >= start) & (bt <= start + length)
    b, bf = bt[keep] - start, beats[keep]
    if len(b) < 8:  # no clear beat: fall back to a cut every half second
        b = np.arange(0, length, 0.5)
        bf = ((b + start) * fps).astype(int)
    e = loud[np.minimum(bf, len(loud) - 1)]
    e = (e - e.min()) / (e.max() - e.min() + 1e-9)
    lo, hi = np.quantile(e, 0.35), np.quantile(e, 0.65)
    hit = onset[np.minimum(bf, len(onset) - 1)]

    segs, i = [], 0
    while i < len(b) - 1:
        n = 8 if e[i] < lo else 4 if e[i] < hi else 2
        n = max(1, round(n / PACE))
        if not segs:
            n = max(n, round(8 / PACE))
        end = lambda k: b[i + k] if i + k < len(b) else length
        t0 = 0.0 if not segs else b[i]
        while end(n) - t0 < 0.7 and i + n < len(b):
            n *= 2
        while n > 1 and end(n) - t0 > MAXSHOT:
            n //= 2
        level = "low" if e[i:i + n].mean() < lo else "mid" if e[i:i + n].mean() < hi else "high"
        segs.append(dict(t0=t0, t1=end(n), level=level, hit=float(hit[i]),
                         energy=float(e[i:i + n].mean())))
        i += n
    segs[-1]["t1"] = length
    if len(segs) > 1 and segs[-1]["t1"] - segs[-1]["t0"] < 0.7:
        segs.pop()
        segs[-1]["t1"] = length
    print(f"song: {bpm:.0f} BPM, using {start:.1f}s to {start + length:.1f}s, {len(segs)} cuts")
    return start, length, segs


def set_clip_count(segs, n):
    """Merges the shortest cuts, or splits the longest, until there are exactly n clips."""
    dur = lambda j: segs[j]["t1"] - segs[j]["t0"]
    while len(segs) > max(1, n):
        k = min(range(len(segs)), key=dur)
        if k == 0:
            j = 1
        elif k == len(segs) - 1 or dur(k - 1) <= dur(k + 1):
            j = k - 1
        else:
            j = k + 1
        a, b = min(k, j), max(k, j)
        la, lb = dur(a), dur(b)
        keep = segs[a] if la >= lb else segs[b]
        energy = (segs[a].get("energy", 0) * la + segs[b].get("energy", 0) * lb) / (la + lb)
        segs[a:b + 1] = [dict(keep, t0=segs[a]["t0"], t1=segs[b]["t1"], hit=segs[a]["hit"], energy=energy)]
    while len(segs) < n:
        k = max(range(len(segs)), key=dur)
        if dur(k) < 0.8:      # clips would get too short to see
            break
        mid = (segs[k]["t0"] + segs[k]["t1"]) / 2
        segs[k:k + 1] = [dict(segs[k], t1=mid), dict(segs[k], t0=mid)]
    print(f"clips: {len(segs)}")


def assign_acts(segs, length):
    highs = [k for k, s in enumerate(segs) if s["level"] == "high"]
    last_high = highs[-1] if highs else len(segs) - 1
    boom = np.quantile([segs[k]["hit"] for k in highs], 0.85) if highs else 0
    for k, s in enumerate(segs):
        if k == 0 or s["t0"] < 0.06 * length:
            s["act"] = "intro"
        elif k > last_high or s["t0"] > 0.9 * length:
            s["act"] = "falling"
        elif s["level"] == "high":
            s["act"] = "explosion" if s["hit"] >= boom else "chase"
        elif s["level"] == "low" and highs and k > highs[0]:
            s["act"] = "cutscene"
        else:
            s["act"] = "rising"
    if highs and not any(s["act"] == "explosion" for s in segs):
        segs[highs[len(highs) // 2]]["act"] = "explosion"


# ---------------------------------------------------------------- video scoring
def scatter_explosions(segs, length, n):
    """Exactly n explosion shots, spread evenly through the video (never the opening shot)."""
    for s in segs:
        if s["act"] == "explosion":
            s["act"] = "chase"
    free = list(range(1, len(segs)))
    for i in range(n):
        if not free:
            break
        target = (i + 0.5) / n * length
        k = min(free, key=lambda j: abs((segs[j]["t0"] + segs[j]["t1"]) / 2 - target))
        segs[k]["act"] = "explosion"
        free.remove(k)


def scatter_act(segs, length, n, act, phase):
    """n shots of this kind, spread evenly through the video, taken from chase slots first."""
    free = [j for j in range(1, len(segs)) if segs[j]["act"] in ("chase", "rising")]
    if len(free) < n:
        free += [j for j in range(1, len(segs)) if segs[j]["act"] in ("cutscene", "falling")]
    for i in range(n):
        if not free:
            break
        target = (i + phase) / n * length
        k = min(free, key=lambda j: abs((segs[j]["t0"] + segs[j]["t1"]) / 2 - target))
        segs[k]["act"] = act
        free.remove(k)


def scatter_emotional(segs, length, n):
    """n emotional shots, spread evenly, taken from the quiet slots first."""
    free = [j for j in range(1, len(segs)) if segs[j]["act"] in ("cutscene", "falling", "rising")]
    if len(free) < n:
        free += [j for j in range(1, len(segs)) if segs[j]["act"] == "chase"]
    for i in range(n):
        if not free:
            break
        target = (i + 0.5) / n * length
        k = min(free, key=lambda j: abs((segs[j]["t0"] + segs[j]["t1"]) / 2 - target))
        segs[k]["act"] = "emotional"
        free.remove(k)


def scatter_landscape(segs, length, n):
    """n city / landscape shots, spread evenly, taken from the quiet slots first."""
    free = [j for j in range(1, len(segs)) if segs[j]["act"] in ("cutscene", "falling", "rising")]
    if len(free) < n:
        free += [j for j in range(1, len(segs)) if segs[j]["act"] == "chase"]
    for i in range(n):
        if not free:
            break
        target = (i + 0.25) / n * length
        k = min(free, key=lambda j: abs((segs[j]["t0"] + segs[j]["t1"]) / 2 - target))
        segs[k]["act"] = "landscape"
        free.remove(k)


def scatter_scuba(segs, length, n):
    """n scuba / underwater shots, spread evenly, taken from the quiet slots first."""
    free = [j for j in range(1, len(segs)) if segs[j]["act"] in ("cutscene", "falling", "rising")]
    if len(free) < n:
        free += [j for j in range(1, len(segs)) if segs[j]["act"] == "chase"]
    for i in range(n):
        if not free:
            break
        target = (i + 0.75) / n * length
        k = min(free, key=lambda j: abs((segs[j]["t0"] + segs[j]["t1"]) / 2 - target))
        segs[k]["act"] = "scuba"
        free.remove(k)


SKIP = []   # (video name, from second, to second) that must never be used


def set_chases(segs, length, n):
    """Exactly n chase shots, spread evenly through the video."""
    mid = lambda j: (segs[j]["t0"] + segs[j]["t1"]) / 2
    have = [j for j, s in enumerate(segs) if s["act"] == "chase"]
    if len(have) > n:      # too many: keep n, the rest become build-up shots
        pool, want, new = have, n, "rising"
    else:                  # too few: take build-up slots first, then quiet ones
        want, new = n - len(have), "chase"
        pool = [j for j in range(1, len(segs)) if segs[j]["act"] == "rising"]
        for quiet in ("cutscene", "falling"):
            if len(pool) < want:
                pool += [j for j in range(1, len(segs)) if segs[j]["act"] == quiet]
    picked = []
    for i in range(want):
        if not pool:
            break
        target = (i + 0.5) / want * length
        k = min(pool, key=lambda j: abs(mid(j) - target))
        picked.append(k)
        pool.remove(k)
    for j in (pool if new == "rising" else picked):
        segs[j]["act"] = new


def prepare(video):
    """Returns (path to use, list of (t, score, label)). Converts AV1 etc. to H.264 once."""
    video = Path(video).resolve()
    if not video.exists():
        sys.exit(f"not found: {video}")
    cache = Path(str(video) + ".scores.txt")
    if not cache.exists() and probe(video, "stream=codec_name", "v:0") != "h264":
        WORK.mkdir(parents=True, exist_ok=True)
        conv = WORK / (video.stem + ".h264.mp4")
        if not conv.exists():
            print(f"converting {video.name} to H.264 (one time) ...", flush=True)
            r = run([FF, "-y", "-v", "error", "-i", video, "-vf", "scale=-2:1080", "-c:v", "libx264",
                     "-preset", "fast", "-crf", "18", "-c:a", "aac", "-b:a", "192k", conv], text=True)
            if r.returncode:
                sys.exit(f"convert failed: {r.stderr[-1500:]}")
        video, cache = conv, Path(str(conv) + ".scores.txt")
    if not cache.exists():
        print(f"scoring {video.name} with the vision model (slow the first time) ...", flush=True)
        r = run([sys.executable, VLM, video], text=True)
        if r.returncode or not r.stdout.strip():
            print(r.stderr[-3000:])
            sys.exit(f"scoring failed for {video}")
        cache.write_text(r.stdout)
    rows = []
    for line in cache.read_text().splitlines():
        p = line.split()
        try:
            rows.append((float(p[0]), float(p[1]), p[2] if len(p) > 2 else "other"))
        except (ValueError, IndexError):
            pass
    return video, rows


def motion_curve(video):
    """Returns a function: time -> how much the picture is moving there, 0 (still) to 1 (most movement)."""
    cache = Path(str(video) + ".motion.txt")
    if not cache.exists():
        print(f"measuring movement in {video.name} ...", flush=True)
        r = run([FF, "-nostats", "-v", "error", "-i", video, "-an", "-vf",
                 "fps=4,scale=160:-2,select='gte(scene,0)',metadata=print:file=-", "-f", "null", "-"], text=True)
        t = re.findall(r"pts_time:([\d.]+)", r.stdout)
        v = re.findall(r"scene_score=([\d.]+)", r.stdout)
        cache.write_text("".join(f"{a} {b}\n" for a, b in zip(t, v)))
    rows = [line.split() for line in cache.read_text().splitlines()]
    if len(rows) < 8:
        return lambda t: 0.5
    tt = np.array([float(a) for a, _ in rows]); vv = np.array([float(b) for _, b in rows])

    def at(t):
        w = vv[(tt >= t - 1) & (tt <= t + 2)]
        return float((vv < np.median(w)).mean()) if len(w) else 0.5

    def peak(t0, t1):
        m = (tt >= t0) & (tt <= t1)
        return float(tt[m][np.argmax(vv[m])]) if m.any() else None
    at.peak = peak
    return at


def scene_cuts(video, dur):
    """Times where the source video itself cuts to a new shot (found once, then cached)."""
    cache = Path(str(video) + ".cuts.txt")
    if not cache.exists():
        print(f"finding shot changes in {video.name} ...", flush=True)
        r = run([FF, "-nostats", "-v", "info", "-i", video, "-an", "-vf",
                 "scale=320:-2,select='gt(scene,0.3)',showinfo", "-f", "null", "-"], text=True)
        cache.write_text("\n".join(re.findall(r"pts_time:([\d.]+)", r.stderr)) + "\n")
    return [0.0] + [float(x) for x in cache.read_text().split()] + [dur]


def luma_curve(video):
    """Returns a function: time -> how bright the picture is there (0 black .. 255 white)."""
    cache = Path(str(video) + ".luma.txt")
    if not cache.exists():
        print(f"measuring brightness in {video.name} ...", flush=True)
        r = run([FF, "-nostats", "-v", "error", "-i", video, "-an", "-vf",
                 "fps=1,scale=64:-2,signalstats,metadata=print:file=-", "-f", "null", "-"], text=True)
        t = re.findall(r"pts_time:([\d.]+)", r.stdout)
        v = re.findall(r"YAVG=([\d.]+)", r.stdout)
        cache.write_text("".join(f"{a} {b}\n" for a, b in zip(t, v)))
    rows = [line.split() for line in cache.read_text().splitlines()]
    if len(rows) < 4:
        return lambda t: 128.0
    tt = np.array([float(a) for a, _ in rows]); vv = np.array([float(b) for _, b in rows])
    return lambda t: float(np.interp(t, tt, vv))


def bars(src, t, d):
    """Returns a crop filter if this part of the source has black bars baked in, else nothing."""
    if src not in _SIZE:
        try:
            _SIZE[src] = tuple(int(x) for x in probe(src, "stream=width,height", "v:0").split(",")[:2])
        except ValueError:
            _SIZE[src] = (0, 0)
    sw, sh = _SIZE[src]
    r = run([FF, "-nostats", "-v", "info", "-ss", f"{t:.3f}", "-t", f"{min(d, 3):.3f}", "-i", src, "-an",
             "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"], text=True)
    found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", r.stderr)
    if not found or not sw:
        return ""
    w, h, x, y = (int(v) for v in found[-1])
    top, left = min(y, sh - h - y), min(x, sw - w - x)   # same amount off both sides
    top = top if 0.03 * sh <= top <= 0.25 * sh else 0
    left = left if 0.03 * sw <= left <= 0.25 * sw else 0
    return f"crop={sw - 2 * left}:{sh - 2 * top}:{left}:{top}," if top or left else ""


def clip_look(src, t, d, pre):
    """Average brightness, contrast, colour strength and colour cast of one clip."""
    r = run([FF, "-nostats", "-v", "error", "-ss", f"{t:.3f}", "-t", f"{min(d, 4):.3f}", "-i", src, "-an",
             "-vf", f"{pre}fps=3,scale=160:-2,signalstats,metadata=print:file=-", "-f", "null", "-"], text=True)
    get = lambda key: [float(x) for x in re.findall(key + r"=([\d.]+)", r.stdout)]
    y = get("YAVG")
    if not y:
        return None
    return dict(y=float(np.mean(y)), spread=float(np.mean(get("YHIGH")) - np.mean(get("YLOW"))),
                sat=float(np.mean(get("SATAVG"))), u=float(np.mean(get("UAVG"))), v=float(np.mean(get("VAVG"))))


def match_looks(segs, k):
    """Pulls every clip toward the middle look of the whole edit. k: 0 = no change, 1 = full match."""
    for s in segs:
        s["pre"] = bars(s["src"], s["src_t"], s["src_d"])
        s["look"] = clip_look(s["src"], s["src_t"], s["src_d"], s["pre"])
    looks = [s["look"] for s in segs if s["look"]]
    if len(looks) < 2:
        return
    tgt = {key: float(np.median([l[key] for l in looks])) for key in looks[0]}
    lim = lambda x, a, b: max(a, min(b, x))
    for s in segs:
        l = s["look"]
        if not l:
            continue
        m, mt = lim((l["y"] - 16) / 219, 0.03, 0.97), lim((tgt["y"] - 16) / 219, 0.03, 0.97)
        gamma = lim(1 + k * (np.log(m) / np.log(mt) - 1), 0.6, 1.8)        # brightness
        con = lim(1 + k * (tgt["spread"] / max(l["spread"], 1) - 1), 0.8, 1.35)   # contrast
        sat = lim(1 + k * (tgt["sat"] / max(l["sat"], 0.5) - 1), 0.6, 1.8)        # colour strength
        du = lim(k * (tgt["u"] - l["u"]), -14, 14)                         # colour cast, blue-yellow
        dv = lim(k * (tgt["v"] - l["v"]), -14, 14)                         # colour cast, red-green
        s["match"] = (f"lutyuv=u='clip(val+{du:.1f},16,240)':v='clip(val+{dv:.1f},16,240)',"
                      f"eq=gamma={gamma:.3f}:contrast={con:.3f}:saturation={sat:.3f}")
    print(f"matched the look of {len(looks)} clips (brightness {tgt['y']:.0f}, colour {tgt['sat']:.0f})")


class Picker:
    """Hands out source moments for each act, never reusing the same footage."""
    ORDER = {  # act: list of (labels, sort high-to-low?) tried in turn
        "explosion": [({"explosion"}, True), ({"violence"}, True), ({"guns"}, True), ({"chase"}, True), (None, True)],
        "guns": [({"guns"}, True), ({"violence"}, True), ({"guns"}, True), ({"chase"}, True), (None, True)],
        "drugs": [({"drugs"}, True), ({"violence"}, True), ({"guns"}, True), ({"chase"}, True), (None, True)],
        "chase": [({"chase"}, True), ({"guns"}, True), ({"violence"}, True), (None, True)],
        "intro": [({"other"}, True), (None, False)],
        "rising": [({"chase"}, False), ({"guns"}, False), ({"other"}, True), (None, False)],
        "cutscene": [({"other"}, True), ({"chase"}, False), (None, False)],
        "emotional": [({"emotional"}, True), ({"other"}, True), ({"chase"}, False), (None, False)],
        "landscape": [({"landscape"}, True), ({"other"}, True), ({"emotional"}, True), (None, False)],
        "airplane": [({"airplane"}, True), ({"landscape"}, True), ({"other"}, True), (None, False)],
        "scuba": [({"scuba"}, True), ({"landscape"}, True), ({"other"}, True), (None, False)],
        "boat": [({"boat"}, True), ({"chase"}, True), ({"landscape"}, True), ({"other"}, True), (None, False)],
        "biker": [({"biker"}, True), ({"chase"}, True), ({"other"}, True), (None, False)],
        "falling": [({"other"}, True), (None, False)],
    }

    def __init__(self, videos):
        self.cands, self.dur, self.used, self.cuts, self.mot = [], {}, {}, {}, {}
        self.avoid, self.count, self.jitter, self.rng = {}, {}, 0.0, random.Random(0)
        for v in videos:
            path, rows = prepare(v)
            self.dur[path] = float(probe(path, "format=duration") or 0)
            self.used[path] = []
            self.cuts[path] = scene_cuts(path, self.dur[path])
            mot = motion_curve(path)
            self.mot[path] = mot
            lum = luma_curve(path) if DARK else None
            for t, sc, lab in rows:
                m = mot(t)
                dk = DARK * max(0.0, (60 - lum(t + 1)) / 60) if lum else 0.0
                self.cands.append((path, t, sc + MOVE * m, lab, m, dk))
            labels = [lab for _, _, lab in rows]
            print(f"  {path.name}: " + ", ".join(f"{labels.count(x)} {x}" for x in sorted(set(labels))))
        if not self.cands:
            sys.exit("no scored frames in any input video")

    def reset(self, seed, jitter):
        """Start a new edit: nothing used yet, and a new random variation."""
        self.used = {p: [] for p in self.dur}
        self.count, self.jitter, self.rng = {}, jitter, random.Random(seed)

    def block(self, segs):
        """Keep later versions away from the footage this edit used."""
        for s in segs:
            self.avoid.setdefault(s["src"], []).append((s["src_t"] - 1, s["src_t"] + s["src_d"] + 1))

    def load_history(self):
        names = {p.name: p for p in self.dur}
        n = 0
        for line in (HIST.read_text().splitlines() if HIST.exists() else []):
            f = line.rsplit(" ", 2)
            if len(f) == 3 and f[0] in names:
                self.avoid.setdefault(names[f[0]], []).append((float(f[1]), float(f[2])))
                n += 1
        print(f"staying away from {n} clips used in earlier videos")

    def free(self, path, s, d, strict=True):
        if any(Path(path).name == n and s < b and s + d > a for n, a, b in SKIP):
            return False
        if strict and any(s < b and s + d > a for a, b in self.avoid.get(path, [])):
            return False
        return all(s + d + 0.5 <= a or s >= b + 0.5 for a, b in self.used[path])

    def pick(self, act, d, frac, energy=None, only=None):
        for attempt in range(4):
            for labels, desc in self.ORDER[act]:
                pool = [c for c in self.cands if (labels is None or c[3] in labels)
                        and (only is None or c[0] == only)]
                fit = ((lambda c: MATCH * (1 - abs(c[4] - energy))) if MATCH and energy is not None
                       else (lambda c: 0.0))
                picked = max(1, sum(self.count.values()))
                share = {q: SPREAD * self.count.get(q, 0) / picked for q in self.dur}   # used a lot = pushed back
                noise = {id(c): self.rng.uniform(0, self.jitter) for c in pool} if self.jitter else {}
                pen = lambda c: c[5] + share[c[0]] + noise.get(id(c), 0.0)
                pool.sort(key=lambda c: -(c[2] + fit(c) - pen(c)) if desc else c[2] - fit(c) + pen(c))
                if act == "rising" and labels and "chase" in labels:
                    pool = pool[:max(1, int(len(pool) * 0.6))]  # keep the best chases for the chase
                first = int(frac * len(pool)) if act in ("rising", "falling") else 0
                for path, t, sc, lab, m, _dk in pool[first:] + pool[:first]:
                    if self.dur[path] < d + 0.2:
                        continue
                    cuts = self.cuts[path]
                    k = bisect.bisect_right(cuts, t) - 1
                    a, b = cuts[k], cuts[min(k + 1, len(cuts) - 1)]
                    if b - a >= d + 0.3:      # keep the whole clip inside one shot
                        s = min(max(a + 0.15, t - d / 2), b - d - 0.15)
                        pk = getattr(self.mot[path], "peak", None)
                        if MATCH and act in HIT_ACTS and pk:
                            p = pk(max(a, t - 2), min(b, t + 2))
                            if p is not None and a + 0.15 <= p - 0.35 and p - 0.35 + d <= b - 0.15:
                                s = p - 0.35
                    elif attempt % 2 == 0:    # shot too short: try other moments first
                        continue
                    else:
                        s = min(max(0.0, t + 1 - d / 2), self.dur[path] - d - 0.1)
                    if self.free(path, s, d, attempt < 2):
                        self.used[path].append((s, s + d))
                        self.count[path] = self.count.get(path, 0) + 1
                        return path, s, sc, lab, m
        if only is not None:      # that video had nothing usable for this slot: take any video instead
            return self.pick(act, d, frac, energy)
        sys.exit("input videos are too short for this edit - add more footage or use a smaller --length")


# ---------------------------------------------------------------- render
def loudness_filter(song, start, total, fades):
    """Two-pass loudness: measure the song first, then apply one fixed volume change.
    (One pass has to guess as it goes, which can make the volume pump up and down.)"""
    one_pass = f"loudnorm=I={LUFS}:TP={TRUE_PEAK}"
    r = run([FF, "-nostats", "-v", "info", "-ss", f"{start:.3f}", "-t", f"{total:.3f}", "-i", song,
             "-af", f"{fades},loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA=11:print_format=json", "-f", "null", "-"], text=True)
    found = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr)
    try:
        m = json.loads(found[-1])
        vals = [float(m[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
    except (IndexError, KeyError, ValueError):
        return one_pass
    if not all(np.isfinite(vals)):
        return one_pass
    lra = min(50, max(11, int(np.ceil(vals[2]))))   # wide enough that the song's own dynamics are kept
    print(f"song loudness: {vals[0]:.1f} LUFS, moving it to {LUFS}")
    return (f"loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA={lra}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
            f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")


def card_length(card, seconds):
    """How long the end card is shown: a picture for `seconds`, a video for its own length (8 s at most)."""
    if not card:
        return 0.0
    if Path(card).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        return float(seconds)
    try:
        return round(min(8.0, float(probe(card, "format=duration"))), 2)
    except ValueError:
        sys.exit(f"could not read the end card: {card}")


def base_vf(s, W, H, wide, tag=""):
    """First part of a clip's filter: cut away black bars, then fill the frame."""
    zoom = 1.12 if s["act"] == "explosion" else 1.0   # punch in on explosions
    zw, zh = int(W * zoom / 2) * 2, int(H * zoom / 2) * 2
    pre = s["pre"] if "pre" in s else bars(s["src"], s["src_t"], s["src_d"])   # cut away black bars
    if STACK and not wide:   # whole picture in the middle, blurred copy filling above and below
        return (f"{pre}split[sa{tag}][sb{tag}];[sa{tag}]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},boxblur=20:2,eq=brightness=-0.10[sbg{tag}];[sb{tag}]scale={W}:-2[sfg{tag}];"
                f"[sbg{tag}][sfg{tag}]overlay=0:(H-h)/2,setsar=1")
    return f"{pre}scale={zw}:{zh}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"


def pip_room(segs, picker, move):
    """Works out how much footage there is just before each previewed clip (the preview box plays it).
    Where there is too little, the clip starts a little later inside its own shot to make room."""
    for s in segs:
        if not (s.get("pip") or s.get("bullet")):
            continue
        q, d, need = s["src"], s["src_d"], (s.get("pip") or s["bullet"]) * s["speed"]
        cuts = picker.cuts[q]
        k = max(0, bisect.bisect_right(cuts, s["src_t"] + 0.01) - 1)
        a, b = cuts[k], cuts[min(k + 1, len(cuts) - 1)]      # the shot this clip sits in
        shift = min(need - (s["src_t"] - a - 0.1), b - 0.15 - (s["src_t"] + d))
        if move and shift > 0 and s["act"] not in HIT_ACTS:
            t = s["src_t"] + shift
            clash = any(o is not s and o["src"] == q and t < o["src_t"] + o["src_d"] and t + d > o["src_t"]
                        for o in segs)
            if not clash and not any(q.name == n and t < t1 and t + d > t0 for n, t0, t1 in SKIP):
                s["src_t"] = t
        s["lead"] = max(0.0, min(need, s["src_t"] - a - 0.1))


def pip_filter(nxt, W, H, wide, frames):
    """Extra ffmpeg input and filter graph for one clip: the next shot shows as a small grey preview box
    during the end of this clip, then grows to the full frame in colour and lands exactly on the cut."""
    n = max(2, min(round(nxt["pip"] * FPS), frames - 2))     # frames the box is on screen
    P, T0, dur = n / FPS, (frames - n) / FPS, frames / FPS
    G = min(PIP_GROW, 0.45 * P)
    spd = nxt["speed"]
    lead = max(0.0, min(P * spd, nxt.get("lead", 0.0), nxt["src_t"]))   # footage just before the next clip
    hold = P - lead / spd                                    # too little of it: hold its first frame
    pv = base_vf(nxt, W, H, wide, "p")
    if spd != 1:
        pv += f",setpts=PTS/{spd}"
    pv += f",fps={FPS}"
    if hold > 0.5 / FPS:
        pv += f",tpad=start_mode=clone:start_duration={hold:.4f}"
    pv += f",trim=end_frame={n}"
    if nxt.get("match"):
        pv += "," + nxt["match"]
    if GRADE:
        pv += "," + GRADE
    p = f"clip((t-{dur - G:.4f})/{G:.4f},0,1)"               # 0 while the box waits, 1 when it is full frame
    e = f"({p}*{p}*(3-2*{p}))"
    f = PIP_SIZE
    w = f"2*trunc({W}*({f:.3f}+{1 - f:.3f}*{e})/2)"
    h = f"2*trunc({H}*({f:.3f}+{1 - f:.3f}*{e})/2)"
    corner = PIP_CORNER or "qtl"
    if corner[0] == "q":   # middle of one quarter of the frame: qtl = upper left quarter
        x0 = max(0.0, min(W - W * f, (0.25 if corner[2] == "l" else 0.75) * W - W * f / 2))
        y0 = max(0.0, min(H - H * f, (0.25 if corner[1] == "t" else 0.75) * H - H * f / 2))
    else:                  # tucked into a corner
        x0 = 0.04 * W if corner[1] == "l" else W - W * f - 0.04 * W
        top, bottom = (0.05 * H, 0.05 * H) if wide else (0.10 * H, 0.22 * H)   # clear of the app's buttons and caption
        y0 = top if corner[0] == "t" else H - H * f - bottom
    x, y = f"{x0:.1f}*(1-{e})", f"{y0:.1f}*(1-{e})"
    b = f"trunc({max(4, round(W * 0.006))}*(1-{e}))"         # white border, gone by the time the box is full
    fade, tv = f"fade=t=in:st={T0:.4f}:d=0.12:alpha=1,", ""   # no colour bars: the box fades in
    if PIP_BARS > 0:   # the box switches on like a TV: colour bars for a few frames, then the preview
        bn = max(1, min(round(PIP_BARS * FPS), n // 3))
        fade = ""
        tv = (f"[pg];smptebars=s={W}x{2 * int(H * 0.75)}:r={FPS}:d=2,crop={W}:{H}:0:0,trim=end_frame={bn},"
              f"setpts=PTS+{T0:.4f}/TB[tv];[pg][tv]overlay=eof_action=pass")
    graph = (f"[1:v]{pv},setpts=PTS-STARTPTS+{T0:.4f}/TB,hue=s='{e}'{tv},format=yuva420p,{fade}"
             f"scale=w='{w}':h='{h}':eval=frame[pp];"
             f"color=c=white:s=64x64:r={FPS}:d={P:.4f},format=yuva420p,setpts=PTS+{T0:.4f}/TB,{fade}"
             f"scale=w='{w}+2*{b}':h='{h}+2*{b}':eval=frame[pb];"
             f"[m][pb]overlay=x='{x}-{b}':y='{y}-{b}':eof_action=pass[m1];"
             f"[m1][pp]overlay=x='{x}':y='{y}':eof_action=pass,format=yuv420p[v]")
    return ["-ss", f"{nxt['src_t'] - lead:.3f}", "-t", f"{lead + 0.5:.3f}", "-i", nxt["src"]], graph


def bullet_filter(nxt, W, H, wide, frames):
    """Extra ffmpeg inputs and filter graph for one clip: a bullet crosses the end of this clip and the
    next shot follows it in (the moving parts are in bullet.py)."""
    n = max(4, min(round(nxt["bullet"] * FPS), frames - 2))   # frames the transition lasts
    P, spd = n / FPS, nxt["speed"]
    lead = max(0.0, min(P * spd, nxt.get("lead", 0.0), nxt["src_t"]))   # footage just before the next clip
    hold = P - lead / spd                                     # too little of it: hold its first frame
    pv = base_vf(nxt, W, H, wide, "p")
    if spd != 1:
        pv += f",setpts=PTS/{spd}"
    pv += f",fps={FPS}"
    if hold > 0.5 / FPS:
        pv += f",tpad=start_mode=clone:start_duration={hold:.4f}"
    pv += f",trim=end_frame={n}"
    if nxt.get("match"):
        pv += "," + nxt["match"]
    if GRADE:
        pv += "," + GRADE
    more, graph = bullet.graph(pv + ",format=yuv420p", W, H, FPS, frames, n, nxt.get("solo", False))
    return ["-ss", f"{nxt['src_t'] - lead:.3f}", "-t", f"{lead + 0.5:.3f}", "-i", nxt["src"]] + more, graph


def hearts_bg(W, H, t0, hd, last, tag):
    """What is behind the falling hearts: the TV colour bars cut in when the hearts start (at t0), then fade
    to black and stay black until the hearts are done. Returns the filter text; its result is [bg<tag>]."""
    return (f";smptebars=s={W}x{2 * int(H * 0.75)}:r={FPS}:d={hd + 0.5:.3f},crop={W}:{H}:0:0,"
            f"trim=end_frame={round(hd * FPS)},fade=t=out:st=0.25:d={0.55 * hd:.3f},format=yuva420p,"
            f"fade=t=in:st=0:d=0.15:alpha=1,setpts=PTS+{t0:.3f}/TB[bars{tag}];"
            f"[{last}][bars{tag}]overlay=eof_action=pass[bg{tag}]")


def glitch_filter(W, H, last, idx):
    """Filter text for --glitch-at. Starting on each chosen cut the picture tears sideways in bands, the
    colours split and static flickers. With --glitch-image that picture (ffmpeg input number idx) flickers
    in over the video and tears with it. Returns (filter text, name of the result)."""
    end = 0.5 / FPS
    on = "+".join(f"between(t,{t:.4f},{t + GLITCH_LEN - end:.4f})" for t in GLITCH)   # 1 while a glitch runs
    n = f"floor(t*{FPS}+0.5)"                    # frame number
    step = f"floor({n}/2)"                       # changes every 2 frames, so each tear can be seen
    fc = ""
    if GLITCH_IMG:
        show = "+".join(f"between(t,{t:.4f},{t + GLITCH_LEN + GLITCH_HOLD - end:.4f})" for t in GLITCH)
        gone = f"({on})*lt(mod({n}*7,5),2)"      # while glitching the picture drops out 2 frames in 5
        fc += (f";[{idx}:v]format=rgba,scale={2 * int(W * GLITCH_SIZE / 2)}:-2[gi];"
               f"[{last}][gi]overlay=x='(W-w)/2+({on})*{W * 0.04:.0f}*sin({n}*2.4)':y='(H-h)/2':"
               f"enable='({show})*(1-{gone})'[gl0]")
        last = "gl0"
    for k in range(5):                           # five bands jump to new heights and slide sideways
        bh = 2 * int(H * (0.025 + 0.02 * k) / 2)
        y = f"2*floor(mod(abs(sin({step}*12.9898+{k}*78.233))*43758.5453,1)*{(H - bh) // 2})"
        dx = f"{W * 0.10:.0f}*sin({step}*{1.7 + k:.1f}+{k})"
        fc += (f";[{last}]split[ga{k}][gb{k}];[gb{k}]crop={W}:{bh}:0:'{y}'[gc{k}];"
               f"[ga{k}][gc{k}]overlay=x='{dx}':y='{y}':enable='{on}'[gl{k + 1}]")
        last = f"gl{k + 1}"
    shift = max(2, round(W * 0.006))
    fc += (f";[{last}]chromashift=cbh={-shift}:crh={shift}:enable='{on}',"     # colours split apart
           f"noise=alls=28:allf=t:enable='{on}'[glx]")                         # static
    return fc, "glx"


def render(segs, song, start, length, out, wide, logo=None, card=None, card_d=0.0):
    W, H = (1920, 1080) if wide else (1080, 1920)
    tmp = Path(tempfile.mkdtemp(prefix="mashup_"))
    parts = []
    for k, s in enumerate(segs):
        vf = base_vf(s, W, H, wide)
        if RAMP > 0 and s["act"] in HIT_ACTS:              # speed ramp: starts fast, settles by the next cut
            v = f"(T-STARTT)/{s['src_d']:.4f}"             # how far through the source clip, 0 to 1
            vf += (f",setpts='{s['frames'] / FPS:.4f}*({1 - RAMP:.3f}*{v}+{RAMP:.3f}*{eases.expr('gentleIn', v)})/TB'")
        elif s["speed"] != 1:                              # slight slow motion as tension falls
            vf += f",setpts=PTS/{s['speed']}"
        vf += f",fps={FPS}"
        dur = s["frames"] / FPS
        punch, tin = s.get("punch", 0.0), s.get("tin")
        if ZOOM > 0 or punch > 0:   # slow push-in across the clip, plus a quick zoom "hit" on the cut
            if HIT_EASE == "old":
                hit = "exp(-t/0.09)"
            else:               # the hit settles along a measured curve and lands exactly on its mark
                hit = f"(1-{eases.expr(HIT_EASE, f't/{eases.seconds(HIT_EASE):.3f}')})"
            grow = f"(1+max(0,{ZOOM}*t/{dur:.3f}+{punch}*{hit}))"
            vf += (f",scale=w='2*trunc({W}*{grow}/2)':h='2*trunc({H}*{grow}/2)':eval=frame"
                   f",crop={W}:{H}:x='{W}*({grow}-1)/2':y='{H}*({grow}-1)/2'")
        if tin == "flash":          # white flash on the strongest beats
            vf += ",fade=t=in:st=0:d=0.14:color=white"
        elif tin == "dip":          # soft fade from black in quiet parts
            vf += f",fade=t=in:st=0:d={min(0.3, dur * 0.3):.3f}"
        elif tin is None and FADE > 0:   # old behaviour: every clip fades in
            vf += f",fade=t=in:st=0:d={min(FADE, dur * 0.4):.3f}"
        if s.get("match"):          # per-clip correction so all clips look alike
            vf += "," + s["match"]
        if GRADE:
            vf += "," + GRADE
        vf += ",format=yuv420p"
        part = tmp / f"seg_{k:03d}.mp4"
        how = ["-ss", f"{s['src_t']:.3f}", "-t", f"{s['src_d'] + 0.5:.3f}", "-i", s["src"]]
        if k + 1 < len(segs) and segs[k + 1].get("bullet"):   # a bullet crosses and the next shot follows it in
            more, graph = bullet_filter(segs[k + 1], W, H, wide, s["frames"])
            how += more + ["-filter_complex", f"[0:v]{vf}[m];{graph}", "-map", "[v]"]
        elif k + 1 < len(segs) and segs[k + 1].get("pip"):   # next shot previews in a box, then fills the frame
            more, graph = pip_filter(segs[k + 1], W, H, wide, s["frames"])
            how += more + ["-filter_complex", f"[0:v]{vf}[m];{graph}", "-map", "[v]"]
        elif s.get("bullet_in"):   # the bullet sequence opens this clip, starting on the cut
            n_in = max(4, min(round(s["bullet_in"] * FPS), s["frames"] - 2))
            more, graph = bullet.graph("null", W, H, FPS, s["frames"], n_in, s.get("solo", False), lead_in=True)
            how += (["-f", "lavfi", "-i", "color=black:s=16x16:r=30:d=0.1"] + more
                    + ["-filter_complex", f"[0:v]{vf}[m];{graph}", "-map", "[v]"])
        else:
            how += ["-vf", vf]
        r = run([FF, "-y", "-v", "error", *how, "-an", "-frames:v", s["frames"], "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "14", part], text=True)
        if r.returncode or not part.exists():
            sys.exit(f"cut failed on {s['src']}: {r.stderr[-1500:]}")
        parts.append(part)
        print(f"\r  cutting {k + 1}/{len(segs)}", end="", flush=True)
    print()
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    total = length + card_d
    cap = ""
    if TEXT:   # caption rises into place and fades in, instead of sitting there from frame one
        (tmp / "text.txt").write_text(TEXT.upper())
        ff = f"fontfile={FONT}:" if Path(FONT).exists() else ""
        t0, rise = 0.4, int(H * 0.035)
        land = eases.expr("softLand", f"(t-{t0})/{eases.seconds('softLand'):.3f}")
        cap = (f",drawtext={ff}textfile={tmp / 'text.txt'}:fontsize={int(W * 0.052)}:fontcolor=white:"
               f"borderw=4:bordercolor=black@0.9:shadowx=3:shadowy=3:x=(w-text_w)/2:"
               f"y='h*0.64+{rise}*(1-{land})':alpha='clip((t-{t0})/0.17,0,1)'")
    fc, last, extra = f"[0:v]fade=t=in:st=0:d=0.4{cap}[main]", "main", []
    if logo:   # logo in the centre: fades in, holds, then fades out while the video keeps playing
        hold, fade = 2.5, 1.0
        extra += ["-loop", "1", "-t", f"{hold + fade + 0.5}", "-i", logo]
        fc += (f";[2:v]format=rgba,scale={int(W * 0.35)}:-1,fade=t=in:st=0:d=0.3:alpha=1,"
               f"fade=t=out:st={hold}:d={fade}:alpha=1[lg];[main][lg]overlay=(W-w)/2:(H-h)/2:eof_action=pass[logod]")
        last = "logod"
    if card:   # end card after the last shot; the song keeps playing under it
        still = Path(card).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        extra += (["-loop", "1", "-framerate", FPS] if still else []) + ["-t", f"{card_d:.3f}", "-i", card]
        fc += (f";[{2 + bool(logo)}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,"
               f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p[card];"
               f"[{last}]setsar=1,format=yuv420p[body];[body][card]concat=n=2:v=1:a=0[joined]")
        last = "joined"
    fade_out = 1.0   # seconds the picture takes to fade to black at the very end
    if HEARTS > 0:   # small hearts fall like petals and cover the screen while it fades to black
        hd = min(HEARTS, max(1.0, total - 1.0))
        petals.make(W, H, FPS, hd, tmp / "hearts.mov", FF)
        extra += ["-i", tmp / "hearts.mov"]
        if HEARTS_BG == "bars":
            fc, last = fc + hearts_bg(W, H, total - hd, hd, last, "e"), "bge"
        fc += (f";[{2 + bool(logo) + bool(card)}:v]setpts=PTS+{total - hd:.3f}/TB[hearts];"
               f"[{last}][hearts]overlay=eof_action=pass[hrt]")
        last, fade_out = "hrt", (0.5 if HEARTS_BG == "bars" else min(2.0, 0.55 * hd))
    if HEARTS_MID > 0:   # the same hearts mid-video: they cover the screen, it dips to black, the next shot fades in
        hd = min(HEARTS_LEN, max(1.0, HEARTS_MID - 0.2))
        dip = 0.25 if HEARTS_BG == "bars" else min(2.0, 0.55 * hd)
        petals.make(W, H, FPS, hd, tmp / "hearts_mid.mov", FF)
        extra += ["-i", tmp / "hearts_mid.mov"]
        if HEARTS_BG == "bars":
            fc, last = fc + hearts_bg(W, H, HEARTS_MID - hd, hd, last, "m"), "bgm"
        fc += (f";[{2 + bool(logo) + bool(card) + (HEARTS > 0)}:v]setpts=PTS+{HEARTS_MID - hd:.3f}/TB[hmid];"
               f"[{last}][hmid]overlay=eof_action=pass[hm1];"
               f"color=black:s={W}x{H}:r={FPS}:d={dip + 0.35:.3f},format=yuva420p,fade=t=in:st=0:d={dip:.3f}:alpha=1,"
               f"fade=t=out:st={dip:.3f}:d=0.30:alpha=1,setpts=PTS+{HEARTS_MID - dip:.3f}/TB[dip];"
               f"[hm1][dip]overlay=eof_action=pass[hm2]")
        last = "hm2"
    if GLITCH:   # glitch that starts on the chosen cuts (it goes over everything above)
        idx = 2 + bool(logo) + bool(card) + (HEARTS > 0) + (HEARTS_MID > 0)
        if GLITCH_IMG:
            extra += ["-i", GLITCH_IMG]
        more, last = glitch_filter(W, H, last, idx)
        fc += more
    fades = f"afade=t=in:st=0:d=0.3,afade=t=out:st={total - 1.5:.2f}:d=1.5"
    fc += (f";[{last}]fade=t=out:st={total - fade_out:.2f}:d={fade_out:.2f}[v];"
           f"[1:a]{fades},{loudness_filter(song, start, total, fades)},"
           f"alimiter=limit=0.84:attack=3:release=60:level=0[a]")
    r = run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-ss", f"{start:.3f}", "-t", f"{total:.3f}", "-i", song, *extra, "-filter_complex", fc,
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out],
            text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        sys.exit(f"final render failed: {r.stderr[-1500:]}")
    return (W, H), total


def read_edit(path):
    """Reads a saved edit list. Each line gives one clip; the order of the lines is the order of the video."""
    if not Path(path).exists():
        sys.exit(f"not found: {path}")
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        m = re.match(r"\s*[\d.]+s\s+[\d.]+s\s+(\w+)\s+(.+?)\s+@\s+([\d.]+)s\s+\((\w+)\s+(-?[\d.]+)\)"
                     r"(?:\s+song\s+[\d.]+\s+/\s+clip\s+([\d.]+))?", line)
        if not m or m.group(1) not in Picker.ORDER:
            sys.exit(f"--edit: cannot read line {n} of {path}:\n  {line}")
        rows.append(dict(act=m.group(1), name=m.group(2), t=float(m.group(3)), label=m.group(4),
                         score=float(m.group(5)), motion=float(m.group(6) or 0.5)))
    if not rows:
        sys.exit(f"--edit: no clips in {path}")
    return rows


def edit_score(segs, n_sources):
    """Scores a whole edit so versions can be compared: footage quality, fit to the song, variety."""
    quality = float(np.mean([s["score"] for s in segs]))
    fit = float(np.mean([1 - abs(s["motion"] - s["energy"]) for s in segs]))
    names = [s["src"] for s in segs]
    share = np.array([names.count(n) for n in set(names)]) / len(names)
    variety = float(-(share * np.log(share)).sum() / np.log(n_sources)) if n_sources > 1 else 1.0
    return quality + 8 * fit + 6 * variety, quality, fit, variety


def main():
    ap = argparse.ArgumentParser(description="Cut several videos into one edit paced to a song.")
    ap.add_argument("--song", required=True)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--length", type=float, default=60)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--logo", default=None)
    ap.add_argument("--explosions", type=int, default=None)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--slow", type=float, default=0.75)
    ap.add_argument("--chase", type=int, default=None)
    ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--emotional", type=int, default=None)
    ap.add_argument("--landscape", type=int, default=None)
    ap.add_argument("--guns", type=int, default=None)
    ap.add_argument("--drugs", type=int, default=None)
    ap.add_argument("--planes", type=int, default=None)
    ap.add_argument("--scuba", type=int, default=None)
    ap.add_argument("--boats", type=int, default=None)
    ap.add_argument("--bikers", type=int, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--zoom", action="store_true")
    ap.add_argument("--zoom-amount", type=float, default=0.04)
    ap.add_argument("--fadein", type=float, default=0.35)
    ap.add_argument("--match", type=float, default=8.0)
    ap.add_argument("--pro", action="store_true")
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--uniform-strength", type=float, default=0.7)
    ap.add_argument("--pace", type=float, default=None)
    ap.add_argument("--sat", type=float, default=None)
    ap.add_argument("--bright", type=float, default=None)
    ap.add_argument("--stack", action="store_true")
    ap.add_argument("--text", default="")
    ap.add_argument("--clips", type=int, default=None)
    ap.add_argument("--versions", type=int, default=1)
    ap.add_argument("--tries", type=int, default=None)
    ap.add_argument("--vary", type=float, default=4.0)
    ap.add_argument("--spread", type=float, default=6.0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--each", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hit-ease", default="snapSettle", choices=eases.NAMES + ["old"])
    ap.add_argument("--ramp", type=float, default=0.0)
    ap.add_argument("--endcard", default=None)
    ap.add_argument("--endcard-seconds", type=float, default=2.5)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--edit", default=None)
    ap.add_argument("--pip", action="store_true")
    ap.add_argument("--pip-time", type=float, default=0.6)
    ap.add_argument("--pip-size", type=float, default=0.3)
    ap.add_argument("--pip-every", type=int, default=4)
    ap.add_argument("--pip-at", type=float, action="append", default=[])
    ap.add_argument("--pip-first", type=int, default=None)
    ap.add_argument("--pip-corner", default="", choices=["", "qtl", "qtr", "qbl", "qbr", "tl", "tr", "bl", "br"])
    ap.add_argument("--pip-bars", type=float, default=0.13)
    ap.add_argument("--hearts", action="store_true")
    ap.add_argument("--hearts-time", type=float, default=3.0)
    ap.add_argument("--hearts-at", type=float, default=None)
    ap.add_argument("--glitch-at", type=float, action="append", default=[])
    ap.add_argument("--glitch-image", default=None)
    ap.add_argument("--glitch-time", type=float, default=0.5)
    ap.add_argument("--glitch-hold", type=float, default=0.0)
    ap.add_argument("--glitch-size", type=float, default=0.6)
    ap.add_argument("--bullet", action="store_true")
    ap.add_argument("--bullet-every", type=int, default=4)
    ap.add_argument("--bullet-at", type=float, action="append", default=[])
    ap.add_argument("--bullet-from", type=float, action="append", default=[])
    ap.add_argument("--bullet-time", type=float, default=0.9)
    ap.add_argument("--bullet-solo", action="store_true")
    a = ap.parse_args()
    for spec in a.skip:   # --skip name.mp4:40-50
        name, _, rng = spec.rpartition(":")
        t0, _, t1 = rng.partition("-")
        if not name or not t1:
            sys.exit(f"--skip should look like video.mp4:40-50, got: {spec}")
        SKIP.append((name, float(t0), float(t1)))
    global FPS, MATCH, ZOOM, FADE, PACE, MAXSHOT, DARK, GRADE, STACK, TEXT, SPREAD, HIT_EASE, RAMP, PIP, PIP_SIZE, PIP_CORNER, PIP_BARS, HEARTS, HEARTS_MID, HEARTS_LEN, GLITCH_IMG, GLITCH_LEN, GLITCH_HOLD, GLITCH_SIZE
    HIT_EASE, RAMP = a.hit_ease, max(0.0, min(0.8, a.ramp))
    PIP = max(0.3, min(3.0, a.pip_time)) if a.pip or a.pip_at else 0.0
    PIP_SIZE, PIP_CORNER = max(0.15, min(0.6, a.pip_size)), a.pip_corner
    PIP_BARS = max(0.0, min(0.5, a.pip_bars))
    HEARTS = max(1.0, min(8.0, a.hearts_time)) if a.hearts else 0.0
    HEARTS_LEN = max(1.0, min(8.0, a.hearts_time))
    if a.endcard and not Path(a.endcard).exists():
        sys.exit(f"not found: {a.endcard}")
    card_d = card_length(a.endcard, a.endcard_seconds)
    SPREAD = a.spread
    ZOOM, FADE = (a.zoom_amount if a.zoom or a.pro else 0.0), a.fadein
    PACE = a.pace or (2.0 if a.pro else 1.0)
    MAXSHOT = 6.0 / PACE
    DARK = a.bright if a.bright is not None else (6.0 if a.pro else 0.0)
    sat = a.sat or (1.5 if a.pro else 0)
    GRADE = f"eq=saturation={sat}:contrast=1.10:gamma=1.06,unsharp=5:5:0.5" if sat else ""
    STACK, TEXT = a.stack, a.text
    MATCH = a.match
    try:
        n, d = probe(a.videos[0], "stream=r_frame_rate", "v:0").split("/")
        FPS = max(24, min(60, round(float(n) / float(d))))
    except ValueError:
        pass
    print(f"output frame rate: {FPS} fps")
    song = Path(a.song).resolve()
    if not song.exists():
        sys.exit(f"not found: {song}")
    out = Path(a.out) if a.out else HOME / "clipper/outbox" / f"mashup_{song.stem}.mp4"
    fixed = read_edit(a.edit) if a.edit else None
    if fixed and not a.out:
        ed = Path(a.edit).resolve()
        base = Path(ed.name.removesuffix(".edit.txt")).stem.removesuffix("_edit")
        out = ed.with_name(base + "_edit.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    start, length, segs = plan_song(song, a.length, a.start, card_d)
    if a.clips:
        set_clip_count(segs, a.clips)
    assign_acts(segs, length)
    if a.explosions is not None:
        scatter_explosions(segs, length, a.explosions)
    if a.emotional:
        scatter_emotional(segs, length, a.emotional)
    if a.landscape:
        scatter_landscape(segs, length, a.landscape)
    if a.scuba:
        scatter_scuba(segs, length, a.scuba)
    if a.guns:
        scatter_act(segs, length, a.guns, "guns", 0.25)
    if a.drugs:
        scatter_act(segs, length, a.drugs, "drugs", 0.75)
    if a.planes:
        scatter_act(segs, length, a.planes, "airplane", 0.5)
    if a.boats:
        scatter_act(segs, length, a.boats, "boat", 0.6)
    if a.bikers:
        scatter_act(segs, length, a.bikers, "biker", 0.4)
    if a.chase is not None:
        set_chases(segs, length, a.chase)
    print("footage:")
    picker = Picker(a.videos)
    if a.fresh:
        picker.load_history()
    names = {v.name for v in picker.used}
    for n, _, _ in SKIP:
        if n not in names:
            sys.exit(f"--skip: no video called {n}; the names are: {', '.join(sorted(names))}")

    by_name = {q.name: q for q in picker.dur}
    if fixed:   # --edit: the saved list decides the clips and their order
        for f in fixed:
            if f["name"] not in by_name:
                sys.exit(f"--edit: {f['name']} is in the list but not on the command; "
                         f"the videos are: {', '.join(sorted(by_name))}")
        if len(fixed) != len(segs):   # lines were removed or added: fit the cuts to the list
            set_clip_count(segs, len(fixed))
        if len(fixed) != len(segs):
            sys.exit(f"--edit: the list has {len(fixed)} clips but the song only fits {len(segs)}")
        for s, f in zip(segs, fixed):
            s["act"] = f["act"]
        a.versions, a.tries = 1, 1
    counts = {}
    for s in segs:
        counts[s["act"]] = counts.get(s["act"], 0) + 1
        f0, f1 = round(s["t0"] * FPS), round(s["t1"] * FPS)
        s["frames"] = f1 - f0
        s["speed"] = a.speed if s["act"] in ("rising", "chase", "explosion", "guns") else 1.0
        if a.sync:   # speed follows the music: quiet = slow motion, loud = fast
            fast = a.speed if getattr(a, "speed", 1.0) > 1.0 else 1.5
            s["speed"] = {"low": a.slow, "mid": 1.0}.get(s["level"], fast)
        s["src_d"] = s["frames"] / FPS * s["speed"]
    if a.pro:   # white flash + zoom hit on the strongest beats, hard cut elsewhere, soft dip when quiet
        hits = sorted(s["hit"] for s in segs)
        strong = hits[int(len(hits) * 0.65)]
        for k, s in enumerate(segs):
            if k == 0:
                s["tin"], s["punch"] = "cut", 0.0
            elif s["level"] == "low":
                s["tin"], s["punch"] = "dip", 0.0
            elif s["level"] == "high" and s["hit"] >= strong:
                s["tin"], s["punch"] = "flash", 0.10
            else:
                s["tin"], s["punch"] = "cut", 0.05
            if k and s["act"] == "explosion":   # explosions always get the white flash and a bigger hit
                s["tin"], s["punch"] = "flash", 0.14
    if PIP:   # about every Nth cut, where the shot before it is long enough to show a preview
        since, n_pip = max(1, a.pip_every) - max(1, a.pip_first or a.pip_every), 0   # --pip-first: cut that gets the first one
        forced = set()   # --pip-at: the cut nearest each of those seconds always gets it
        for t in (a.pip_at if len(segs) > 1 else []):
            near = min(range(1, len(segs)), key=lambda j: abs(segs[j]["t0"] - t))
            forced.add(near)
            print(f"--pip-at {t:g}: using the cut at {segs[near]['t0']:.2f}s")
        for k in range(1, len(segs)):
            since += 1
            before = segs[k - 1]["frames"] / FPS
            auto = a.pip and since >= max(1, a.pip_every) and before >= 0.9 and segs[k]["act"] != "explosion"
            if k in forced and before < 0.5:
                print(f"  the shot before {segs[k]['t0']:.2f}s is too short for a preview box, skipped")
            if auto or (k in forced and before >= 0.5):
                segs[k]["pip"] = min(PIP, 0.55 * before)
                segs[k]["tin"], segs[k]["punch"] = "cut", 0.0   # the growing box is the transition
                since, n_pip = 0, n_pip + 1
        print(f"picture-in-picture on {n_pip} of {len(segs) - 1} cuts"
              + ("" if n_pip else " (no shot is long enough: try --pace 1 or a smaller --clips)"))
    if a.bullet or a.bullet_at:
        bullet.choose(segs, FPS, a.bullet, a.bullet_every, a.bullet_at,
                      max(0.5, min(2.5, a.bullet_time)), a.bullet_solo)
    if a.bullet_from:   # the bullet sequence starts on these cuts
        bullet.choose_from(segs, FPS, a.bullet_from, max(0.5, min(2.5, a.bullet_time)), a.bullet_solo)
    if a.hearts_at is not None and len(segs) > 1:   # hearts in the middle: they finish on the cut nearest that second
        k = min(range(1, len(segs)), key=lambda j: abs(segs[j]["t0"] - a.hearts_at))
        HEARTS_MID = round(segs[k]["t0"] * FPS) / FPS
        for key in ("pip", "bullet", "bullet_in"):      # no other transition on that cut
            segs[k].pop(key, None)
        segs[k]["tin"], segs[k]["punch"] = "cut", 0.0
        print(f"--hearts-at {a.hearts_at:g}: the hearts finish on the cut at {HEARTS_MID:.2f}s")
    if a.glitch_image and not Path(a.glitch_image).exists():
        sys.exit(f"not found: {a.glitch_image}")
    if a.glitch_image and not a.glitch_at:
        sys.exit("--glitch-image needs --glitch-at SECONDS to say where the glitch goes")
    GLITCH_IMG = str(Path(a.glitch_image).resolve()) if a.glitch_image else None
    GLITCH_LEN = max(0.1, min(3.0, a.glitch_time))
    GLITCH_HOLD = max(0.0, min(10.0, a.glitch_hold))
    GLITCH_SIZE = max(0.1, min(1.0, a.glitch_size))
    for t in a.glitch_at:   # the glitch starts on the cut nearest each of those seconds
        k = min(range(len(segs)), key=lambda j: abs(segs[j]["t0"] - t))
        at = round(segs[k]["t0"] * FPS) / FPS
        if at not in GLITCH:
            GLITCH.append(at)
        print(f"--glitch-at {t:g}: the glitch starts on the cut at {at:.2f}s")
    out0, made, failed = out, [], []
    tries = a.tries or (6 if a.versions > 1 else 1)
    for ver in range(max(1, a.versions)):
        best = None
        for trial in range(max(1, tries)):   # try several selections, keep the best whole edit
            if fixed:   # --edit: no picking, take the footage from the list
                for s, f in zip(segs, fixed):
                    q = by_name[f["name"]]
                    s["src"], s["score"], s["label"], s["motion"] = q, f["score"], f["label"], f["motion"]
                    s["src_t"] = max(0.0, min(f["t"], picker.dur[q] - s["src_d"] - 0.1))
                best = (edit_score(segs, len(picker.dur)), [dict(x) for x in segs])
                break
            picker.reset(ver * 100 + trial, 0.0 if ver == 0 and trial == 0 else a.vary)
            for x in segs:
                x.pop("only", None)
            if a.each:   # reserve slots so every input video gets at least this many clips
                has = {q: {c[3] for c in picker.cands if c[0] == q} for q in picker.dur}
                open_slots = list(range(len(segs)))
                picker.rng.shuffle(open_slots)
                for q in [q for _ in range(a.each) for q in picker.dur]:
                    if not open_slots:
                        break
                    want = lambda k: bool((picker.ORDER[segs[k]["act"]][0][0] or set()) & has[q])
                    k = next((k for k in open_slots if want(k)), open_slots[0])   # a slot this video suits
                    segs[k]["only"] = q
                    open_slots.remove(k)
            seen = {}
            for act in ("explosion", "airplane", "scuba", "boat", "biker", "emotional", "landscape", "guns", "drugs", "chase", "intro", "rising", "cutscene", "falling"):  # best footage first
                for s in segs:
                    if s["act"] == act:
                        frac = seen.get(act, 0) / counts[act]
                        seen[act] = seen.get(act, 0) + 1
                        s["src"], s["src_t"], s["score"], s["label"], s["motion"] = picker.pick(
                            act, s["src_d"], frac, s["energy"], s.get("only"))
            total = edit_score(segs, len(picker.dur))
            if best is None or total[0] > best[0][0]:
                best = (total, [dict(x) for x in segs])
        for x, b in zip(segs, best[1]):
            x.update(b)
        if PIP or a.bullet or a.bullet_at:
            pip_room(segs, picker, not fixed)
        picker.block(segs)                       # the next version must use different footage
        if a.versions > 1:
            out = out0.with_name(f"{out0.stem}_v{ver + 1}{out0.suffix}")
        t = best[0]
        used = sorted({x["src"].name for x in segs})
        print(f"version {ver + 1}: score {t[0]:.1f}  (footage {t[1]:.1f}, fits song {t[2]:.2f}, "
              f"variety {t[3]:.2f})  from {len(used)} videos, best of {max(1, tries)} tries")
        print("  clips per video: " + ", ".join(
            f"{q.name} {sum(1 for x in segs if x['src'] == q)}" for q in picker.dur))

        if a.uniform:
            print("matching the look of the clips ...", flush=True)
            match_looks(segs, max(0.0, min(1.0, a.uniform_strength)))
        lines = [f"{s['t0']:6.2f}s  {s['t1'] - s['t0']:4.1f}s  {s['act']:<9}  "
                 f"{s['src'].name} @ {s['src_t']:.0f}s  ({s['label']} {s['score']:.0f})  "
                 f"song {s['energy']:.2f} / clip {s['motion']:.2f}" for s in segs]
        print("edit list:\n" + "\n".join(lines))
        Path(str(out) + ".edit.txt").write_text("".join(
            f"{s['t0']:6.2f}s  {s['t1'] - s['t0']:4.1f}s  {s['act']:<9}  "
            f"{s['src'].name} @ {s['src_t']:.2f}s  ({s['label']} {s['score']:.0f})  "
            f"song {s['energy']:.2f} / clip {s['motion']:.2f}\n" for s in segs))
        print(f"edit list saved: {out}.edit.txt")
        if a.logo and not Path(a.logo).exists():
            sys.exit(f"not found: {a.logo}")
        size, total = render(segs, song, start, length, out, a.wide, a.logo, a.endcard, card_d)
        print(f"made {out}")
        made.append(out)
        if not a.no_check and not check.report(check.check(out, total, size, card_d, LUFS), out.name):
            failed.append(out)
        with open(HIST, "a") as h:               # remember what was used, for --fresh next time
            h.write("".join(f"{x['src'].name} {x['src_t']:.1f} {x['src_t'] + x['src_d']:.1f}\n" for x in segs))
    if len(made) > 1:
        if a.review:
            page = review.build(made, title=f"{song.stem} versions")
            print(f"blind comparison page: {page}\n  open it with:  xdg-open {page}")
        else:
            print("compare the versions blind with:\n  python3 " + str(Path(__file__).resolve().parent / "review.py")
                  + " " + " ".join(str(m) for m in made))
    if failed:
        sys.exit("FAILED CHECKS, look before posting: " + ", ".join(f.name for f in failed))


if __name__ == "__main__":
    main()
