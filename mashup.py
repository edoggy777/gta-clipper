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
"""
import argparse, bisect, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HOME = Path.home()
FF = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else shutil.which("ffmpeg")
FP = "/usr/bin/ffprobe" if Path("/usr/bin/ffprobe").exists() else shutil.which("ffprobe")
VLM = HOME / "clipper/vlm_score.py"
WORK = HOME / "clipper/work"
SR, HOP, NFFT, FPS = 22050, 512, 1024, 30
MOVE = 8


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


def plan_song(song, length, start):
    """Returns (start, length, segments). Each segment: dict(t0, t1, energy, hit)."""
    y = load_audio(song)
    onset, rms = analyse(y)
    fps = SR / HOP
    dur = len(y) / SR
    beats, bpm = find_beats(onset)
    bt = beats / fps
    loud = smooth(rms, fps * 2)
    length = min(length, dur)
    if start is None:  # put the first loud section about a third of the way in
        drop = np.argmax(smooth(rms, fps * 4) >= 0.7 * smooth(rms, fps * 4).max()) / fps
        start = max(0.0, drop - 0.35 * length)
        if len(bt):
            start = float(bt[np.argmin(np.abs(bt - start))])
    start = max(0.0, min(start, dur - length))
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
        if not segs:
            n = max(n, 8)
        end = lambda k: b[i + k] if i + k < len(b) else length
        t0 = 0.0 if not segs else b[i]
        while end(n) - t0 < 0.7 and i + n < len(b):
            n *= 2
        while n > 1 and end(n) - t0 > 6.0:
            n //= 2
        level = "low" if e[i:i + n].mean() < lo else "mid" if e[i:i + n].mean() < hi else "high"
        segs.append(dict(t0=t0, t1=end(n), level=level, hit=float(hit[i])))
        i += n
    segs[-1]["t1"] = length
    if len(segs) > 1 and segs[-1]["t1"] - segs[-1]["t0"] < 0.7:
        segs.pop()
        segs[-1]["t1"] = length
    print(f"song: {bpm:.0f} BPM, using {start:.1f}s to {start + length:.1f}s, {len(segs)} cuts")
    return start, length, segs


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
        "falling": [({"other"}, True), (None, False)],
    }

    def __init__(self, videos):
        self.cands, self.dur, self.used, self.cuts = [], {}, {}, {}
        for v in videos:
            path, rows = prepare(v)
            self.dur[path] = float(probe(path, "format=duration") or 0)
            self.used[path] = []
            self.cuts[path] = scene_cuts(path, self.dur[path])
            mot = motion_curve(path)
            self.cands += [(path, t, sc + MOVE * mot(t), lab) for t, sc, lab in rows]
            labels = [lab for _, _, lab in rows]
            print(f"  {path.name}: " + ", ".join(f"{labels.count(x)} {x}" for x in sorted(set(labels))))
        if not self.cands:
            sys.exit("no scored frames in any input video")

    def free(self, path, s, d):
        if any(Path(path).name == n and s < b and s + d > a for n, a, b in SKIP):
            return False
        return all(s + d + 0.5 <= a or s >= b + 0.5 for a, b in self.used[path])

    def pick(self, act, d, frac):
        for attempt in range(4):
            for labels, desc in self.ORDER[act]:
                pool = [c for c in self.cands if labels is None or c[3] in labels]
                pool.sort(key=lambda c: -c[2] if desc else c[2])
                if act == "rising" and labels and "chase" in labels:
                    pool = pool[:max(1, int(len(pool) * 0.6))]  # keep the best chases for the chase
                first = int(frac * len(pool)) if act in ("rising", "falling") else 0
                for path, t, sc, lab in pool[first:] + pool[:first]:
                    if self.dur[path] < d + 0.2:
                        continue
                    cuts = self.cuts[path]
                    k = bisect.bisect_right(cuts, t) - 1
                    a, b = cuts[k], cuts[min(k + 1, len(cuts) - 1)]
                    if b - a >= d + 0.3:      # keep the whole clip inside one shot
                        s = min(max(a + 0.15, t - d / 2), b - d - 0.15)
                    elif attempt % 2 == 0:    # shot too short: try other moments first
                        continue
                    else:
                        s = min(max(0.0, t + 1 - d / 2), self.dur[path] - d - 0.1)
                    if self.free(path, s, d):
                        self.used[path].append((s, s + d))
                        return path, s, sc, lab
            if attempt % 2:  # everything used once: allow reuse
                for p in self.used:
                    self.used[p] = []
        sys.exit("input videos are too short for this edit - add more footage or use a smaller --length")


# ---------------------------------------------------------------- render
def render(segs, song, start, length, out, wide, logo=None):
    W, H = (1920, 1080) if wide else (1080, 1920)
    tmp = Path(tempfile.mkdtemp(prefix="mashup_"))
    parts = []
    for k, s in enumerate(segs):
        zoom = 1.12 if s["act"] == "explosion" else 1.0   # punch in on explosions
        zw, zh = int(W * zoom / 2) * 2, int(H * zoom / 2) * 2
        vf = f"scale={zw}:{zh}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
        if s["speed"] != 1:                                # slight slow motion as tension falls
            vf += f",setpts=PTS/{s['speed']}"
        vf += f",fps={FPS},format=yuv420p"
        part = tmp / f"seg_{k:03d}.mp4"
        r = run([FF, "-y", "-v", "error", "-ss", f"{s['src_t']:.3f}", "-t", f"{s['src_d'] + 0.5:.3f}",
                 "-i", s["src"], "-an", "-vf", vf, "-frames:v", s["frames"], "-c:v", "libx264",
                 "-preset", "veryfast", "-crf", "14", part], text=True)
        if r.returncode or not part.exists():
            sys.exit(f"cut failed on {s['src']}: {r.stderr[-1500:]}")
        parts.append(part)
        print(f"\r  cutting {k + 1}/{len(segs)}", end="", flush=True)
    print()
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    fc = (f"[0:v]fade=t=in:st=0:d=0.4,fade=t=out:st={length - 1:.2f}:d=1[v];"
          f"[1:a]afade=t=in:st=0:d=0.3,afade=t=out:st={length - 1.5:.2f}:d=1.5,loudnorm=I=-14:TP=-1.5[a]")
    extra = []
    if logo:   # logo in the centre: fades in, holds, then fades out while the video keeps playing
        hold, fade = 2.5, 1.0
        extra = ["-loop", "1", "-t", f"{hold + fade + 0.5}", "-i", logo]
        fc = fc.replace("[v];", "[base];", 1) + (
            f";[2:v]format=rgba,scale={int(W * 0.7)}:-1,fade=t=in:st=0:d=0.3:alpha=1,"
            f"fade=t=out:st={hold}:d={fade}:alpha=1[lg];[base][lg]overlay=(W-w)/2:(H-h)/2:eof_action=pass[v]")
    r = run([FF, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
             "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", song, *extra, "-filter_complex", fc,
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out],
            text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode:
        sys.exit(f"final render failed: {r.stderr[-1500:]}")


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
    ap.add_argument("--guns", type=int, default=None)
    ap.add_argument("--drugs", type=int, default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for spec in a.skip:   # --skip name.mp4:40-50
        name, _, rng = spec.rpartition(":")
        t0, _, t1 = rng.partition("-")
        if not name or not t1:
            sys.exit(f"--skip should look like video.mp4:40-50, got: {spec}")
        SKIP.append((name, float(t0), float(t1)))
    global FPS
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
    out.parent.mkdir(parents=True, exist_ok=True)

    start, length, segs = plan_song(song, a.length, a.start)
    assign_acts(segs, length)
    if a.explosions is not None:
        scatter_explosions(segs, length, a.explosions)
    if a.emotional:
        scatter_emotional(segs, length, a.emotional)
    if a.guns:
        scatter_act(segs, length, a.guns, "guns", 0.25)
    if a.drugs:
        scatter_act(segs, length, a.drugs, "drugs", 0.75)
    if a.chase is not None:
        set_chases(segs, length, a.chase)
    print("footage:")
    picker = Picker(a.videos)
    names = {v.name for v in picker.used}
    for n, _, _ in SKIP:
        if n not in names:
            sys.exit(f"--skip: no video called {n}; the names are: {', '.join(sorted(names))}")

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
    seen = {}
    for act in ("explosion", "emotional", "guns", "drugs", "chase", "intro", "rising", "cutscene", "falling"):  # best footage first
        for s in segs:
            if s["act"] == act:
                frac = seen.get(act, 0) / counts[act]
                seen[act] = seen.get(act, 0) + 1
                s["src"], s["src_t"], s["score"], s["label"] = picker.pick(act, s["src_d"], frac)

    lines = [f"{s['t0']:6.2f}s  {s['t1'] - s['t0']:4.1f}s  {s['act']:<9}  "
             f"{s['src'].name} @ {s['src_t']:.0f}s  ({s['label']} {s['score']:.0f})" for s in segs]
    print("edit list:\n" + "\n".join(lines))
    Path(str(out) + ".edit.txt").write_text("\n".join(lines) + "\n")
    if a.logo and not Path(a.logo).exists():
        sys.exit(f"not found: {a.logo}")
    render(segs, song, start, length, out, a.wide, a.logo)
    print(f"made {out}")


if __name__ == "__main__":
    main()
