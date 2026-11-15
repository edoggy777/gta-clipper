#!/usr/bin/env python3
"""check.py - quality checks on a finished video. mashup.py runs this after every render.

Usage:
  python3 check.py outbox/mashup_song.mp4
  python3 check.py outbox/mashup_song.mp4 --expect 60 --size 1080x1920

Checks:
  file      has picture and sound, right size, right length
  black     no black stretch in the middle of the video
  frozen    the picture never stands still for a second or more
  loudness  about -14 LUFS (what TikTok, Reels and Shorts play at), peaks under -1 dB
  silence   the sound never drops out

Exit code: 0 = everything passed (warnings allowed), 1 = something failed.
"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

FF = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else shutil.which("ffmpeg")
FP = "/usr/bin/ffprobe" if Path("/usr/bin/ffprobe").exists() else shutil.which("ffprobe")
LUFS = -14.0       # loudness target
PEAK = -1.0        # loudest allowed peak, in dB
BLACK = 0.5        # a black stretch this long (seconds) fails
FROZEN = 1.0       # a frozen picture this long (seconds) fails
SILENT = 1.0       # silence this long (seconds) fails
EDGE_IN = 0.6      # the fade-in at the start may be black/quiet for this long
EDGE_OUT = 1.6     # the fade-out at the end may be black/quiet for this long


def _run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def _probe(path, stream, entry):
    out = _run([FP, "-v", "error", "-select_streams", stream, "-show_entries", entry, "-of", "csv=p=0", path]).stdout
    return out.strip().splitlines()[0].strip(",") if out.strip() else ""


def _spans(text, start_key, end_key):
    """Pairs of (start, end) seconds from ffmpeg's detect filters. An open span runs to the end."""
    starts = [float(x) for x in re.findall(start_key + r"[:=]\s*(-?[\d.]+)", text)]
    ends = [float(x) for x in re.findall(end_key + r"[:=]\s*(-?[\d.]+)", text)]
    return [(s, ends[i] if i < len(ends) else None) for i, s in enumerate(starts)]


def check(path, expect=None, size=None, tail=0.0, target=LUFS):
    """Returns a list of (name, "PASS" | "WARN" | "FAIL", detail).
    expect: the length the video should have, in seconds.  size: (width, height) it should have.
    tail: seconds at the end that are a still end card (a frozen picture is fine there)."""
    path = Path(path)
    res = []
    if not path.exists() or path.stat().st_size < 1000:
        return [("file", "FAIL", f"missing or empty: {path}")]

    # ---- file: picture, sound, size, length
    why, bad = [], False
    wh = _probe(path, "v:0", "stream=width,height")
    has_audio = bool(_probe(path, "a:0", "stream=codec_name"))
    try:
        dur = float(_run([FP, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).stdout)
    except ValueError:
        dur = 0.0
    if not wh:
        why.append("no picture"); bad = True
    if not has_audio:
        why.append("no sound"); bad = True
    if wh and size and wh != f"{size[0]},{size[1]}":
        why.append(f"size is {wh.replace(',', 'x')}, wanted {size[0]}x{size[1]}"); bad = True
    if expect and abs(dur - expect) > 0.3:
        why.append(f"length is {dur:.2f}s, wanted {expect:.2f}s"); bad = True
    res.append(("file", "FAIL" if bad else "PASS",
                "; ".join(why) if why else f"{wh.replace(',', 'x')}, {dur:.2f}s, picture and sound"))
    if not wh or dur <= 0:
        return res

    # ---- one pass through the video for everything else
    graph = f"[0:v]blackdetect=d=0.1:pix_th=0.10,freezedetect=n=-50dB:d={FROZEN}[v]"
    maps = ["-map", "[v]"]
    if has_audio:
        graph += f";[0:a]ebur128=peak=true,silencedetect=n=-50dB:d={SILENT}[a]"
        maps += ["-map", "[a]"]
    log = _run([FF, "-nostats", "-v", "info", "-i", path, "-filter_complex", graph, *maps, "-f", "null", "-"]).stderr
    # a stretch counts unless it sits wholly inside the fade-in or the fade-out
    middle = lambda s, e: not ((e if e is not None else dur) <= EDGE_IN + 0.05 or s >= dur - EDGE_OUT - 0.05)
    span = lambda s, e: f"{s:.1f}s-{(e if e is not None else dur):.1f}s"

    # ---- black
    blacks = [(s, e) for s, e in _spans(log, "black_start", "black_end") if middle(s, e)]
    long_black = [(s, e) for s, e in blacks if (e if e is not None else dur) - s >= BLACK]
    if long_black:
        res.append(("black", "FAIL", "black picture at " + ", ".join(span(s, e) for s, e in long_black[:6])))
    elif blacks:
        res.append(("black", "WARN", "short black flashes at " + ", ".join(span(s, e) for s, e in blacks[:6])))
    else:
        res.append(("black", "PASS", "no black stretches"))

    # ---- frozen (a still end card in the last `tail` seconds is allowed)
    limit = dur - tail - 0.1
    frozen = [(s, e) for s, e in _spans(log, "freeze_start", "freeze_end") if s < limit - FROZEN]
    if frozen:
        res.append(("frozen", "FAIL", "picture stands still at " + ", ".join(span(s, e) for s, e in frozen[:6])))
    else:
        res.append(("frozen", "PASS", "picture keeps moving"))

    if not has_audio:
        return res

    # ---- loudness
    loud = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", log)
    if not loud:
        res.append(("loudness", "WARN", "could not measure"))
    else:
        i, p = float(loud[-1]), (float(peak[-1]) if peak else None)
        off = abs(i - target)
        state = "PASS" if off <= 1.0 else "WARN" if off <= 2.5 else "FAIL"
        note = f"{i:.1f} LUFS (target {target:.0f})"
        if p is not None:
            note += f", peak {p:.1f} dB"
            if p > PEAK:
                state = "FAIL" if p > 0 else "WARN" if state == "PASS" else state
                note += f" (should be under {PEAK:.0f})"
        res.append(("loudness", state, note))

    # ---- silence
    quiet = [(s, e) for s, e in _spans(log, "silence_start", "silence_end") if middle(s, e)]
    if quiet:
        res.append(("silence", "FAIL", "no sound at " + ", ".join(span(s, e) for s, e in quiet[:6])))
    else:
        res.append(("silence", "PASS", "sound all the way through"))
    return res


def report(results, name=""):
    """Prints the results. Returns True if nothing failed."""
    print(f"checks{' for ' + name if name else ''}:")
    for gate, state, detail in results:
        print(f"  {state:<4}  {gate:<9} {detail}")
    return not any(state == "FAIL" for _, state, _ in results)


def main():
    ap = argparse.ArgumentParser(description="Quality checks on a finished video.")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--expect", type=float, default=None, help="length the video should have, in seconds")
    ap.add_argument("--size", default=None, help="size the video should have, like 1080x1920")
    ap.add_argument("--tail", type=float, default=0.0, help="seconds of still end card at the end")
    ap.add_argument("--lufs", type=float, default=LUFS, help="loudness target (default -14)")
    a = ap.parse_args()
    size = tuple(int(x) for x in a.size.lower().split("x")) if a.size else None
    ok = True
    for v in a.videos:
        ok = report(check(v, a.expect, size, a.tail, a.lufs), Path(v).name) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
