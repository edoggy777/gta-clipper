#!/usr/bin/env python3
"""review.py - blind side-by-side page for comparing versions of a video.

You see the videos as A, B, C ... in a random order, without knowing which file is which.
Pick the best one, then press Reveal to see the real names. That way you judge the edit,
not the file name.

Usage:
  python3 review.py outbox/mashup_song_v1.mp4 outbox/mashup_song_v2.mp4 outbox/mashup_song_v3.mp4
  python3 review.py a.mp4 b.mp4 --out /home/edog/clipper/outbox/review

Writes  <out>/index.html      open this in a browser
        <out>/videos/A.mp4 ...  links to your videos (no extra disk space when possible)
The default <out> is a "review" folder next to the first video.
"""
import argparse, base64, html, json, os, random, shutil, sys
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind review</title>
<style>
  body{margin:0;padding:20px;background:#111;color:#eee;font:15px/1.4 system-ui,sans-serif}
  h1{font-size:18px;margin:0 0 4px} p{margin:0 0 14px;color:#aaa}
  .bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
  button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:7px 12px;font:inherit;cursor:pointer}
  button:hover{background:#383838} button.on{background:#2f6f3f;border-color:#4a9a5c}
  .row{display:flex;gap:14px;flex-wrap:wrap}
  .card{flex:1 1 220px;max-width:420px;background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:10px}
  .card.best{border-color:#4a9a5c}
  .card h2{margin:0 0 8px;font-size:16px;display:flex;justify-content:space-between}
  video{width:100%;max-height:68vh;background:#000;border-radius:4px}
  .card .bar{margin:8px 0 0} textarea{width:100%;box-sizing:border-box;margin-top:8px;background:#111;color:#eee;
    border:1px solid #444;border-radius:6px;padding:6px;font:inherit;min-height:44px}
  #out{white-space:pre-wrap;background:#1b1b1b;border:1px solid #333;border-radius:8px;padding:10px;margin-top:14px;display:none}
  .name{color:#8fd19e;font-weight:normal;font-size:13px}
</style></head><body>
<h1>__TITLE__</h1>
<p>Same edit, different versions, in a random order. Watch with the sound off first, then listen to each one. Pick the best, then Reveal.</p>
<div class="bar">
  <button id="play">Play all</button><button id="restart">Restart</button>
  <button id="slow">Speed: 1x</button><button id="reveal">Reveal names</button><button id="copy">Copy verdict</button>
</div>
<div class="row" id="row"></div>
<div id="out"></div>
<script>
const LABELS = __LABELS__, KEY = "__KEY__";
const row = document.getElementById('row'), out = document.getElementById('out');
let best = null, shown = false, speeds = [1, 0.5, 0.25], si = 0;
const vids = [], notes = {};
LABELS.forEach((L, i) => {
  const c = document.createElement('div'); c.className = 'card'; c.id = 'card' + L;
  c.innerHTML = `<h2><span>${L}</span><span class="name" id="name${L}"></span></h2>
    <video src="videos/${L}.mp4" playsinline preload="auto" ${i ? 'muted' : ''}></video>
    <div class="bar"><button data-pick="${L}">Best</button><button data-solo="${L}">Hear this one</button></div>
    <textarea placeholder="Worst moment / notes for ${L}"></textarea>`;
  row.appendChild(c);
  vids.push(c.querySelector('video'));
  c.querySelector('textarea').oninput = e => notes[L] = e.target.value;
});
const lead = vids[0];
document.getElementById('play').onclick = e => {
  const go = lead.paused;
  vids.forEach(v => { v.currentTime = lead.currentTime; go ? v.play() : v.pause(); });
  e.target.textContent = go ? 'Pause all' : 'Play all';
};
document.getElementById('restart').onclick = () => vids.forEach(v => { v.currentTime = 0; });
document.getElementById('slow').onclick = e => {
  si = (si + 1) % speeds.length; vids.forEach(v => v.playbackRate = speeds[si]);
  e.target.textContent = 'Speed: ' + speeds[si] + 'x';
};
lead.onended = () => { vids.forEach(v => { v.currentTime = 0; v.play(); }); };
row.onclick = e => {
  const p = e.target.dataset.pick, s = e.target.dataset.solo;
  if (p) { best = p; LABELS.forEach(L => {
    document.getElementById('card' + L).classList.toggle('best', L === p);
    document.querySelector(`[data-pick="${L}"]`).classList.toggle('on', L === p); }); }
  if (s) LABELS.forEach((L, i) => { vids[i].muted = L !== s;
    document.querySelector(`[data-solo="${L}"]`).classList.toggle('on', L === s); });
};
const names = () => JSON.parse(atob(KEY));
const verdict = () => {
  const n = shown ? names() : {};
  return 'Blind review: __TITLE__\n' + 'Best: ' + (best ? best + (n[best] ? ' = ' + n[best] : '') : '(not picked)') + '\n' +
    LABELS.map(L => `${L}${n[L] ? ' = ' + n[L] : ''}: ${notes[L] || '-'}`).join('\n');
};
document.getElementById('reveal').onclick = () => {
  if (!best && !confirm('You have not picked a best one yet. Reveal anyway?')) return;
  shown = true; const n = names();
  LABELS.forEach(L => document.getElementById('name' + L).textContent = n[L]);
  out.style.display = 'block'; out.textContent = verdict();
};
document.getElementById('copy').onclick = () => {
  out.style.display = 'block'; out.textContent = verdict();
  if (navigator.clipboard) navigator.clipboard.writeText(verdict());
};
</script></body></html>
"""


def build(videos, out=None, title=None):
    """Makes the review page. Returns the path of index.html."""
    videos = [Path(v).resolve() for v in videos]
    for v in videos:
        if not v.exists():
            sys.exit(f"not found: {v}")
    if len(videos) < 2:
        sys.exit("give at least two videos to compare")
    if len(videos) > 8:
        sys.exit("at most 8 videos at a time")
    out = Path(out).resolve() if out else videos[0].parent / "review"
    vdir = out / "videos"
    shutil.rmtree(vdir, ignore_errors=True)
    vdir.mkdir(parents=True, exist_ok=True)
    order = list(videos)
    random.SystemRandom().shuffle(order)
    labels = [chr(ord("A") + i) for i in range(len(order))]
    key = {}
    for lab, v in zip(labels, order):
        dst = vdir / f"{lab}.mp4"
        try:
            os.link(v, dst)          # same file under a second name: no extra disk space
        except OSError:
            shutil.copy2(v, dst)     # different disk: plain copy
        key[lab] = v.name
    page = (PAGE.replace("__TITLE__", html.escape(title or videos[0].parent.name + " versions"))
            .replace("__LABELS__", json.dumps(labels))
            .replace("__KEY__", base64.b64encode(json.dumps(key).encode()).decode()))
    index = out / "index.html"
    index.write_text(page)
    return index


def main():
    ap = argparse.ArgumentParser(description="Blind side-by-side page for comparing versions of a video.")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default=None, help="folder for the page (default: review/ next to the first video)")
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    index = build(a.videos, a.out, a.title)
    print(f"made {index}")
    print(f"open it with:  xdg-open {index}")


if __name__ == "__main__":
    main()
