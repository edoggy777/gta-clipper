# gta-clipper

Turns long gameplay videos into a short, music-paced mashup. You give it a
song and one or more videos; it finds the beats, rates every moment of the
footage with a local vision model, and cuts the best moments to the music.

Everything runs locally. No cloud services.

## Quick start

````bash
python3 mashup.py --song music/song.mp3 done/video1.mp4 done/video2.mp4
````

The result is saved to `outbox/mashup_<song name>.mp4`, with the cut list
next to it as `outbox/mashup_<song name>.mp4.edit.txt`.

## How it works

1. **Song.** Finds the beats and how loud each part is, then picks a window
   of the song with the first loud section about a third of the way in.
2. **Pacing.** Quiet parts get long shots, medium parts shorter ones, loud
   parts fast cuts. Every cut lands on a beat.
3. **Scoring.** `vlm_score.py` shows frames of each video to a vision model,
   which labels each one and rates how clip-worthy it is.
4. **Story.** Each slot in the song gets a role, and the script fills it
   with the best matching footage that has not been used yet.
5. **Render.** ffmpeg cuts and joins the shots and lays the song on top.

Scoring is slow the first time. The results are cached in a
`<video>.scores.txt` file next to each video, so later runs are fast.

### Footage labels

`explosion`, `guns`, `drugs`, `violence`, `chase`, `emotional`, `other`

### Slot roles

| Role | Where it goes | Footage it prefers |
|---|---|---|
| `intro` | Opening shot | Calm establishing shots |
| `rising` | Medium parts | Milder driving footage |
| `chase` | Loud parts | The best chase footage |
| `explosion` | Hardest hits | Explosions |
| `guns` | Chase slots, with `--guns` | Gunfights |
| `drugs` | Chase slots, with `--drugs` | Drug scenes |
| `emotional` | Quiet parts, with `--emotional` | Close-ups, quiet dramatic moments |
| `cutscene` | Quiet breaks | Calm shots |
| `falling` | After the last loud part | Calm shots |

If the videos run out of a kind of footage, the slot is filled with the
next best thing. The label in brackets in the edit list shows what each
shot really is.

## Options

Videos go at the end with no option name. Use as many as you like.

| Option | What it does | Default |
|---|---|---|
| `--song FILE` | Song to cut to (required) | |
| `--length N` | Length of the result in seconds | 60 |
| `--start N` | Second of the song to start from | automatic |
| `--out FILE` | Where to save the result | `outbox/mashup_<song>.mp4` |
| `--wide` | Keep the video widescreen | off |
| `--logo FILE.png` | Logo at the start, fading out over the video | none |
| `--explosions N` | Exactly N explosion shots, spread evenly | automatic |
| `--guns N` | N gunfight shots, spread evenly | none |
| `--drugs N` | N drug-scene shots, spread evenly | none |
| `--emotional N` | N emotional shots in the quiet parts | none |
| `--chase N` | Exactly N chase shots | automatic |
| `--speed N` | Speed of chase, explosion and guns shots | 1.0 |
| `--sync` | Speed follows the music (see below) | off |
| `--slow N` | Slow-motion speed for quiet parts with `--sync` | 0.75 |
| `--skip NAME:FROM-TO` | Never use this stretch of a video | none |

### Order of placement

Explosions are placed first, then emotional, guns and drugs shots.
`--chase` runs last and only adjusts the slots that are left. A 60-second
video has roughly 20 to 30 slots depending on the song's tempo, so the
numbers you ask for have to fit.

### Speed

Without `--sync`, `--speed` applies to chase, explosion and guns shots.
Everything else plays at normal speed.

With `--sync`, speed depends only on the music:

| Part of the song | Speed |
|---|---|
| Quiet | `--slow` (slow motion) |
| Medium | normal |
| Loud | `--speed`, or 1.5 if not given |

Slow motion below about 0.6 looks choppy on 30 fps footage.

### Skipping footage

Copy the video name and second from the edit list, and block a few
seconds either side:

````bash
--skip video2.mp4:1128-1140
````

Use `--skip` more than once to block several stretches. A name that does
not match any input video stops the script with a list of valid names.

## Full example

````bash
python3 mashup.py \
  --song music/song.mp3 \
  --length 60 \
  --explosions 4 --emotional 5 --guns 4 --drugs 3 --chase 4 \
  --sync --speed 1.7 --slow 0.75 \
  --logo logo/logo.png \
  --skip video2.mp4:1128-1140 \
  done/video1.mp4 done/video2.mp4
````

## Reading the edit list

````
 15.51s   3.9s  chase      video1.mp4 @ 746s  (chase 16)
````

From left to right: when the shot starts in the result, how long it lasts,
the slot's role, the source video and second, and the footage's real label
and score.

## Re-scoring videos

If you change the labels in `vlm_score.py`, old cached scores will not
include them. Move the cache files aside and run again:

````bash
mkdir -p old_scores
mv done/*.scores.txt old_scores/
````

## Folders

| Folder | Use |
|---|---|
| `inbox/` | New videos and songs |
| `done/` | Source videos and their score caches |
| `music/` | Songs |
| `logo/` | Logo images |
| `outbox/` | Finished mashups and edit lists |

## Requirements

- Python 3 with NumPy
- `eases.py`, `check.py` and `review.py` in the same folder as `mashup.py`
- ffmpeg and ffprobe
- A local vision model for `vlm_score.py`

## Footage and music

Only publish footage and music you have the right to use. Game publishers
and music labels have their own rules about reposting and monetizing
their content, and platforms remove videos that break them.

## License

MIT

## mashup.py: cut several videos into one edit paced to a song

```
python3 mashup.py --song song.mp3 video1.mp4 video2.mp4 ...
```
Cuts land on the song's beats. The first run scores each video with the vision
model (slow, once per video); later runs reuse the saved scores.

### Basics
| Switch | What it does |
|---|---|
| `--length 60` | seconds of the song to use |
| `--start 30` | where in the song to begin (default: chosen around the drop) |
| `--wide` | 1920x1080 output (default is vertical 1080x1920) |
| `--out FILE` | output file (default `~/clipper/outbox/mashup_<song>.mp4`) |
| `--logo FILE.png` | logo that fades in and out at the start |
| `--text "CAPTION"` | caption in the middle of the video (rises into place and fades in) |
| `--endcard FILE` | picture or short video shown after the last shot, song keeps playing |
| `--endcard-seconds 2.5` | how long a picture end card stays up (a video uses its own length, 8 s at most) |

### What goes in the edit
| Switch | What it does |
|---|---|
| `--clips 30` | exact number of clips |
| `--explosions 6` | number of explosion shots, spread evenly |
| `--chase 8`, `--guns 12`, `--drugs 2`, `--emotional 3` | number of each shot type |
| `--skip video.mp4:40-50` | never use seconds 40 to 50 of that video (repeatable) |
| `--each 5` | at least 5 clips from every input video |
| `--spread 6` | how strongly clips are shared between videos (0 = off) |
| `--fresh` | stay away from clips used in earlier runs (`used_clips.txt`) |

### Fit to the music
| Switch | What it does |
|---|---|
| `--match 8` | prefer footage whose movement matches the song's loudness (0 = off) |
| `--pace 2` | cut twice as often |
| `--speed 1.7` | play action clips faster |
| `--sync` | speed follows loudness: quiet = `--slow` (0.75), loud = `--speed` |
| `--ramp 0.4` | speed ramp on chase, guns and explosion clips: they start fast and settle (0 = off, 0.8 = strongest) |

### Look
| Switch | What it does |
|---|---|
| `--pro` | faster cuts, colour grade, flash and zoom-hit transitions, avoids dark footage |
| `--uniform` | match brightness and colour of all clips (`--uniform-strength 0.7`) |
| `--zoom` | slow zoom-in on every clip (`--zoom-amount 0.04`) |
| `--fadein 0.35` | fade-in length on each clip, in seconds (0 = off) |
| `--sat 1.5` | colour strength |
| `--bright 6` | how strongly dark footage is avoided |
| `--stack` | whole wide picture in the middle with blurred fill above and below |
| `--hit-ease snapSettle` | curve the zoom hit settles with; any name from `eases.py`, or `old` for the previous look |

Black bars baked into the source are cropped automatically.

### Several versions
| Switch | What it does |
|---|---|
| `--versions 3` | make three videos from different footage, each scored as a whole edit |
| `--tries 6` | selections tested per version |
| `--vary 4` | how much the versions differ |
| `--review` | also make a blind comparison page of the versions |

### Checks
Every finished video is checked automatically. `--no-check` turns this off.

| Check | Fails when |
|---|---|
| file | no picture or sound, wrong size, wrong length |
| black | black picture for half a second or more (outside the fades) |
| frozen | picture stands still for a second or more (a still end card is allowed) |
| loudness | far from -14 LUFS, or peaks above 0 dB |
| silence | no sound for a second or more |

A failed check does not delete the video. The script finishes the other
versions, then stops with the names of the ones to look at.

The sound is set to -14 LUFS in two passes: the song is measured first, then
one fixed volume change is applied, so the volume does not pump.

### Example
```
python3 mashup.py --song song.mp3 --pro --uniform --versions 3 --clips 30 \
  --each 5 --explosions 6 --guns 12 --chase 8 --speed 1.7 \
  --text "COMING SOON" video1.mp4 video2.mp4 video3.mp4 \
  --skip video3.mp4:220-230
```

## check.py: check any finished video

```
python3 check.py outbox/mashup_song.mp4
python3 check.py outbox/mashup_song.mp4 --expect 60 --size 1080x1920 --tail 2.5
```
`--tail` is the length of a still end card, so it is not reported as frozen.
Exit code 0 means nothing failed.

## review.py: compare versions blind

```
python3 review.py outbox/mashup_song_v1.mp4 outbox/mashup_song_v2.mp4 outbox/mashup_song_v3.mp4
xdg-open outbox/review/index.html
```
The page shows the videos as A, B, C in a random order, plays them in sync,
and only shows the real file names after you press Reveal.

## eases.py: easing curves

Twelve motion curves used for the zoom hit, the speed ramp and the caption.
`python3 eases.py` prints them. The curve maths is ported from
[motionmaxxing](https://github.com/Tejashmakwana/motionmaxxing) (Apache-2.0).

Only use footage and music you have the rights to.
