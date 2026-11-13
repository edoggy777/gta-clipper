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
- ffmpeg and ffprobe
- A local vision model for `vlm_score.py`

## Footage and music

Only publish footage and music you have the right to use. Game publishers
and music labels have their own rules about reposting and monetizing
their content, and platforms remove videos that break them.

## License

MIT
