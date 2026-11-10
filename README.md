# gta-clipper

Local, fully offline pipeline that turns long gameplay recordings into vertical short-form clips.

- **vlm_score.py** — Qwen2.5-VL scores every 2 s frame and labels it (guns / drugs / violence / chase / other)
- **pipeline.py** — blends VLM + loudness scores, picks the top non-overlapping moments, calls polish
- **polish.py** — YOLOv8 subject-tracked 9:16 crop, tight cut around the peak, punch-in zoom, word-by-word karaoke captions (faster-whisper + ASS), optional hook text and impact SFX, loudness-normalized audio
- **watch.sh / clipper.service** — inotify watcher so anything dropped in `inbox/` gets processed

## Setup

python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
sudo apt install ffmpeg inotify-tools
mkdir -p inbox outbox done

Hook text uses a local LLM via `ollama run qwen2.5:7b`; edit `gen_hook()` in polish.py to change that.

## Use

python3 pipeline.py inbox/recording.mp4 # 3 clips -> outbox/
python3 polish.py --src in.mp4 --start 125 --out clip.mp4 --hook “TEXT”

Source video must be H.264 (OpenCV can't decode AV1). Knobs are at the top of each script.

## Notes
Only clip footage you have rights to (your own gameplay, or a campaign that licenses it). Models (Qwen2.5-VL, YOLOv8, Whisper) download on first run under their own licenses.
