#!/bin/bash
# Stamps the logo in the center of every clip in outbox/, saves copies to outbox/logo/
LOGO=/home/edog/clipper/logo/GTA6-Logo.png
IN=/home/edog/clipper/outbox
OUT=/home/edog/clipper/outbox/logo
SECS=1      # how long the logo shows; set to 0 to keep it on for the whole clip
WIDTH=700   # logo width in pixels (clip is 1080 wide)

mkdir -p "$OUT"
if [ "$SECS" = "0" ]; then EN=""; else EN=":enable='lt(t,$SECS)'"; fi

for f in "$IN"/*.mp4; do
  name=$(basename "$f")
  /usr/bin/ffmpeg -y -loglevel error -i "$f" -i "$LOGO" \
    -filter_complex "[1:v]scale=$WIDTH:-1[lg];[0:v][lg]overlay=(W-w)/2:(H-h)/2$EN[v]" \
    -map "[v]" -map 0:a? -c:v libx264 -preset fast -crf 18 -c:a copy \
    "$OUT/$name" && echo "made $OUT/$name"
done
