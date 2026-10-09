#!/bin/sh
# Draft voiceover with a macOS voice: one clip per line (lines/<voice>/NN.m4a) plus a joined read with short pauses.
#   sh make_vo.sh "Samantha" 180
set -e
VOICE="$1"; RATE="${2:-180}"
cd "$(dirname "$0")"
SLUG=$(echo "$VOICE" | tr -cd '[:alnum:]')
OUT="lines/$SLUG"; mkdir -p "$OUT"
i=0; LIST="$OUT/list.txt"; : > "$LIST"
ffmpeg -loglevel error -y -f lavfi -i anullsrc=r=44100:cl=mono -t 0.45 "$OUT/gap.wav"
grep '^| [0-9]' script.md | awk -F'|' '{gsub(/^ +| +$/,"",$4); print $4}' | while IFS= read -r LINE; do
  i=$((i+1)); N=$(printf %02d $i)
  say -v "$VOICE" -r "$RATE" -o "$OUT/$N.aiff" "$LINE"
  ffmpeg -loglevel error -y -i "$OUT/$N.aiff" -ar 44100 -ac 1 "$OUT/$N.wav"
  rm "$OUT/$N.aiff"
  printf "file '%s'\nfile 'gap.wav'\n" "$N.wav" >> "$LIST"
done
ffmpeg -loglevel error -y -f concat -safe 0 -i "$LIST" -c:a aac -b:a 160k "vo_$SLUG.m4a"
printf "%s: %.1fs\n" "$VOICE" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "vo_$SLUG.m4a")"
