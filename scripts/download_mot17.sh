#!/usr/bin/env bash
# Download MOT17 train split (~1.8 GB)
# Usage: bash scripts/download_mot17.sh data/MOT17

set -e
DEST=${1:-data/MOT17}
mkdir -p "$DEST"

echo "Downloading MOT17 train split to $DEST ..."
wget -q --show-progress -O /tmp/MOT17.zip \
  "https://motchallenge.net/data/MOT17.zip"

echo "Extracting ..."
unzip -q /tmp/MOT17.zip -d /tmp/MOT17_raw
mv /tmp/MOT17_raw/MOT17/train "$DEST/train"
rm -rf /tmp/MOT17.zip /tmp/MOT17_raw

echo "Done. Sequences:"
ls "$DEST/train"
