#!/usr/bin/env bash
set -euo pipefail
revision=84b1eac0887756b4f1a9d7a1ff49105939749ed2
root="${1:-NO_TRACK/reference}"
mkdir -p "$root/models/bs_roformer"
for file in bs_roformer.py mel_band_roformer.py attend.py; do
  curl --fail --location --retry 3 "https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/$revision/models/bs_roformer/$file" \
    --output "$root/models/bs_roformer/$file"
done
