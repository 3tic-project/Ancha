#!/usr/bin/env bash
set -euo pipefail
destination="${1:-NO_TRACK/models/bs_leap_xe_voc.ckpt}"
expected=b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d
url=https://huggingface.co/pcunwa/BS-Roformer-Leap/resolve/4e47d6662ae82eaa8b4ac4329fe66099a843b48e/Xe/bs_leap_xe_voc.ckpt
mkdir -p "$(dirname "$destination")"
if [[ -e "$destination" ]]; then
  echo "File already exists: $destination" >&2
  exit 2
fi
curl --fail --location --retry 3 --continue-at - "$url" --output "$destination.part"
actual="$(shasum -a 256 "$destination.part" | cut -d ' ' -f 1)"
if [[ "$actual" != "$expected" ]]; then
  echo "Checkpoint SHA256 mismatch" >&2
  exit 2
fi
mv "$destination.part" "$destination"
