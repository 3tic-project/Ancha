#!/usr/bin/env bash
set -euo pipefail
# Run serialized, on an otherwise idle machine. Input should be a 3-second F32 WAV.
model="${1:-NO_TRACK/models/leap-xe-voc}"
input="${2:-NO_TRACK/runs/clip-3s.wav}"
root="${3:-NO_TRACK/runs/layout-benchmark}"
backend="${4:-wgpu}"
mkdir "$root"
for mode in baseline optimized; do
  args=(separate "$input" --model "$model" --output "$root/$mode"
    --backend "$backend" --chunk-samples 132300 --overlap 1)
  if [[ "$mode" == optimized ]]; then args+=(--flatten-linear); fi
  target/release/ancha "${args[@]}" > "$root/$mode.json"
done
NO_TRACK/.venv-parity/bin/python scripts/compare_runs.py "$root/baseline" "$root/optimized" \
  --output "$root/comparison.json"
