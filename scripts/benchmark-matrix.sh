#!/usr/bin/env bash
# Serial benchmark matrix with identical PCM, weights and audio context.
# ANCHA_OLD / ANCHA_NEW are two release binaries; both run their own defaults.
# Run on an idle machine. Phase "warm" fills the WGPU autotune cache first.
set -uo pipefail
cd "$(dirname "$0")/.."
OLD="${ANCHA_OLD:?set ANCHA_OLD to the baseline binary}"
NEW="${ANCHA_NEW:-target/release/ancha}"
ROOT="${ANCHA_BENCH_DIR:-NO_TRACK/runs/benchmark-matrix}"
PYTHON="${ANCHA_PYTHON:-python3}"
C3="${ANCHA_CLIP_3S:-NO_TRACK/runs/clip-3s.wav}"
C30="${ANCHA_CLIP_30S:-NO_TRACK/runs/clip-30s.wav}"
M=NO_TRACK/models
L=$M/leap-xe-voc D=$M/deux H=$M/hyperace-v2-voc
K2=$M/UVR_MDXNET_KARA_2.onnx HQ=$M/UVR-MDX-NET-Inst_HQ_2.onnx
K1=$M/UVR_MDXNET_KARA.onnx N9=$M/UVR_MDXNET_9482.onnx
SHORT=(--chunk-samples 132300 --overlap 1)
mkdir -p "$ROOT"
cp "$OLD" "$ROOT/ancha-old" && cp "$NEW" "$ROOT/ancha-new"

run() { # kind label input model backend [args...]
  local kind=$1 label=$2 input=$3 model=$4 backend=$5; shift 5
  local n=1 out
  while [[ -e "$ROOT/$label/$kind-$n.json" ]]; do n=$((n + 1)); done
  mkdir -p "$ROOT/$label"
  out="$ROOT/$label/$kind-$n"
  if ! "$ROOT/ancha-$kind" separate "$input" --model "$model" --backend "$backend" \
      --output "$out" "$@" > "$out.json" 2> "$out.log"; then
    echo "$label $kind failed: $(tail -1 "$out.log")"
    rm -f "$out.json"
    return
  fi
  echo "$label $kind-$n done"
}
pair() { # label input model backend [args...]; alternates order to spread drift
  run old "$@"; run new "$@"; run new "$@"; run old "$@"
}

phase="${1:-all}"
if [[ $phase == warm || $phase == all ]]; then
  for model in "$L" "$D" "$H"; do run new warm "$C3" "$model" wgpu "${SHORT[@]}"; done
  for model in "$L" "$H" "$K2" "$HQ" "$K1" "$N9"; do run new warm "$C3" "$model" wgpu; done
fi
if [[ $phase == gpu || $phase == all ]]; then
  pair gpu-leap-short30 "$C30" "$L" wgpu "${SHORT[@]}"
  run old gpu-leap-native3 "$C3" "$L" wgpu; run new gpu-leap-native3 "$C3" "$L" wgpu
  pair gpu-deux-short30 "$C30" "$D" wgpu "${SHORT[@]}"
  pair gpu-hace-short30 "$C30" "$H" wgpu "${SHORT[@]}"
  run old gpu-hace-native3 "$C3" "$H" wgpu; run new gpu-hace-native3 "$C3" "$H" wgpu
  pair gpu-kara2-30 "$C30" "$K2" wgpu
  pair gpu-hq2-30 "$C30" "$HQ" wgpu
  for kind in old new; do
    run $kind gpu-kara-30 "$C30" "$K1" wgpu; run $kind gpu-9482-30 "$C30" "$N9" wgpu
  done
fi
if [[ $phase == cpu || $phase == all ]]; then
  pair cpu-kara2-3 "$C3" "$K2" cpu
  for kind in old new; do
    run $kind cpu-kara2-30 "$C30" "$K2" cpu; run $kind cpu-hq2-3 "$C3" "$HQ" cpu
  done
  pair cpu-leap-short3 "$C3" "$L" cpu "${SHORT[@]}"
  for kind in old new; do run $kind cpu-deux-short3 "$C3" "$D" cpu "${SHORT[@]}"; done
  for kind in old new; do run $kind cpu-hace-short3 "$C3" "$H" cpu "${SHORT[@]}"; done
fi
"$PYTHON" scripts/summarize_benchmarks.py "$ROOT" --output "$ROOT/summary.json"
