#!/usr/bin/env bash
# Serial backend matrix for one binary: identical PCM, weights and audio context per row.
# Usage: ANCHA_BACKENDS="cuda wgpu cpu" bash scripts/benchmark-backends.sh [warm|run|all]
# Run on an idle machine; "warm" fills the autotune / PTX caches first.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${ANCHA_BINARY:-target/release/ancha}"
BACKENDS="${ANCHA_BACKENDS:-cuda wgpu cpu}"
REPEAT="${ANCHA_REPEAT:-2}"
ROOT="${ANCHA_BENCH_DIR:-NO_TRACK/runs/benchmark-backends}"
PYTHON="${ANCHA_PYTHON:-python3}"
C3="${ANCHA_CLIP_3S:-NO_TRACK/runs/clip-3s.wav}"
C30="${ANCHA_CLIP_30S:-NO_TRACK/runs/clip-30s.wav}"
M=NO_TRACK/models
L=$M/leap-xe-voc D=$M/deux H=$M/hyperace-v2-voc
K2=$M/UVR_MDXNET_KARA_2.onnx HQ=$M/UVR-MDX-NET-Inst_HQ_2.onnx
K1=$M/UVR_MDXNET_KARA.onnx N9=$M/UVR_MDXNET_9482.onnx
SHORT=(--chunk-samples 132300 --overlap 1)
mkdir -p "$ROOT"
cp "$BIN" "$ROOT/ancha"

run() { # label backend input model [args...]
  local label=$1 backend=$2 input=$3 model=$4; shift 4
  local n=1 out
  while [[ -e "$ROOT/$label/$backend-$n.json" ]]; do n=$((n + 1)); done
  mkdir -p "$ROOT/$label"
  out="$ROOT/$label/$backend-$n"
  if ! "$ROOT/ancha" separate "$input" --model "$model" --backend "$backend" \
      --output "$out" "$@" > "$out.json" 2> "$out.log"; then
    echo "$label $backend failed: $(grep -m1 'ancha:' "$out.log")"
    rm -f "$out.json"
    return
  fi
  echo "$label $backend-$n done"
}
has() { [[ " $BACKENDS " == *" $1 "* ]]; }
rows() { # kind: warm runs once into warm-*, run uses the real labels
  local p=$1 b
  for b in $BACKENDS; do
    [[ $b == cpu ]] && continue
    run "${p}gpu-leap-short30" "$b" "$C30" "$L" "${SHORT[@]}"
    run "${p}gpu-deux-short30" "$b" "$C30" "$D" "${SHORT[@]}"
    run "${p}gpu-hace-short30" "$b" "$C30" "$H" "${SHORT[@]}"
    run "${p}gpu-leap-native3" "$b" "$C3" "$L"
    run "${p}gpu-hace-native3" "$b" "$C3" "$H"
    run "${p}gpu-kara2-30" "$b" "$C30" "$K2"
    run "${p}gpu-hq2-30" "$b" "$C30" "$HQ"
    run "${p}gpu-kara-30" "$b" "$C30" "$K1"
    run "${p}gpu-9482-30" "$b" "$C30" "$N9"
  done
  if has cpu && [[ $p == "" ]]; then
    run cpu-leap-short3 cpu "$C3" "$L" "${SHORT[@]}"
    run cpu-deux-short3 cpu "$C3" "$D" "${SHORT[@]}"
    run cpu-hace-short3 cpu "$C3" "$H" "${SHORT[@]}"
    run cpu-kara2-30 cpu "$C30" "$K2"
    run cpu-hq2-3 cpu "$C3" "$HQ"
  fi
}
phase="${1:-all}"
if [[ $phase == warm || $phase == all ]]; then rows warm-; fi
if [[ $phase == run || $phase == all ]]; then
  for ((i = 1; i <= REPEAT; i++)); do rows ""; done
fi
"$PYTHON" scripts/summarize_backends.py "$ROOT" --output "$ROOT/summary.json"
