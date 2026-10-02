#!/usr/bin/env bash
# Run pairs serially with one binary; rotate order to reduce warmup/order bias.
set -euo pipefail
cd "$(dirname "$0")/.."
BINARY="${ANCHA_BINARY:-target/release/ancha}"
MODEL="${ANCHA_MODEL:-NO_TRACK/models/UVR_MDXNET_KARA_2.onnx}"
INPUT="${ANCHA_INPUT:-NO_TRACK/runs/clip-3s.wav}"
BACKEND="${ANCHA_BACKEND:-wgpu}"
ROOT="${ANCHA_BENCH_DIR:-NO_TRACK/runs/mdx-ablation}"
ITERATIONS="${ANCHA_ITERATIONS:-3}"
PYTHON="${ANCHA_PYTHON:-NO_TRACK/.venv-parity/bin/python}"
[[ ! -e "$ROOT" ]] || { echo "Benchmark output exists: $ROOT" >&2; exit 2; }
mkdir -p "$ROOT"
cp "$BINARY" "$ROOT/ancha-benchmark"
BINARY="$ROOT/ancha-benchmark"
for ((i=1; i<=ITERATIONS; i++)); do
  if ((i % 2)); then kinds=(baseline optimized); else kinds=(optimized baseline); fi
  for kind in "${kinds[@]}"; do
    args=(--backend "$BACKEND")
    if [[ "$kind" == baseline ]]; then args+=(--mdx-no-optimize); fi
    "$BINARY" separate "$INPUT" --model "$MODEL" "${args[@]}" \
      --output "$ROOT/$kind-$i" > "$ROOT/$kind-$i.log" 2>&1
  done
done
"$PYTHON" scripts/compare_mdx_runs.py "$ROOT" --iterations "$ITERATIONS" --binary "$BINARY" --output "$ROOT/comparison.json"
