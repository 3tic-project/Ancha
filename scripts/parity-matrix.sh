#!/usr/bin/env bash
# Numerical regression of one binary against the pinned PyTorch / UVR references.
# Usage: ANCHA_BACKENDS="cuda cpu" bash scripts/parity-matrix.sh
# Needs NO_TRACK/.venv-parity, NO_TRACK/reference, checkpoints and clip-3s.wav.
set -uo pipefail
cd "$(dirname "$0")/.."
BIN="${ANCHA_BINARY:-target/release/ancha}"
BACKENDS="${ANCHA_BACKENDS:-cuda}"
PY="${ANCHA_PYTHON:-NO_TRACK/.venv-parity/bin/python}"
C3="${ANCHA_CLIP_3S:-NO_TRACK/runs/clip-3s.wav}"
if [[ ! -f "$C3" && -z "${ANCHA_CLIP_3S+x}" ]]; then C3=outputs/clip-3s.wav; fi
OUT="${ANCHA_PARITY_DIR:-NO_TRACK/runs/parity-matrix}"
M="${ANCHA_MODELS:-models}"
if [[ ! -d "$M" && -z "${ANCHA_MODELS+x}" ]]; then M=NO_TRACK/models; fi
H=NO_TRACK/reference/hyperace/bs_roformer.py
mkdir -p "$OUT"
summary() { # report label
  "$PY" - "$1" "$2" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
rows = r.get("comparisons") or [dict(stem="predicted", **r)]
print(f"{sys.argv[2]:30s} {r['status']:7s}", " ".join(
    f"{c['stem']}:max_abs={c['max_abs']:.3e},snr={c['waveform_snr_db']:.2f}" for c in rows))
EOF
}
roformer() { # name package checkpoint backend [verify args]
  local name=$1 pkg=$2 ckpt=$3 backend=$4; shift 4
  local run="$OUT/$name-$backend"
  rm -rf "$run"
  "$BIN" separate "$C3" --model "$pkg" --backend "$backend" --chunk-samples 132300 --overlap 1 \
    --output "$run" > "$run.json" 2> "$run.log" || { echo "$name $backend RUST FAILED"; return; }
  "$PY" scripts/verify_parity.py --reference NO_TRACK/reference --checkpoint "$ckpt" \
    --package "$pkg" --input "$C3" --rust-output "$run" --report "$run-parity.json" "$@" \
    > "$run-parity.log" 2>&1 || true
  summary "$run-parity.json" "$name-$backend"
}
mdx() { # model backend
  local name run
  name=$(basename "$1" .onnx)
  run="$OUT/$name-$2"
  rm -rf "$run"
  "$BIN" separate "$C3" --model "$1" --backend "$2" --output "$run" > "$run.json" 2> "$run.log" \
    || { echo "$name $2 RUST FAILED"; return; }
  "$PY" scripts/verify_mdx.py --uvr-source NO_TRACK/reference/uvr --input "$C3" --model "$1" \
    --rust-output "$run" --report "$run-parity.json" > "$run-parity.log" 2>&1 || true
  summary "$run-parity.json" "$name-$2"
}
for b in $BACKENDS; do
  roformer leap "$M/leap-xe-voc" "$M/bs_leap_xe_voc.ckpt" "$b"
  roformer deux "$M/deux" "$M/becruily_deux.ckpt" "$b"
  roformer hace-voc "$M/hyperace-v2-voc" "$M/hyperace-v2-voc.ckpt" "$b" --hyperace-source "$H"
  roformer hace-inst "$M/hyperace-v2-inst" "$M/hyperace-v2-inst.ckpt" "$b" --hyperace-source "$H"
  for m in UVR_MDXNET_9482 UVR_MDXNET_KARA UVR_MDXNET_KARA_2 UVR-MDX-NET-Inst_HQ_2; do
    mdx "$M/$m.onnx" "$b"
  done
done
"$PY" scripts/summarize_parity.py "$OUT" --binary "$BIN" --output "$OUT/summary.json"
