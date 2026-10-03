#!/usr/bin/env bash
# Linux CUDA handoff: rebuild native binaries and independently validate model paths.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-help}" in
  build)
    export PATH="/usr/local/cuda/bin:$PATH"
    if ! command -v nvcc >/dev/null && [[ -z "${CUDARC_CUDA_VERSION:-}" ]]; then
      echo 'Put nvcc in PATH, or set CUDARC_CUDA_VERSION to match the installed toolkit (12.2 = 12020).' >&2
      exit 2
    fi
    cargo build --release --locked --features cuda,convert --bin ancha --examples
    target/release/ancha doctor --backend cuda --device 0
    ;;
  reference)
    command -v uv >/dev/null || { echo 'Install uv to recreate the Python 3.11 reference environment.' >&2; exit 2; }
    py=NO_TRACK/.venv-parity/bin/python
    [[ -x "$py" ]] || uv venv --python 3.11 NO_TRACK/.venv-parity
    uv pip install --python "$py" --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.2.2+cpu' 'torchaudio==2.2.2+cpu'
    uv pip install --python "$py" -r scripts/requirements-mdx-parity.txt
    ;;
  test)
    cargo fmt --all --check
    cargo test --workspace --locked --features convert
    # CUDA service-thread failure state is process-wide; run the intentional OOM test separately.
    cargo test --release --locked --features cuda,convert \
      --test cuda_contracts --test cuda_mdx23c_contracts -- --test-threads=1
    cargo test --release --locked --features cuda,convert \
      --test cuda_device_failure -- --test-threads=1
    ;;
  parity)
    py="${ANCHA_PYTHON:-NO_TRACK/.venv-parity/bin/python}"
    mkdir -p NO_TRACK/runs
    out=$(mktemp -d NO_TRACK/runs/cuda-acceptance.XXXXXX)
    ANCHA_BACKENDS=cuda ANCHA_PARITY_DIR="$out/original-eight" bash scripts/parity-matrix.sh
    "$py" - "$out/original-eight/summary.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if (r['passed'], r['total']) != (8, 8):
    raise SystemExit(f"Original-model parity failed: {r['passed']}/{r['total']}")
PY
    ANCHA_BACKENDS=cuda ANCHA_PARITY_DIR="$out/derur" bash scripts/parity-derur.sh
    "$py" - "$out/derur" <<'PY'
import json, pathlib, sys
reports = sorted(pathlib.Path(sys.argv[1]).glob('*-parity.json'))
if len(reports) != 2 or any(json.load(open(p))['status'] != 'passed' for p in reports):
    raise SystemExit('Mel Karaoke / MDX23C parity failed or incomplete')
print('CUDA reference parity passed: 8 original models + 2 Derur models')
PY
    echo "Reports: $out"
    ;;
  smoke)
    target/release/examples/separate_all --backend cuda --start 30 --duration 30
    ;;
  help|--help|-h)
    echo 'Usage: bash scripts/cuda-dev.sh build|reference|test|parity|smoke'
    echo 'Run build first; reference is needed only for parity. smoke uses native 30-second contexts.'
    ;;
  *) echo "Unknown command: $1" >&2; exit 2 ;;
esac
