#!/usr/bin/env bash
# Native MDX23C overlap; single-chunk Mel reference. Uses only ignored local assets.
set -euo pipefail
cd "$(dirname "$0")/.."
binary="${ANCHA_BINARY:-target/release/ancha}"
python="${ANCHA_PYTHON:-NO_TRACK/.venv-parity/bin/python}"
backends="${ANCHA_BACKENDS:-cpu wgpu}"
input="${ANCHA_CLIP_3S:-NO_TRACK/runs/clip-3s.wav}"
if [[ ! -f "$input" && -z "${ANCHA_CLIP_3S+x}" ]]; then input=outputs/clip-3s.wav; fi
models="${ANCHA_MODELS:-models}"
if [[ ! -d "$models" && -z "${ANCHA_MODELS+x}" ]]; then models=NO_TRACK/models; fi
root="${ANCHA_PARITY_DIR:-NO_TRACK/runs/derur-parity}"
samples=$("$python" -c 'import soundfile,sys; print(soundfile.info(sys.argv[1]).frames)' "$input")
[[ ! -e "$root" ]] || { echo "Output exists: $root" >&2; exit 2; }
mkdir -p "$root"
cp "$binary" "$root/ancha-parity"
binary="$root/ancha-parity"
for backend in $backends; do
  run="$root/mel-karaoke-$backend"
  "$binary" separate "$input" --model "$models/mel-karaoke-aufr33-viperx" \
    --backend "$backend" --chunk-samples "$samples" --overlap 1 --output "$run" > "$run.log" 2>&1
  "$python" scripts/verify_parity.py --reference NO_TRACK/reference \
    --checkpoint "$models/derur-download/mel_band_roformer_karaoke_aufr33_viperx_sdr_10/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt" \
    --package "$models/mel-karaoke-aufr33-viperx" --input "$input" \
    --rust-output "$run" --report "$run-parity.json" > "$run-parity.log" 2>&1
  run="$root/mdx23c-$backend"
  "$binary" separate "$input" --model "$models/mdx23c-inst-voc-hq2" \
    --backend "$backend" --output "$run" > "$run.log" 2>&1
  "$python" scripts/verify_mdx23c.py --uvr-source NO_TRACK/reference/uvr \
    --checkpoint "$models/derur-download/MDX23C-8KFFT-InstVoc_HQ_2/MDX23C-8KFFT-InstVoc_HQ_2.ckpt" \
    --input "$input" --rust-output "$run" --report "$run-parity.json" \
    --reference-cache "$root/reference-cache" > "$run-parity.log" 2>&1
done
