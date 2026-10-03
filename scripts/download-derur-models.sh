#!/usr/bin/env bash
# Selected public checkpoint/config files; revisions and SHA256 never follow main at runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
revision=f3bb9a312519f4404dde996ef1054ec30353c46f
root=NO_TRACK/models/derur-download
mkdir -p "$root"
fetch() {
  local name="$1" digest="$2" target="$root/$1"
  mkdir -p "$(dirname "$target")"
  if [[ ! -f "$target" ]]; then
    if command -v hf >/dev/null 2>&1; then
      HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download Derur/UVR-models "$name" \
        --revision "$revision" --local-dir "$root"
    else
      curl -fL --retry 3 -C - \
        "https://huggingface.co/Derur/UVR-models/resolve/$revision/$name?download=true" -o "$target.part"
      [[ "$(shasum -a 256 "$target.part" | awk '{print $1}')" == "$digest" ]] || { echo "Checksum mismatch: $name" >&2; exit 2; }
      mv "$target.part" "$target"
    fi
  fi
  [[ "$(shasum -a 256 "$target" | awk '{print $1}')" == "$digest" ]] || { echo "Checksum mismatch: $name" >&2; exit 2; }
}
mode="${1:-all}"
case "$mode" in all|mel-karaoke|mdx23c) ;; *) echo 'Usage: download-derur-models.sh [all|mel-karaoke|mdx23c]' >&2; exit 2 ;; esac
if [[ "$mode" != mdx23c ]]; then
  dir=mel_band_roformer_karaoke_aufr33_viperx_sdr_10
  fetch "$dir/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt" 1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813
  fetch "$dir/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml" b35077d94861f068097cce1a5e54633c055e7dcc2613eade4e4dc7c7c9c3f48b
fi
if [[ "$mode" != mel-karaoke ]]; then
  dir=MDX23C-8KFFT-InstVoc_HQ_2
  fetch "$dir/MDX23C-8KFFT-InstVoc_HQ_2.ckpt" 7d960d8e40a458120412c1bd807e013d2dbca7b959cc9da2bbcb0eb203d1daea
  fetch "$dir/model_2_stem_full_band_8k.yaml" 451765e869b78dcb9ca9188a74da31f581b7254ff0e8b532aa76b974148de947
fi
