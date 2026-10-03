#!/usr/bin/env bash
# Fixed model revisions and publishing SHA256s. Files land in models/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
fetch() {
  local path="$1" digest="$2" url="$3"
  if [[ -e "$path" ]]; then
    [[ "$(shasum -a 256 "$path" | awk '{print $1}')" == "$digest" ]] || { echo "Checksum mismatch: $path" >&2; exit 2; }
    return
  fi
  curl -fL --retry 3 -C - "$url" -o "$path.part"
  [[ "$(shasum -a 256 "$path.part" | awk '{print $1}')" == "$digest" ]] || { echo "Checksum mismatch: $path.part" >&2; exit 2; }
  mv "$path.part" "$path"
}
case "${1:-all}" in
  hyperace|all)
    rev=5b1f8283125d5e4a3614d0e3635a636e09c84059
    fetch models/hyperace-v2-voc.ckpt 54cf516f621f2f460bf660ed137e244b8931bf7a2ce85ddceecff816dbc4d668 \
      "https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/$rev/v2_voc/bs_roformer_voc_hyperacev2.ckpt?download=true"
    fetch models/hyperace-v2-inst.ckpt 4d61178ef966d2b4e9ad456ffbbc6fd5b2828df07a7af32931142c1a5ff1fe6f \
      "https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/$rev/v2_inst/bs_roformer_inst_hyperacev2.ckpt?download=true"
    ;;
  mdx) ;;
  *) echo 'Usage: download-models.sh [all|hyperace|mdx]' >&2; exit 2 ;;
esac
if [[ "${1:-all}" != hyperace ]]; then
  base=https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models
  fetch models/UVR_MDXNET_9482.onnx f4f365207c56deb115bceedff3ad8fe98a751c745f9e370cecec6226b8b47184 "$base/UVR_MDXNET_9482.onnx"
  fetch models/UVR_MDXNET_KARA.onnx e3167c87333a48548413e972a286bf40bf5694001d2853861eb1435953f02d63 "$base/UVR_MDXNET_KARA.onnx"
  fetch models/UVR_MDXNET_KARA_2.onnx bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4 "$base/UVR_MDXNET_KARA_2.onnx"
  fetch models/UVR-MDX-NET-Inst_HQ_2.onnx 197f8ab296df850f961e68c595f6649acb7d9e621b5600b460f3458967299112 "$base/UVR-MDX-NET-Inst_HQ_2.onnx"
fi
