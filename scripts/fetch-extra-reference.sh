#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p NO_TRACK/reference/hyperace NO_TRACK/reference/uvr
fetch() {
  local path="$1" url="$2"
  [[ ! -e "$path" ]] || return 0
  curl -fL --retry 3 "$url" -o "$path.part"
  mv "$path.part" "$path"
}
hyper=https://huggingface.co/pcunwa/BS-Roformer-HyperACE/resolve/5b1f8283125d5e4a3614d0e3635a636e09c84059/v2_voc
uvr=https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/5517e0cf0d1acd16a1618eeedec596957523f9e1
fetch NO_TRACK/reference/hyperace/bs_roformer.py "$hyper/bs_roformer.py"
fetch NO_TRACK/reference/hyperace/config.yaml "$hyper/config.yaml"
[[ "$(shasum -a 256 NO_TRACK/reference/hyperace/bs_roformer.py | awk '{print $1}')" == 48571e20d70ea8f245cffc6afbfa279f62042e7ba16fbaa3fe43dd2cbc25e1db ]] || { echo 'HyperACE source checksum mismatch' >&2; exit 2; }
fetch NO_TRACK/reference/uvr/separate.py "$uvr/separate.py"
fetch NO_TRACK/reference/uvr/tfc_tdf_v3.py "$uvr/lib_v5/tfc_tdf_v3.py"
