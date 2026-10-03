#!/usr/bin/env python3
"""Generate original synthetic spectral golden data with the pinned UVR network."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("NO_TRACK/reference/uvr/tfc_tdf_v3.py"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.source.read_bytes()).hexdigest()
    assert digest == "12aa778119eb96dd8549df6559122bc91e92352ce8f12f51259ad72a7b91111b"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("pinned_tfc", a.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = dict(family="mdx23c", sample_rate=44100, n_fft=64, hop=8, bins=32, frames=8, overlap=4,
             subbands=2, channels=2, growth=2, scales=2, blocks_per_scale=1, bottleneck_factor=2,
             scale=[2, 2], stems=["vocals", "instrumental"])
    config = SimpleNamespace(
        audio=SimpleNamespace(n_fft=64, hop_length=8, dim_f=32, num_channels=2),
        model=SimpleNamespace(norm="InstanceNorm", act="gelu", num_subbands=2, num_scales=2,
                              scale=[2, 2], num_blocks_per_scale=1, num_channels=2, growth=2, bottleneck_factor=2),
        training=SimpleNamespace(target_instrument=None, instruments=["Vocals", "Instrumental"]))
    model = module.TFC_TDF_net(config, "cpu").eval()
    weights = []
    with torch.no_grad():
        for name, parameter in sorted(model.named_parameters()):
            seed = sum(name.encode()) % 97
            phase = (np.arange(parameter.numel(), dtype=np.float64) + 1) * 0.17 + seed * 0.11
            value = 1 + 0.03 * np.sin(phase) if parameter.ndim == 1 and name.endswith("weight") else 0.02 * np.sin(phase)
            parameter.copy_(torch.from_numpy(value.astype(np.float32).reshape(parameter.shape)))
            weights.append(dict(name=name, shape=list(parameter.shape)))
    # Keep the upstream spectral network and packing untouched; DSP has its own independent test.
    class SpectrumOnly:
        def __call__(self, x):
            return x
        def inverse(self, x):
            return x.flatten(1, 2)
    model.stft = SpectrumOnly()
    shape = [2, 4, 32, 8]
    data = (0.3 * np.sin((np.arange(np.prod(shape), dtype=np.float64) + 1) * 0.071)).astype(np.float32)
    with torch.inference_mode():
        result = model(torch.from_numpy(data.reshape(shape)))
    fixture = dict(scope="Original synthetic weights and input. Golden spectral forward from pinned UVR TFC_TDF_net; no real checkpoint or song.",
                   source_revision="UVR/5517e0cf0d1acd16a1618eeedec596957523f9e1", torch=torch.__version__,
                   source_sha256=digest, config=c, weights=weights, input_shape=shape,
                   output_shape=list(result.shape), expected=result.flatten().tolist())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
