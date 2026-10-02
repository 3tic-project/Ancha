#!/usr/bin/env python3
"""Independent, development-only PyTorch waveform parity against pinned MSST.

Use a one-chunk Rust run: --chunk-samples <WAV frames> --overlap 1.
The released Rust runtime does not import or execute this script.
"""
import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--rust-output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.reference.resolve()))
    manifest = json.loads((args.package / "manifest.json").read_text())
    c = manifest["config"]
    common = dict(dim=c["dim"], depth=c["depth"], stereo=True,
                  num_stems=len(c["stems"]), time_transformer_depth=1,
                  freq_transformer_depth=1, dim_head=c["head_dim"], heads=c["heads"],
                  attn_dropout=0, ff_dropout=0, flash_attn=False,
                  stft_n_fft=c["n_fft"], stft_hop_length=c["hop"],
                  stft_win_length=c["n_fft"], stft_normalized=False,
                  zero_dc=c["zero_dc"], mask_estimator_depth=c["mask_depth"],
                  mlp_expansion_factor=c["mask_expansion"])
    if c["family"] == "bs-roformer":
        cls = importlib.import_module("models.bs_roformer.bs_roformer").BSRoformer
        common["freqs_per_bands"] = tuple(map(len, c["bands"]))
    else:
        cls = importlib.import_module("models.bs_roformer.mel_band_roformer").MelBandRoformer
        common.update(num_bands=len(c["bands"]), sample_rate=c["sample_rate"],
                      match_input_audio_length=True)
    model = cls(**common).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    model.load_state_dict(state, strict=True)
    source, sr = sf.read(args.input, dtype="float32", always_2d=True)
    assert sr == c["sample_rate"] and source.shape[1] == 2
    with torch.inference_mode():
        start = time.perf_counter()
        result = model(torch.from_numpy(source.T.copy())[None]).numpy()[0]
        forward_seconds = time.perf_counter() - start
    if result.ndim == 2:
        result = result[None]
    comparisons = []
    for stem, expected in zip(c["stems"], result):
        actual, rate = sf.read(args.rust_output / f"{stem}.wav", dtype="float32", always_2d=True)
        assert rate == sr and actual.shape == expected.T.shape
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        error = actual.astype(np.float64) - expected.T.astype(np.float64)
        mse = np.mean(error ** 2)
        power = np.mean(expected.astype(np.float64) ** 2)
        snr = 10 * np.log10(max(power, 1e-30) / max(mse, 1e-30))
        max_abs = float(np.max(np.abs(error)))
        passed = max_abs < 1e-3 and snr > 50
        comparisons.append(dict(stem=stem, max_abs=max_abs, mean_abs=float(np.mean(np.abs(error))),
                                waveform_snr_db=float(snr), status="passed" if passed else "failed"))
        sf.write(args.report.parent / f"{manifest['model_id']}-torch-{stem}.wav", expected.T, sr, subtype="FLOAT")
    report = dict(scope="single-chunk FP32 waveform parity; not SDR or dataset quality evaluation",
                  source_revision=manifest["forward_revision"], torch=torch.__version__,
                  numpy=np.__version__, platform=platform.platform(), threads=args.threads,
                  samples=len(source), sample_rate=sr, model_id=manifest["model_id"],
                  pytorch_forward_seconds=forward_seconds, comparisons=comparisons,
                  status="passed" if all(x["status"] == "passed" for x in comparisons) else "failed")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
