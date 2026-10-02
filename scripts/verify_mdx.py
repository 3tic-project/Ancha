#!/usr/bin/env python3
"""Development-only parity: execute pinned UVR methods with CPU ORT, telemetry disabled.

Use --uvr-source NO_TRACK/reference/uvr with pinned separate.py/tfc_tdf_v3.py.
The Rust executable itself does not require ORT or Python.
"""
import argparse
import ast
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort

# Before creating any session, opt out of telemetry and use CPU explicitly.
ort.disable_telemetry_events()


def load_uvr(path):
    spec = importlib.util.spec_from_file_location("uvr_stft", path / "tfc_tdf_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((path / "separate.py").read_text())
    source_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SeperateMDX")
    names = {"initialize_model_settings", "demix", "run_model"}
    methods = [n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == 3
    cls = ast.ClassDef(name="UvrReference", bases=[], keywords=[], body=methods, decorator_list=[])
    module_tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    env = dict(np=np, torch=torch, STFT=module.STFT, DEFAULT="Default")
    exec(compile(module_tree, str(path / "separate.py"), "exec"), env)
    return env["UvrReference"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uvr-source", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--rust-output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    run = json.loads((a.rust_output / "run.json").read_text())
    c = run["effective_config"]
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = a.threads
    settings.inter_op_num_threads = 1
    settings.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(a.model), settings, providers=["CPUExecutionProvider"])
    reference = load_uvr(a.uvr_source)()
    reference.n_fft = c["n_fft"]
    reference.hop = c["hop"]
    reference.dim_f = c["bins"]
    reference.mdx_segment_size = c["frames"]
    reference.mdx_batch_size = 1
    reference.device = "cpu"
    reference.overlap_mdx = run["overlap_fraction"] if run["overlap_fraction"] is not None else "Default"
    reference.is_pitch_change = False
    reference.is_denoise_model = False
    reference.is_denoise = run["denoise"]
    reference.adjust = 1
    reference.compensate = c["compensate"]
    reference.running_inference_progress_bar = lambda *args, **kw: None
    calls = 0
    def model_run(spectrum):
        nonlocal calls
        calls += 1
        return session.run(None, {"input": spectrum.numpy()})[0]
    reference.model_run = model_run
    source, sr = sf.read(a.input, dtype="float32", always_2d=True)
    assert sr == c["sample_rate"] and len(source) == run["samples_per_channel"]
    start = time.perf_counter()
    with torch.inference_mode():
        expected = reference.demix(source.T.copy()).T
    elapsed = time.perf_counter() - start
    actual, rate = sf.read(a.rust_output / (c["predicted"] + ".wav"), dtype="float32", always_2d=True)
    assert rate == sr and actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all()
    error = actual.astype(np.float64) - expected.astype(np.float64)
    mse = np.mean(error ** 2)
    snr = 10 * np.log10(max(np.mean(expected.astype(np.float64) ** 2), 1e-30) / max(mse, 1e-30))
    max_abs = float(np.max(np.abs(error)))
    report = dict(scope="pinned UVR DSP + CPU ONNX Runtime waveform parity; not clean-stem SDR",
                  source_revision=c["source_revision"], weights_sha256=c["weights_sha256"], model_id=c["model_id"],
                  ort=ort.__version__, torch=torch.__version__, telemetry="disabled before session creation",
                  ort_threads=a.threads, backend=run["backend"], samples=len(source),
                  denoise=run["denoise"], overlap_fraction=run["overlap_fraction"],
                  original_uvr_forwards=calls, rust_forwards=run["model_forwards"],
                  reference_seconds=elapsed, max_abs=max_abs, waveform_snr_db=float(snr),
                  status="passed" if max_abs < 1e-3 and snr > 50 else "failed")
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    sf.write(a.report.with_suffix(".predicted.wav"), expected, sr, subtype="FLOAT")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
