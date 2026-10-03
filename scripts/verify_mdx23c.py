#!/usr/bin/env python3
"""Development-only waveform parity using pinned UVR MDXC demix and TFC/TDF v3.

The Rust runtime never invokes Python. Reference caching keys PCM, checkpoint, source,
effective context and torch version; cached waveforms are checked against stored hashes.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch


def output_stem(name):
    if name in ("vocals", "lead_vocals", "all_vocals"):
        return "vocals"
    if name in ("instrumental", "instrument", "karaoke_mix"):
        return "instrument"
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uvr-source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--rust-output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--reference-cache", type=Path)
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    run = json.loads((a.rust_output / "run.json").read_text())
    c = run["effective_config"]
    source, sr = sf.read(a.input, dtype="float32", always_2d=True)
    assert sr == c["sample_rate"] and source.shape == (run["samples_per_channel"], 2)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    code = (a.uvr_source / "tfc_tdf_v3.py").read_bytes()
    demix_code = (a.uvr_source / "separate.py").read_bytes()
    assert hashlib.sha256(code).hexdigest() == "12aa778119eb96dd8549df6559122bc91e92352ce8f12f51259ad72a7b91111b", "pinned TFC/TDF source mismatch"
    assert hashlib.sha256(demix_code).hexdigest() == "d3d1fd3288491895415a8487bb56d0cf75914e0080d1f1cf37aab145c5c576e8", "pinned UVR demix source mismatch"
    assert hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest() == "7d960d8e40a458120412c1bd807e013d2dbca7b959cc9da2bbcb0eb203d1daea", "MDX23C checkpoint mismatch"
    key_data = dict(config=c, checkpoint_sha256=hashlib.file_digest(a.checkpoint.open("rb"), "sha256").hexdigest(),
                    input_pcm_sha256=hashlib.sha256(source.T.copy().tobytes()).hexdigest(),
                    tfc_source_sha256=hashlib.sha256(code).hexdigest(),
                    demix_source_sha256=hashlib.sha256(demix_code).hexdigest(),
                    torch=torch.__version__, threads=a.threads)
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    cache = a.reference_cache / key if a.reference_cache else None
    cached = False
    expected = {}
    elapsed = None
    if cache and (cache / "reference.json").exists():
        meta = json.loads((cache / "reference.json").read_text())
        assert meta["key_data"] == key_data
        for stem in c["stems"]:
            path = cache / f"{stem}.wav"
            assert hashlib.file_digest(path.open("rb"), "sha256").hexdigest() == meta["sha256"][stem]
            expected[stem], rate = sf.read(path, dtype="float32", always_2d=True)
            assert rate == sr
        elapsed = meta["reference_seconds"]
        cached = True
    else:
        spec = importlib.util.spec_from_file_location("pinned_tfc", a.uvr_source / "tfc_tdf_v3.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tree = ast.parse(demix_code)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SeperateMDXC")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "demix")
        wrapper = ast.ClassDef(name="UvrReference", bases=[], keywords=[], body=[method], decorator_list=[])
        class SafeTorch:
            def __getattr__(self, name):
                return getattr(torch, name)
            def load(self, *args, **kwargs):
                return torch.load(*args, weights_only=True, **kwargs)
        env = dict(torch=SafeTorch(), TFC_TDF_net=module.TFC_TDF_net, cpu="cpu")
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                     str(a.uvr_source / "separate.py"), "exec"), env)
        ref = env["UvrReference"]()
        ref.mdx_c_configs = SimpleNamespace(
            audio=SimpleNamespace(n_fft=c["n_fft"], hop_length=c["hop"], dim_f=c["bins"], num_channels=2),
            model=SimpleNamespace(norm="InstanceNorm", act="gelu", num_subbands=c["subbands"],
                                  num_scales=c["scales"], scale=c["scale"],
                                  num_blocks_per_scale=c["blocks_per_scale"], num_channels=c["channels"],
                                  growth=c["growth"], bottleneck_factor=c["bottleneck_factor"]),
            training=SimpleNamespace(target_instrument=None, instruments=["Vocals", "Instrumental"]),
            inference=SimpleNamespace(dim_t=c["frames"]))
        ref.is_pitch_change = False
        ref.is_denoise_model = False
        ref.model_path = str(a.checkpoint)
        ref.device = "cpu"
        ref.is_mdx_c_seg_def = True
        ref.mdx_batch_size = 1
        ref.overlap_mdx23 = run["overlap"]
        ref.running_inference_progress_bar = lambda *args, **kwargs: None
        start = time.perf_counter()
        with torch.inference_mode():
            result = ref.demix(source.T.copy())
        elapsed = time.perf_counter() - start
        expected = {stem: result[label].T for stem, label in zip(c["stems"], ["Vocals", "Instrumental"])}
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for stem, wave in expected.items():
                path = cache / f"{stem}.wav"
                sf.write(path, wave, sr, subtype="FLOAT")
                hashes[stem] = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            (cache / "reference.json").write_text(json.dumps(dict(key_data=key_data, sha256=hashes,
                                                                reference_seconds=elapsed), indent=2) + "\n")
    comparisons = []
    for stem in c["stems"]:
        actual, rate = sf.read(a.rust_output / f"{output_stem(stem)}.wav", dtype="float32", always_2d=True)
        wave = expected[stem]
        assert rate == sr and actual.shape == wave.shape and np.isfinite(actual).all() and np.isfinite(wave).all()
        error = actual.astype(np.float64) - wave.astype(np.float64)
        snr = 10 * np.log10(max(np.mean(wave.astype(np.float64) ** 2), 1e-30) / max(np.mean(error ** 2), 1e-30))
        max_abs = float(np.max(np.abs(error)))
        comparisons.append(dict(stem=stem, max_abs=max_abs, waveform_snr_db=float(snr),
                                status="passed" if max_abs < 1e-3 and snr > 50 else "failed"))
    report = dict(scope="pinned UVR MDXC demix + TFC/TDF v3, stereo two native heads; no denoise, pitch shift or postprocessing; waveform consistency, not SDR",
                  source_revision="UVR/5517e0cf0d1acd16a1618eeedec596957523f9e1",
                  reference_identity=key_data, reference_cache_key=key, reference_cache_used=cached,
                  backend=run["backend"], profile=run["profile"], samples=len(source),
                  torch=torch.__version__, numpy=np.__version__, reference_seconds=elapsed,
                  comparisons=comparisons, status="passed" if all(c["status"] == "passed" for c in comparisons) else "failed")
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
