#!/usr/bin/env python3
"""Serial, warmed, frozen-binary ablation; compare identical PCM and model context."""
import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--input", type=Path, default=Path("NO_TRACK/runs/clip-3s.wav"))
    p.add_argument("--backend", choices=["cpu", "ndarray", "wgpu", "cuda"], required=True)
    p.add_argument("--binary", type=Path, default=Path("target/release/ancha"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--chunk-samples", type=int)
    p.add_argument("--overlap", type=int)
    p.add_argument("--ablation", choices=["math", "conv", "linear"], required=True,
                   help="math: CUDA TDF/tail pruning; conv: backend vs GEMM; linear: RoFormer batched vs flattened")
    a = p.parse_args()
    assert a.iterations > 0 and not a.output.exists()
    a.output.mkdir(parents=True)
    binary = a.output / "ancha-benchmark"
    shutil.copy2(a.binary, binary)
    base = [str(binary.resolve()), "separate", str(a.input), "--model", str(a.model), "--backend", a.backend]
    if a.chunk_samples is not None:
        base += ["--chunk-samples", str(a.chunk_samples)]
    if a.overlap is not None:
        base += ["--overlap", str(a.overlap)]
    flags = {"math": {"baseline": ["--mdx-no-optimize"], "optimized": []},
             "conv": {"baseline": ["--conv-strategy", "backend"], "optimized": ["--conv-strategy", "gemm"]},
             "linear": {"baseline": ["--linear-layout", "batched"], "optimized": ["--linear-layout", "flattened"]}}[a.ablation]

    def run(kind, suffix):
        name = f"{kind}-{suffix}"
        out = a.output / name
        with (a.output / f"{name}.log").open("w") as log:
            subprocess.run(base + flags[kind] + ["--output", str(out)], stdout=log, stderr=log, check=True)
        report = json.loads((out / "run.json").read_text())
        print(f"{name}: {report['timings']['total_seconds']:.3f} s", flush=True)
        return report

    warmups = {kind: run(kind, "warmup") for kind in flags}
    reports = {kind: [] for kind in flags}
    comparisons = []
    for i in range(1, a.iterations + 1):
        order = ["baseline", "optimized"] if i % 2 else ["optimized", "baseline"]
        pair = {kind: run(kind, i) for kind in order}
        for key in ["weights_sha256", "input_pcm_sha256", "backend", "device", "effective_config",
                    "profile", "precision", "samples_per_channel", "build_features", "cpu_thread_environment"]:
            assert pair["baseline"][key] == pair["optimized"][key], key
        for kind in flags:
            reports[kind].append(pair[kind])
        for stem in pair["baseline"]["stems"]:
            waves = [sf.read(a.output / f"{kind}-{i}" / f"{stem['name']}.wav", dtype="float32", always_2d=True)
                     for kind in flags]
            (x, sr), (y, yr) = waves
            assert sr == yr and x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            error = x.astype(np.float64) - y.astype(np.float64)
            maximum = float(np.max(np.abs(error)))
            snr = float(10 * np.log10(max(np.mean(x.astype(np.float64) ** 2), 1e-30) / max(np.mean(error ** 2), 1e-30)))
            assert maximum < 1e-3 and snr > 50, (stem["name"], maximum, snr)
            comparisons.append(dict(iteration=i, stem=stem["name"], max_abs=maximum, waveform_snr_db=snr))
    medians = {kind: {k: statistics.median(r["timings"][k] for r in rows)
                      for k in ["total_seconds", "model_seconds"]} for kind, rows in reports.items()}
    result = dict(scope="serial same-binary/PCM/context ablation; one warmup per variant excluded; alternating order; consistency is not SDR",
                  ablation=a.ablation, backend=a.backend, model_id=pair["baseline"]["model_id"],
                  binary_sha256=hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
                  command=base, flags=flags, iterations=a.iterations, warmups=warmups,
                  reports=reports, medians=medians, waveform_comparisons=comparisons,
                  total_speedup=medians["baseline"]["total_seconds"] / medians["optimized"]["total_seconds"],
                  model_speedup=medians["baseline"]["model_seconds"] / medians["optimized"]["model_seconds"], status="passed")
    (a.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["medians", "total_speedup", "model_speedup", "status"]}, indent=2))


if __name__ == "__main__":
    main()
