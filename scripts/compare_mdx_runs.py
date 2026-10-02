#!/usr/bin/env python3
"""Compare paired MDX reports and WAVs; numerical consistency is not SDR."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert a.iterations > 0
    timings = {"baseline": [], "optimized": []}
    differences = []
    for i in range(1, a.iterations + 1):
        paths = [a.directory / f"{kind}-{i}" for kind in timings]
        reports = [json.loads((x / "run.json").read_text()) for x in paths]
        for key in ("weights_sha256", "input_pcm_sha256", "effective_config", "backend", "device",
                    "precision", "cpu_thread_environment", "batch_size", "denoise", "overlap_fraction", "samples_per_channel", "build_features"):
            assert reports[0][key] == reports[1][key], key
        assert not reports[0]["optimized"] and reports[1]["optimized"]
        for kind, report in zip(timings, reports):
            timings[kind].append(dict(total_seconds=report["timings"]["total_seconds"],
                                      model_seconds=report["timings"]["model_seconds"],
                                      forwards=report["model_forwards"], folded_bn=report["folded_bn"]))
        for stem in reports[0]["stems"]:
            x, sr = sf.read(paths[0] / (stem["name"] + ".wav"), dtype="float32", always_2d=True)
            y, yr = sf.read(paths[1] / (stem["name"] + ".wav"), dtype="float32", always_2d=True)
            assert sr == yr and x.shape == y.shape
            delta = x.astype(np.float64) - y.astype(np.float64)
            maximum = float(np.max(np.abs(delta)))
            snr = float(10 * np.log10(max(np.mean(x.astype(np.float64)**2), 1e-30) /
                                       max(np.mean(delta**2), 1e-30)))
            assert maximum < 1e-3 and snr > 50
            differences.append(dict(iteration=i, stem=stem["name"], max_abs=maximum, waveform_snr_db=snr))
    medians = {k: {metric: statistics.median(x[metric] for x in runs)
                   for metric in ("total_seconds", "model_seconds")} for k, runs in timings.items()}
    result = dict(scope="serial same-binary same-PCM/context MDX ablation; no denoise/overlap changes",
                  model_id=reports[0]["model_id"], weights_sha256=reports[0]["weights_sha256"],
                  pcm_sha256=reports[0]["input_pcm_sha256"], backend=reports[0]["backend"],
                  samples=reports[0]["samples_per_channel"], iterations=a.iterations,
                  denoise=reports[0]["denoise"], overlap_fraction=reports[0]["overlap_fraction"],
                  build_features=reports[0]["build_features"],
                  cpu_thread_environment=reports[0]["cpu_thread_environment"],
                  binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest() if a.binary else None,
                  runs=timings, medians=medians,
                  total_speedup=medians["baseline"]["total_seconds"]/medians["optimized"]["total_seconds"],
                  model_speedup=medians["baseline"]["model_seconds"]/medians["optimized"]["model_seconds"],
                  waveform_comparisons=differences, status="passed")
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
