#!/usr/bin/env python3
"""Compare two completed runs with identical model, PCM and audio context."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("baseline", type=Path)
p.add_argument("optimized", type=Path)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
baseline = json.loads((args.baseline / "run.json").read_text())
optimized = json.loads((args.optimized / "run.json").read_text())
for key in ("weights_sha256", "input_pcm_sha256", "samples_per_channel", "sample_rate",
            "effective_config", "backend", "precision"):
    assert baseline[key] == optimized[key], f"comparison changes {key}"
comparisons = []
for entry in baseline["stems"]:
    stem = entry["name"]
    a, sr_a = sf.read(args.baseline / f"{stem}.wav", dtype="float32", always_2d=True)
    b, sr_b = sf.read(args.optimized / f"{stem}.wav", dtype="float32", always_2d=True)
    assert sr_a == sr_b and a.shape == b.shape
    error = a.astype(np.float64) - b.astype(np.float64)
    max_abs = float(np.max(np.abs(error)))
    assert max_abs < 1e-3 and np.isfinite(b).all(), f"layout regression: {stem} {max_abs}"
    comparisons.append(dict(stem=stem, max_abs=max_abs,
                            mean_abs=float(np.mean(np.abs(error))), status="passed"))
base_seconds = baseline["timings"]["total_seconds"]
optimized_seconds = optimized["timings"]["total_seconds"]
report = dict(scope="same-backend single-run execution-plan ablation; no context change",
              model_id=baseline["model_id"], backend=baseline["backend"],
              audio_seconds=baseline["audio_seconds"], weights_sha256=baseline["weights_sha256"],
              input_pcm_sha256=baseline["input_pcm_sha256"],
              baseline_linear_layout=baseline.get("linear_layout", "batched"),
              optimized_linear_layout=optimized.get("linear_layout", "flattened"),
              baseline_query_tile=baseline["query_tile"], baseline_group_tile=baseline["group_tile"],
              optimized_query_tile=optimized["query_tile"], optimized_group_tile=optimized["group_tile"],
              baseline_total_seconds=base_seconds, optimized_total_seconds=optimized_seconds,
              baseline_model_seconds=baseline["timings"]["model_seconds"],
              optimized_model_seconds=optimized["timings"]["model_seconds"],
              total_speedup=base_seconds / optimized_seconds,
              model_speedup=baseline["timings"]["model_seconds"] / optimized["timings"]["model_seconds"],
              comparisons=comparisons, status="passed")
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
