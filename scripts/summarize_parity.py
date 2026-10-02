#!/usr/bin/env python3
"""Summarize a scripts/parity-matrix.sh directory into a JSON report without local paths."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--binary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entries = []
    for report in sorted(args.root.glob("*-parity.json")):
        name = report.name.removesuffix("-parity.json")
        run = json.loads((args.root / name / "run.json").read_text())
        ref = json.loads(report.read_text())
        rows = ref.get("comparisons") or [{
            "stem": run["effective_config"]["predicted"], "max_abs": ref["max_abs"],
            "waveform_snr_db": ref["waveform_snr_db"], "status": ref["status"]}]
        entries.append({
            "run": name,
            "status": ref["status"],
            "rust_backend": run["backend"],
            "rust_device": run["device"],
            "model_id": run["model_id"],
            "rust_profile": run.get("profile"),
            "linear_layout": run.get("linear_layout"),
            "conv_strategy": run.get("conv_strategy"),
            "reference": {k: ref[k] for k in ("scope", "source_revision", "torch", "ort", "numpy",
                                              "threads", "ort_threads") if k in ref},
            "comparisons": [{k: c[k] for k in ("stem", "max_abs", "waveform_snr_db", "status")}
                            for c in rows],
        })
    report = {
        "scope": "Rust run vs pinned PyTorch (RoFormer, one 132300-sample chunk) and UVR + ONNX "
                 "Runtime (MDX); threshold max_abs < 1e-3 and waveform SNR > 50 dB; implementation "
                 "consistency, not SDR",
        "rust_binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()
        if args.binary else None,
        "passed": sum(e["status"] == "passed" for e in entries),
        "total": len(entries),
        "entries": entries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['passed']}/{report['total']} passed")


if __name__ == "__main__":
    main()
