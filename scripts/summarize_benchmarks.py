#!/usr/bin/env python3
"""Summarize scripts/benchmark-matrix.sh runs into a JSON report without local paths.

Each `<root>/<label>/{old,new}-N.json` is a run.json printed by `ancha separate`.
Medians are reported; WGPU first calls include kernel compilation / autotuning.
"""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path


def load(path):
    r = json.loads(path.read_text())
    t = r["timings"]
    calls = t.get("model_call_seconds") or []
    return {
        "model_id": r["model_id"],
        "backend": r["backend"],
        "profile": r.get("profile"),
        "audio_seconds": round(r["audio_seconds"], 3),
        "chunks": r["chunks"],
        "tiles": [r.get(k) for k in ("group_tile", "query_tile", "frequency_group_tile",
                                     "frequency_query_tile")] if "group_tile" in r else None,
        "host_threads": r.get("host_threads"),
        "conv_strategy": r.get("conv_strategy"),
        "input_pcm_sha256": r["input_pcm_sha256"],
        "weights_sha256": r["weights_sha256"],
        "total_seconds": t["total_seconds"],
        "model_seconds": t["model_seconds"],
        "first_call_seconds": calls[0] if calls else None,
        "steady_call_seconds": statistics.median(calls[1:]) if len(calls) > 1 else None,
        "rtf": r["rtf"],
    }


def median(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return statistics.median(values) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entries = []
    for label_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        runs = {kind: [load(f) for f in sorted(label_dir.glob(f"{kind}-*.json"))]
                for kind in ("old", "new")}
        if not runs["old"] or not runs["new"]:
            continue
        rows = runs["old"] + runs["new"]
        for key in ("input_pcm_sha256", "weights_sha256", "audio_seconds", "chunks"):
            assert len({row[key] for row in rows}) == 1, f"{label_dir.name}: {key} differs"
        entry = {"label": label_dir.name, "model_id": rows[0]["model_id"],
                 "audio_seconds": rows[0]["audio_seconds"], "chunks": rows[0]["chunks"]}
        for kind, name in (("old", "baseline"), ("new", "optimized")):
            first = runs[kind][0]
            entry[name] = {"runs": len(runs[kind]), "backend": first["backend"],
                           "profile": first["profile"], "tiles": first["tiles"],
                           "host_threads": first["host_threads"],
                           "conv_strategy": first["conv_strategy"]}
            for key in ("total_seconds", "model_seconds", "first_call_seconds",
                        "steady_call_seconds", "rtf"):
                entry[name][key] = median(runs[kind], key)
        entry["total_speedup"] = entry["baseline"]["total_seconds"] / entry["optimized"]["total_seconds"]
        old_steady = entry["baseline"]["steady_call_seconds"]
        new_steady = entry["optimized"]["steady_call_seconds"]
        if old_steady and new_steady:
            entry["steady_call_speedup"] = old_steady / new_steady
        entries.append(entry)
    binaries = {kind: hashlib.sha256((args.root / f"ancha-{kind}").read_bytes()).hexdigest()
                for kind in ("old", "new") if (args.root / f"ancha-{kind}").exists()}
    report = {
        "scope": "serial runs, identical PCM / weights / audio context, each binary at its defaults; "
                 "medians; warm WGPU autotune cache; not SDR or quality evaluation",
        "platform": f"{platform.system()}-{platform.machine()}",
        "binaries_sha256": binaries,
        "entries": entries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for e in entries:
        steady = f" steady x{e['steady_call_speedup']:.2f}" if "steady_call_speedup" in e else ""
        print(f"{e['label']:20s} {e['baseline']['total_seconds']:9.3f} -> "
              f"{e['optimized']['total_seconds']:9.3f} s  x{e['total_speedup']:.2f}{steady}")


if __name__ == "__main__":
    main()
