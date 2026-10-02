#!/usr/bin/env python3
"""Summarize scripts/benchmark-backends.sh runs (`<root>/<label>/<backend>-N/run.json`) into medians.

Warm-up rows (`warm-*`) are skipped; GPU first calls of the measured rows use warm caches.
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
        "device": r["device"],
        "input_pcm_sha256": r["input_pcm_sha256"],
        "weights_sha256": r["weights_sha256"],
        "audio_seconds": round(r["audio_seconds"], 3),
        "chunks": r["chunks"],
        "settings": {k: r.get(k) for k in ("profile", "linear_layout", "conv_strategy",
                                           "host_threads", "group_tile", "query_tile",
                                           "frequency_group_tile", "frequency_query_tile")
                     if r.get(k) is not None},
        "total_seconds": t["total_seconds"],
        "model_load_seconds": t["model_load_seconds"],
        "model_seconds": t["model_seconds"],
        "first_call_seconds": calls[0] if calls else None,
        "steady_call_seconds": statistics.median(calls[1:]) if len(calls) > 1 else None,
        "rtf": r["rtf"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    entries = []
    for label_dir in sorted(d for d in args.root.iterdir() if d.is_dir()):
        if label_dir.name.startswith("warm-"):
            continue
        runs = {}
        # Completed runs only: outputs are published atomically with their run.json.
        for f in sorted(label_dir.glob("*-*/run.json")):
            runs.setdefault(f.parent.name.rsplit("-", 1)[0], []).append(load(f))
        if not runs:
            continue
        rows = [row for group in runs.values() for row in group]
        for key in ("input_pcm_sha256", "weights_sha256", "audio_seconds", "chunks"):
            assert len({row[key] for row in rows}) == 1, f"{label_dir.name}: {key} differs"
        entry = {"label": label_dir.name, "model_id": rows[0]["model_id"],
                 "audio_seconds": rows[0]["audio_seconds"], "chunks": rows[0]["chunks"],
                 "backends": {}}
        for backend, group in runs.items():
            values = {"runs": len(group), "device": group[0]["device"], "settings": group[0]["settings"]}
            for key in ("total_seconds", "model_load_seconds", "model_seconds",
                        "first_call_seconds", "steady_call_seconds", "rtf"):
                xs = [row[key] for row in group if row[key] is not None]
                values[key] = round(statistics.median(xs), 3) if xs else None
            entry["backends"][backend] = values
        entries.append(entry)
    report = {
        "scope": "serial runs of one binary, identical PCM / weights / audio context per label; "
                 "medians over measured runs after a warm-up pass; not SDR or quality evaluation",
        "platform": f"{platform.system()}-{platform.machine()}",
        "binary_sha256": hashlib.sha256((args.root / "ancha").read_bytes()).hexdigest()
        if (args.root / "ancha").exists() else None,
        "entries": entries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for e in entries:
        cells = "  ".join(f"{b}: total {v['total_seconds']:.2f} model {v['model_seconds']:.2f}"
                          for b, v in sorted(e["backends"].items()))
        print(f"{e['label']:20s} {cells}")


if __name__ == "__main__":
    main()
