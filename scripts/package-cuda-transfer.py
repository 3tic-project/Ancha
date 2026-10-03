#!/usr/bin/env python3
"""Package a clean Git checkout and local CUDA development assets; stdlib only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"target", ".cache", "__pycache__", ".venv", ".venv-parity", ".DS_Store"}
TEXT_SUFFIXES = {".rs", ".py", ".sh", ".toml", ".lock", ".md", ".txt", ".json",
                 ".yaml", ".yml", ".wgsl", ".patch"}
REFERENCE_DIRS = (
    "reference", "lightweight-separation-lab", "acceleration-lab", "rust-roformer-prd",
    "eplyt-reference", "speed", "imports/cuda-transfer-20261003",
)
REFERENCE_FILES = (
    "Karaoke.md", "bs_roformer_reference.py", "mel_roformer_reference.py", "model_utils_reference.py",
    "deux-config.yaml", "deux-metadata.json", "leap-metadata.json", "leap-metadata-blobs.json",
    "leap_xe_config_voc.yaml",
)
PACKAGES = ("leap-xe-voc", "deux", "hyperace-v2-voc", "hyperace-v2-inst",
            "mel-karaoke-aufr33-viperx", "mdx23c-inst-voc-hq2")
ONNX = ("UVR_MDXNET_9482.onnx", "UVR_MDXNET_KARA.onnx", "UVR_MDXNET_KARA_2.onnx",
        "UVR-MDX-NET-Inst_HQ_2.onnx")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def walk(directory, text_only=False, nested_git=False):
    if not directory.exists():
        return []
    found = []
    for base, dirs, names in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and (nested_git or d != ".git"))
        for name in sorted(names):
            path = Path(base) / name
            if name in SKIP or name.startswith("._") or name.endswith((".part", ".lock.tmp")):
                continue
            if text_only and path.suffix not in TEXT_SUFFIXES and name not in {"LICENSE", ".gitignore"}:
                continue
            found.append(path)
    return found


def select_files():
    if git("status", "--porcelain", "--untracked-files=normal").strip():
        raise ValueError("Commit the source changes first; the transfer must have a clean Git checkout")
    if not (ROOT / ".git").is_dir() or (ROOT / ".git/objects/info/alternates").exists():
        raise ValueError("A standalone .git directory without external object storage is required")
    tracked = [ROOT / os.fsdecode(p) for p in git("ls-files", "-z").split(b"\0") if p]
    if any(p.relative_to(ROOT).parts[0] in {"NO_TRACK", "NOTRACK"} for p in tracked):
        raise ValueError("NO_TRACK / NOTRACK must not be tracked by Git")
    source = set(tracked)
    for name in ("objects", "refs", "logs", "info"):
        source.update(walk(ROOT / ".git" / name, nested_git=True))
    for name in ("HEAD", "config", "index", "packed-refs", "shallow", "description"):
        path = ROOT / ".git" / name
        if path.exists():
            source.add(path)
    for name in REFERENCE_DIRS:
        source.update(walk(ROOT / "NO_TRACK" / name, text_only=True))
    source.update(ROOT / "NO_TRACK" / name for name in REFERENCE_FILES
                  if (ROOT / "NO_TRACK" / name).is_file())
    assets = set(walk(ROOT / "NO_TRACK/models"))
    # Hub metadata and conversion configs are retained, incomplete downloads are excluded.
    assets = {p for p in assets if p.suffix in {".ckpt", ".safetensors", ".onnx", ".json", ".yaml", ".yml"}}
    assets.update(walk(ROOT / "NO_TRACK/test_file"))
    assets.update(ROOT / "NO_TRACK/runs" / f"clip-{seconds}s.wav" for seconds in (3, 30))
    required = [ROOT / "NO_TRACK/models" / p / f for p in PACKAGES
                for f in ("manifest.json", "model.safetensors")]
    required += [ROOT / "NO_TRACK/models" / name for name in ONNX]
    required += [ROOT / "NO_TRACK/models" / name for name in (
        "bs_leap_xe_voc.ckpt", "becruily_deux.ckpt", "hyperace-v2-voc.ckpt", "hyperace-v2-inst.ckpt",
        "derur-download/MDX23C-8KFFT-InstVoc_HQ_2/MDX23C-8KFFT-InstVoc_HQ_2.ckpt",
        "derur-download/mel_band_roformer_karaoke_aufr33_viperx_sdr_10/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    )]
    for path in required + list(source | assets):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing file or unsupported symlink: {path}")
    if not walk(ROOT / "NO_TRACK/test_file"):
        raise ValueError("NO_TRACK/test_file is empty")
    return {"ancha-source.tar.gz": sorted(source), "ancha-assets.tar.gz": sorted(assets)}


class HashingReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.handle.read(size)
        self.digest.update(data)
        return data


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_archive(destination, paths):
    records = []
    partial = destination.with_suffix(destination.suffix + ".part")
    with tarfile.open(partial, "w:gz", compresslevel=1, format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            before = path.stat()
            name = "Ancha/" + path.relative_to(ROOT).as_posix()
            if before.st_size > 10_000_000:
                print(f"  {name}: {before.st_size / 2**20:.1f} MiB", flush=True)
            info = tarfile.TarInfo(name)
            info.size = before.st_size
            info.mode = 0o755 if before.st_mode & 0o111 else 0o644
            info.mtime = 0
            with path.open("rb") as handle:
                reader = HashingReader(handle)
                archive.addfile(info, reader)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"File changed during packaging: {path}")
            records.append({"path": name, "bytes": info.size, "mode": info.mode,
                            "sha256": reader.digest.hexdigest()})
    partial.rename(destination)
    return {"bytes": destination.stat().st_size, "sha256": sha256_file(destination), "files": records}


def verify(directory):
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    seen_all = set()
    for name, record in manifest["archives"].items():
        path = directory / name
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")
        expected = {r["path"]: r for r in record["files"]}
        seen = set()
        print(f"Verifying {name}: {len(expected)} files", flush=True)
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                if (not member.isfile() or not parts or parts[0] != "Ancha" or ".." in parts
                        or member.name not in expected or member.name in seen_all):
                    raise ValueError(f"Unexpected or unsafe archive entry: {member.name}")
                row = expected[member.name]
                with archive.extractfile(member) as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if member.size != row["bytes"] or member.mode != row["mode"] or digest != row["sha256"]:
                    raise ValueError(f"Member checksum/metadata mismatch: {member.name}")
                seen.add(member.name)
                seen_all.add(member.name)
        if seen != set(expected):
            raise ValueError(f"Missing archive members: {name}")
    print(f"Verified {len(seen_all)} files and both archive checksums", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory (must not exist)")
    parser.add_argument("--verify", type=Path, help="Verify an existing transfer directory")
    parser.add_argument("--list", action="store_true", help="Print the selection without packaging")
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
        return
    groups = select_files()
    for name, paths in groups.items():
        print(f"{name}: {len(paths)} files, {sum(p.stat().st_size for p in paths) / 2**20:.1f} MiB", flush=True)
    if args.list:
        for name, paths in groups.items():
            for path in paths:
                print(name, path.relative_to(ROOT).as_posix())
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = (args.output or ROOT / "NO_TRACK/transfers" / f"ancha-cuda-{stamp}").resolve()
    # A completed directory is never overwritten; failed .part files remain inspectable.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "created_utc": stamp, "git_commit": git("rev-parse", "HEAD").decode().strip(),
                "root": "Ancha", "archives": {}}
    for name, paths in groups.items():
        print(f"Writing {name}", flush=True)
        manifest["archives"][name] = write_archive(output / name, paths)
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(ROOT / "docs/cuda-transfer.md", output / "README-transfer.md")
    checksums = {name: row["sha256"] for name, row in manifest["archives"].items()}
    checksums.update({name: sha256_file(output / name) for name in ("MANIFEST.json", "README-transfer.md")})
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    verify(output)
    print(f"Ready: {output}", flush=True)


if __name__ == "__main__":
    main()
