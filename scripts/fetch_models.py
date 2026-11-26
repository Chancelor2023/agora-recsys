"""Download the frozen models and processed data instead of rebuilding them.

The archive is published as a GitHub release asset and holds exactly the files listed in
artifacts/freeze/manifest.json (models, text encoder, processed IDEA data): the state that
was evaluated once on the test split. Every file is checked against the manifest hash.

    python scripts/fetch_models.py              # download, extract, verify (about 70 MB)
    python scripts/fetch_models.py --zip FILE   # use an archive already downloaded
    python scripts/fetch_models.py pack         # maintainers: build the archive from the
                                                # frozen files on the freeze machine

Standard library only: it can run before or while the requirements install.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "freeze" / "manifest.json"
TAG = "frozen-2026-11-22"
ARCHIVE = f"agora-{TAG}.zip"
URL = f"https://github.com/Chancelor2023/agora-recsys/releases/download/{TAG}/{ARCHIVE}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_files() -> dict[Path, str]:
    """Repository-relative path -> expected SHA-256, for the data and model files."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    roots = {"data": Path(manifest["paths"]["data"]), "models": Path(manifest["paths"]["models"])}
    files = {}
    for name, digest in manifest["files"].items():
        kind, _, rest = name.partition("/")
        if kind in roots:
            files[roots[kind] / rest] = digest
    return files


def check(files: dict[Path, str]) -> list[str]:
    problems = []
    for rel, digest in files.items():
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel.as_posix()}: missing")
        elif sha256(path) != digest:
            problems.append(f"{rel.as_posix()}: differs from the frozen file")
    return problems


def download(url: str, target: Path) -> None:
    start = time.time()

    def progress(blocks, size, total):
        if total > 0:
            done = min(blocks * size, total)
            print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, target, reporthook=progress)
    print(f"\r  downloaded {target.stat().st_size / 1e6:.1f} MB in {time.time() - start:.0f} s")


def fetch(args) -> None:
    files = frozen_files()
    if not check(files):
        print(f"Frozen models and data already in place ({len(files)} files verified).")
        return
    archive = Path(args.zip) if args.zip else ROOT / "artifacts" / ARCHIVE
    if not args.zip:
        archive.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        download(URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        wanted = {rel.as_posix() for rel in files}
        for member in bundle.namelist():
            if member in wanted:  # extract only the expected paths
                bundle.extract(member, ROOT)
    problems = check(files)
    if problems:
        sys.exit("Archive does not match the frozen manifest:\n- " + "\n- ".join(problems))
    print(f"Frozen models and data in place ({len(files)} files verified against the manifest).")


def pack(args) -> None:
    files = frozen_files()
    problems = check(files)
    if problems:
        sys.exit("These files are not the frozen ones; refusing to pack:\n- " + "\n- ".join(problems))
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for rel in sorted(files):
            bundle.write(ROOT / rel, rel.as_posix())
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(files)} frozen files).")
    print(f"Publish it as release asset {ARCHIVE} under tag {TAG}.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch", "pack"])
    parser.add_argument("--zip", help="Archive already downloaded (fetch)")
    parser.add_argument("--out", default=ARCHIVE, help="Archive to write (pack)")
    args = parser.parse_args(argv)
    (pack if args.command == "pack" else fetch)(args)


if __name__ == "__main__":
    main()
