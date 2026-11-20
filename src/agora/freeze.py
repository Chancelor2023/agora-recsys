"""Freeze everything the final test depends on, and refuse the test if anything moved.

    python -m agora.freeze create --note "<validation results and decisions this freeze follows>"
    python -m agora.freeze verify

--data/--models/--config/--rerank-config default to the paths of agora.experiment.

The manifest (artifacts/freeze/manifest.json, i.e. next to --models) records the SHA-256 of the models,
embeddings and fitted encoder, of every processed input (including the test files,
hashed as bytes only), of both configs (so lambda and agent targets are fixed), of
the source code, plus the git commit, the IDEA dataset commit, package versions and
the closed list of rows the test will report. `python -m agora.experiment --split
test` refuses to run unless `verify` passes. Use the frozen artifacts as they are:
a rebuild on another machine does not reproduce the neural model byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL_FILES = ["model.json", "neural.pt", "embeddings.npz"]
MODEL_FILES_IF_PRESENT = ["embeddings.npz.json", "embeddings.npz.encoder.joblib"]
DATA_FILES = ["articles.json", "reads_train.csv"] + [
    name for split in ("train", "validation", "test")
    for name in (f"feeds_{split}.jsonl", f"{split}_labels.json")]
DATA_FILES_IF_PRESENT = ["eligible_users.json"]
DEFAULTS = {"data": "data/processed", "models": "artifacts/models",
            "config": "configs/experiment.toml", "rerank_config": "configs/reranking_agents.json"}


def freeze_dir(models) -> Path:
    """Manifest and run log live next to the models folder (artifacts/freeze by default)."""
    return Path(models).parent / "freeze"


def manifest_path(models) -> Path:
    return freeze_dir(models) / "manifest.json"


def run_log_path(models) -> Path:
    return freeze_dir(models) / "test_runs.jsonl"


def frozen_files(data, models, config, rerank_config, present_only=False) -> dict[str, Path]:
    """Logical name -> path of every file the test result depends on."""
    data, models = Path(data), Path(models)
    files = {f"models/{n}": models / n for n in MODEL_FILES}
    files |= {f"models/{n}": models / n for n in MODEL_FILES_IF_PRESENT
              if not present_only or (models / n).exists()}
    files |= {f"data/{n}": data / n for n in DATA_FILES}
    files |= {f"data/{n}": data / n for n in DATA_FILES_IF_PRESENT
              if not present_only or (data / n).exists()}
    files |= {"configs/experiment.toml": Path(config), "configs/reranking_agents.json": Path(rerank_config)}
    return files


PACKAGES = ["numpy", "pandas", "scikit-learn", "torch", "joblib", "fastapi"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash(root: Path = REPO) -> str:
    """One hash over every Python file of src/agora (path + content, LF-normalised)."""
    digest = hashlib.sha256()
    for path in sorted((root / "src" / "agora").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _git(root: Path, *args) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions():
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def current_state(data, models, config, rerank_config, idea_dir: Path | None = None) -> dict:
    from agora.experiment import TEST_ROWS

    files = frozen_files(data, models, config, rerank_config, present_only=True)
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise SystemExit(f"Cannot freeze, missing files: {missing}")
    rerank = json.loads(Path(rerank_config).read_text(encoding="utf-8"))
    idea_commit = _git(idea_dir, "rev-parse", "HEAD") if idea_dir and idea_dir.exists() else None
    return {
        "paths": {k: str(v) for k, v in DEFAULTS.items()} | {
            "data": str(data), "models": str(models), "config": str(config), "rerank_config": str(rerank_config)},
        "files": {name: sha256(path) for name, path in files.items()},
        "source_sha256": source_hash(),
        "git_commit": _git(REPO, "rev-parse", "HEAD"),
        "git_tracked_changes": bool(_git(REPO, "status", "--porcelain", "--untracked-files=no")),
        "idea_commit": idea_commit,
        "packages": _versions(),
        "test_rows": TEST_ROWS,
        "lambda_selected": float(rerank["lambda"]),
        "lambda_test": sorted({0.0, float(rerank["lambda"])}),
        "agents": rerank["agents"],
    }


def verify(data=DEFAULTS["data"], models=DEFAULTS["models"], config=DEFAULTS["config"],
           rerank_config=DEFAULTS["rerank_config"]) -> list[str]:
    """Return the list of differences with the manifest (empty = frozen state intact)."""
    path = manifest_path(models)
    if not path.exists():
        return [f"no manifest: run python -m agora.freeze create (expected {path})"]
    frozen = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    now = frozen_files(data, models, config, rerank_config, present_only=True)
    for name in sorted(set(frozen["files"]) | set(now)):
        file = frozen_files(data, models, config, rerank_config).get(name) or now.get(name)
        if name not in frozen["files"]:
            problems.append(f"{name}: added since the freeze")
        elif file is None or not file.exists():
            problems.append(f"{name}: missing")
        elif sha256(file) != frozen["files"][name]:
            problems.append(f"{name}: changed since the freeze")
    if source_hash() != frozen["source_sha256"]:
        problems.append("src/agora: code changed since the freeze")
    from agora.experiment import TEST_ROWS
    if TEST_ROWS != frozen["test_rows"]:
        problems.append("test rows changed since the freeze")
    for name, version in _versions().items():
        if frozen["packages"].get(name) != version:
            problems.append(f"package {name}: {frozen['packages'].get(name)} at freeze, {version} now")
    return problems


def _add_paths(parser):
    parser.add_argument("--data", default=DEFAULTS["data"])
    parser.add_argument("--models", default=DEFAULTS["models"])
    parser.add_argument("--config", default=DEFAULTS["config"])
    parser.add_argument("--rerank-config", default=DEFAULTS["rerank_config"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--note", required=True, help="Validation results and team decisions behind this freeze")
    create.add_argument("--idea-dir", default="data/raw/Datasets", help="Git clone of Informfully/Datasets")
    create.add_argument("--replace", action="store_true", help="Replace a manifest (refused once a test run exists)")
    _add_paths(create)
    _add_paths(sub.add_parser("verify"))
    args = parser.parse_args(argv)
    paths = (args.data, args.models, args.config, args.rerank_config)
    manifest = manifest_path(args.models)
    if args.command == "verify":
        problems = verify(*paths)
        if problems:
            print("NOT FROZEN / CHANGED:\n- " + "\n- ".join(problems))
            sys.exit(1)
        print(f"Frozen state intact ({manifest})")
        return
    if run_log_path(args.models).exists():
        sys.exit("A test run already exists: the freeze can no longer be replaced.")
    if manifest.exists() and not args.replace:
        sys.exit(f"{manifest} exists; use --replace (only before any test run).")
    if not args.note.strip():
        sys.exit("--note must say which validation results and decisions this freeze follows")
    state = current_state(*paths, Path(args.idea_dir))
    if state["git_tracked_changes"]:
        print("warning: tracked files differ from the git commit; commit them before freezing")
    state.update({"created_at": datetime.now(timezone.utc).isoformat(), "note": args.note})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Frozen {len(state['files'])} files, source, configs and packages in {manifest}")
    print(f"Test rows: {state['test_rows']}; lambda on test: {state['lambda_test']}")


if __name__ == "__main__":
    main()
