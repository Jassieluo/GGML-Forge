#!/usr/bin/env python3
"""Mirror whisper.cpp model conversion tools into the ASR provider."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "src/categories/asr/providers/whisper_cpp/whisper.cpp"
UPSTREAM_MODELS = UPSTREAM / "models"
PROVIDER = ROOT / "scripts/conversion/categories/asr/providers/whisper_cpp"
ACTIVE_TOOLS = PROVIDER / "tools"
SOURCE_INFO = PROVIDER / "source.json"
LOCK = Path(__file__).with_name("lock.json")
PATTERNS = ("convert-*.py", "ggml_to_pt.py", "requirements-*.txt")
REQUIRED = {"convert-pt-to-ggml.py", "convert-h5-to-ggml.py"}


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def sources() -> list[Path]:
    result: dict[str, Path] = {}
    for pattern in PATTERNS:
        for path in UPSTREAM_MODELS.glob(pattern):
            if path.is_file():
                result[path.name] = path
    return [result[name] for name in sorted(result)]


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate(staged: Path) -> None:
    names = {path.name for path in staged.iterdir() if path.is_file()}
    missing = sorted(REQUIRED - names)
    if missing:
        raise RuntimeError(f"required whisper.cpp converters are missing: {', '.join(missing)}")
    for path in staged.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def stage() -> tuple[Path, dict[str, object]]:
    if not UPSTREAM_MODELS.is_dir():
        raise RuntimeError(f"whisper.cpp models directory is missing: {UPSTREAM_MODELS}")
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="whisper-conversion-stage-", dir=ROOT / "tmp"))
    staged = temporary / "tools"
    staged.mkdir()
    for source in sources():
        shutil.copy2(source, staged / source.name)
    validate(staged)
    info: dict[str, object] = {
        "schema": 1,
        "whisper_cpp_commit": output("git", "rev-parse", "HEAD", cwd=UPSTREAM),
        "scripts": [path.name for path in staged.iterdir() if path.suffix == ".py"],
        "tree_sha256": digest_tree(staged),
    }
    info["scripts"] = sorted(info["scripts"])
    return staged, info


def compare_trees(left: Path, right: Path) -> list[str]:
    left_files = {path.relative_to(left).as_posix(): path for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right).as_posix(): path for path in right.rglob("*") if path.is_file()}
    differences = []
    for relative in sorted(left_files.keys() | right_files.keys()):
        if relative not in left_files:
            differences.append(f"unexpected copied file: {relative}")
        elif relative not in right_files:
            differences.append(f"missing copied file: {relative}")
        elif left_files[relative].read_bytes() != right_files[relative].read_bytes():
            differences.append(f"content differs: {relative}")
    return differences


def install(staged: Path, info: dict[str, object], force: bool) -> None:
    if ACTIVE_TOOLS.exists():
        status = output("git", "status", "--porcelain", "--", ACTIVE_TOOLS.relative_to(ROOT).as_posix())
        if status and not force:
            raise RuntimeError("copied whisper.cpp conversion tools have local changes; use --force")
    PROVIDER.mkdir(parents=True, exist_ok=True)
    backup = staged.parent / "tools.previous"
    try:
        if ACTIVE_TOOLS.exists():
            ACTIVE_TOOLS.rename(backup)
        staged.rename(ACTIVE_TOOLS)
        SOURCE_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    except Exception:
        if ACTIVE_TOOLS.exists():
            shutil.rmtree(ACTIVE_TOOLS)
        if backup.exists():
            backup.rename(ACTIVE_TOOLS)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    shutil.rmtree(staged.parent, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--stage-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        staged, info = stage()
        if args.stage_only:
            print(staged)
            return 0
        if args.check:
            differences = compare_trees(staged, ACTIVE_TOOLS) if ACTIVE_TOOLS.exists() else ["tools directory is missing"]
            shutil.rmtree(staged.parent, ignore_errors=True)
            current = json.loads(SOURCE_INFO.read_text(encoding="utf-8")) if SOURCE_INFO.exists() else {}
            if differences or current != info:
                print("whisper.cpp conversion tools are out of sync", file=sys.stderr)
                for difference in differences[:100]:
                    print(f"  - {difference}", file=sys.stderr)
                return 1
            print("whisper.cpp conversion tools are synchronized")
            return 0
        install(staged, info, args.force)
        print(f"Copied whisper.cpp conversion tools from {info['whisper_cpp_commit']}")
        return 0
    except (OSError, RuntimeError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"whisper.cpp conversion sync failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
