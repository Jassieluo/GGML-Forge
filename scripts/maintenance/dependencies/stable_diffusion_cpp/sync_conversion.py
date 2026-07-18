#!/usr/bin/env python3
"""Mirror stable-diffusion.cpp conversion tools into visual_generation."""

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
UPSTREAM = ROOT / "src/categories/visual_generation/providers/stable_diffusion_cpp/stable-diffusion.cpp"
UPSTREAM_SCRIPTS = UPSTREAM / "scripts"
PROVIDER = ROOT / "scripts/conversion/categories/visual_generation/providers/stable_diffusion_cpp"
ACTIVE_TOOLS = PROVIDER / "tools"
SOURCE_INFO = PROVIDER / "source.json"
PATTERN = "convert*.py"
REQUIRED = {
    "convert_fp8_scale_to_bf16.py",
    "convert_qwen3_vl.py",
    "convert_sefi.py",
    "convert_yolov8_to_safetensors.py",
}


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


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
        raise RuntimeError(f"required stable-diffusion.cpp converters are missing: {', '.join(missing)}")
    for path in staged.glob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def stage() -> tuple[Path, dict[str, object]]:
    if not UPSTREAM_SCRIPTS.is_dir():
        raise RuntimeError(f"stable-diffusion.cpp scripts directory is missing: {UPSTREAM_SCRIPTS}")
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="stable-diffusion-conversion-stage-", dir=ROOT / "tmp"))
    staged = temporary / "tools"
    staged.mkdir()
    for source in sorted(UPSTREAM_SCRIPTS.glob(PATTERN)):
        if source.is_file():
            shutil.copy2(source, staged / source.name)
    validate(staged)
    info: dict[str, object] = {
        "schema": 1,
        "stable_diffusion_cpp_commit": output("git", "rev-parse", "HEAD", cwd=UPSTREAM),
        "scripts": sorted(path.name for path in staged.glob("*.py")),
        "tree_sha256": digest_tree(staged),
    }
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
            raise RuntimeError("copied stable-diffusion.cpp conversion tools have local changes; use --force")
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
    parser = argparse.ArgumentParser(description=__doc__)
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
                print("stable-diffusion.cpp conversion tools are out of sync", file=sys.stderr)
                for difference in differences[:100]:
                    print(f"  - {difference}", file=sys.stderr)
                return 1
            print("stable-diffusion.cpp conversion tools are synchronized")
            return 0
        install(staged, info, args.force)
        print(f"Copied stable-diffusion.cpp conversion tools from {info['stable_diffusion_cpp_commit']}")
        return 0
    except (OSError, RuntimeError, SyntaxError, subprocess.CalledProcessError) as error:
        print(f"stable-diffusion.cpp conversion sync failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
