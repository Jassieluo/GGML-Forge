#!/usr/bin/env python3
"""Rebuild the active root GGML from llama.cpp's pinned GGML plus Forge bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "src/categories/llm/providers/llama_cpp/llama.cpp"
UPSTREAM_GGML = UPSTREAM / "ggml"
ACTIVE_GGML = ROOT / "ggml"
BRIDGE = ROOT / "scripts/maintenance/patches/llama_cpp/ggml_bridge"
ASSETS = BRIDGE / "assets"
LOCK = Path(__file__).with_name("lock.json")
sys.path.insert(0, str(BRIDGE))

from bridge import BridgeConflict, apply_bridge, verify_bridge  # noqa: E402


def run(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def ensure_layout() -> None:
    for path in (ROOT, UPSTREAM, UPSTREAM_GGML, ACTIVE_GGML, ASSETS):
        if not path.exists():
            raise RuntimeError(f"required path does not exist: {path}")
    if ACTIVE_GGML.resolve().parent != ROOT.resolve():
        raise RuntimeError("active GGML is outside the workspace root")


def stage() -> tuple[Path, dict[str, object]]:
    ensure_layout()
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="ggml-forge-stage-", dir=ROOT / "tmp"))
    staged = temp / "ggml"
    shutil.copytree(UPSTREAM_GGML, staged)
    apply_bridge(staged, ASSETS)
    verify_bridge(staged)

    commit = run("git", "rev-parse", "HEAD", cwd=UPSTREAM)
    ggml_tree = run("git", "rev-parse", "HEAD:ggml", cwd=UPSTREAM)
    lock = {
        "schema": 1,
        "llama_cpp_commit": commit,
        "llama_cpp_describe": run("git", "describe", "--always", "--tags", cwd=UPSTREAM),
        "ggml_upstream_tree": ggml_tree,
        "ggml_forge_tree_sha256": digest_tree(staged),
        "bridge_revision": 1,
    }
    return staged, lock


def compare_trees(left: Path, right: Path) -> list[str]:
    left_files = {p.relative_to(left).as_posix(): p for p in left.rglob("*") if p.is_file()}
    right_files = {p.relative_to(right).as_posix(): p for p in right.rglob("*") if p.is_file()}
    differences: list[str] = []
    for relative in sorted(left_files.keys() | right_files.keys()):
        if relative not in left_files:
            differences.append(f"missing active file: {relative}")
        elif relative not in right_files:
            differences.append(f"unexpected active file: {relative}")
        elif left_files[relative].read_bytes() != right_files[relative].read_bytes():
            differences.append(f"content differs: {relative}")
    return differences


def install(staged: Path, lock: dict[str, object], force: bool) -> None:
    status = run("git", "status", "--porcelain", "--", "ggml")
    if status and not force:
        raise RuntimeError("root ggml has uncommitted changes; commit them or use --force")

    stage_parent = staged.parent
    backup = stage_parent / "ggml.previous"
    active = ACTIVE_GGML.resolve()
    try:
        active.rename(backup)
        staged.rename(active)
    except Exception:
        if not active.exists() and backup.exists():
            backup.rename(active)
        raise

    current_lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    current_lock.update(lock)
    LOCK.write_text(json.dumps(current_lock, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(backup)
    shutil.rmtree(stage_parent, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="atomically replace root ggml")
    mode.add_argument("--check", action="store_true", help="verify root ggml is reproducible")
    mode.add_argument("--stage-only", action="store_true", help="build and print a temporary staged path")
    parser.add_argument("--force", action="store_true", help="allow replacing a dirty root ggml")
    args = parser.parse_args()

    try:
        staged, lock = stage()
        if args.stage_only:
            print(staged)
            return 0
        if args.check:
            differences = compare_trees(staged, ACTIVE_GGML)
            shutil.rmtree(staged.parent, ignore_errors=True)
            if differences:
                print("GGML stack is out of sync:", file=sys.stderr)
                for difference in differences[:100]:
                    print(f"  - {difference}", file=sys.stderr)
                return 1
            current_lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
            if any(current_lock.get(key) != value for key, value in lock.items()):
                print("GGML contents match but lock metadata is stale", file=sys.stderr)
                return 1
            print("GGML stack is synchronized and reproducible")
            return 0
        install(staged, lock, args.force)
        print(f"Installed GGML from llama.cpp {lock['llama_cpp_commit']}")
        return 0
    except (BridgeConflict, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"sync failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
