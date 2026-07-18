#!/usr/bin/env python3
"""Update and verify the managed EUI-NEO UI framework submodule."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEPENDENCY = ROOT / "scripts/maintenance/dependencies/eui_neo"
CONFIG = DEPENDENCY / "config.json"
LOCK = DEPENDENCY / "lock.json"
UPSTREAM = ROOT / "ui/eui_neo"


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.check_call(args, cwd=cwd)


def dependency_info() -> dict[str, object]:
    return {
        "schema": 1,
        "eui_neo_commit": output("git", "rev-parse", "HEAD", cwd=UPSTREAM),
        "eui_neo_describe": output("git", "describe", "--always", "--tags", cwd=UPSTREAM),
        "eui_neo_tree": output("git", "rev-parse", "HEAD^{tree}", cwd=UPSTREAM),
        "patch_revision": 0,
    }


def require_clean() -> None:
    if output("git", "status", "--porcelain", "--untracked-files=normal", cwd=UPSTREAM):
        raise RuntimeError("EUI-NEO submodule has local modifications")


def check() -> int:
    require_clean()
    expected = dependency_info()
    current = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    if current != expected:
        print("EUI-NEO lock metadata is stale", file=sys.stderr)
        return 1
    print(f"EUI-NEO is synchronized at {expected['eui_neo_describe']}")
    return 0


def update(target_ref: str | None) -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    remote = str(config["remote"])
    target = target_ref or str(config["default_ref"])
    require_clean()
    old_commit = output("git", "rev-parse", "HEAD", cwd=UPSTREAM)
    old_lock = LOCK.read_bytes() if LOCK.exists() else None
    try:
        run("git", "fetch", "--tags", remote, cwd=UPSTREAM)
        run("git", "checkout", "--detach", target, cwd=UPSTREAM)
        info = dependency_info()
        LOCK.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"Updated EUI-NEO to {info['eui_neo_describe']} ({info['eui_neo_commit']})")
        return 0
    except Exception:
        if old_lock is None:
            LOCK.unlink(missing_ok=True)
        else:
            LOCK.write_bytes(old_lock)
        run("git", "checkout", "--detach", old_commit, cwd=UPSTREAM)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="commit, tag, or remote ref; defaults to dependency config")
    parser.add_argument("--check", action="store_true", help="verify without fetching or changing files")
    parser.add_argument("--force", action="store_true", help="accepted for all.py compatibility; never discards UI changes")
    args = parser.parse_args()
    try:
        return check() if args.check else update(args.ref)
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"EUI-NEO update failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
