#!/usr/bin/env python3
"""Update and patch every managed upstream dependency."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
UPDATES = Path(__file__).resolve().parent


def update_scripts() -> list[Path]:
    scripts = sorted(
        path for path in UPDATES.glob("*.py")
        if path.name not in {"__init__.py", "all.py"}
    )
    by_name = {path.stem: path for path in scripts}
    ordered: list[Path] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in visiting:
            raise RuntimeError(f"dependency update cycle includes {name}")
        if name not in by_name:
            raise RuntimeError(f"managed dependency has no updater: {name}")
        visiting.add(name)
        config_path = ROOT / "scripts" / "maintenance" / "dependencies" / name / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for dependency in config.get("depends_on", []):
            visit(str(dependency))
        visiting.remove(name)
        visited.add(name)
        ordered.append(by_name[name])

    for name in sorted(by_name):
        visit(name)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify every dependency without updating")
    parser.add_argument("--force", action="store_true", help="allow each updater to replace synchronized outputs")
    args = parser.parse_args()

    try:
        scripts = update_scripts()
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"cannot resolve dependency update order: {error}", file=sys.stderr)
        return 1
    if not scripts:
        print("no managed dependencies found", file=sys.stderr)
        return 1
    for script in scripts:
        command = [sys.executable, str(script)]
        if args.check:
            command.append("--check")
        if args.force:
            command.append("--force")
        print(f"==> {script.stem}", flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            print(f"update failed: {script.stem}", file=sys.stderr)
            return result.returncode
    print("All managed dependencies are synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
