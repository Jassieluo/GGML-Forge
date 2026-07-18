#!/usr/bin/env python3
"""Update llama.cpp, patch root GGML, and mirror its conversion tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
DEPENDENCY = ROOT / "scripts/maintenance/dependencies/llama_cpp"
CONFIG = DEPENDENCY / "config.json"
LOCK = DEPENDENCY / "lock.json"
UPSTREAM = ROOT / "src/categories/llm/providers/llama_cpp/llama.cpp"
ACTIVE_GGML = ROOT / "ggml"
TOOLS_PROVIDER = ROOT / "scripts/conversion/categories/llm/providers/llama_cpp"
ACTIVE_TOOLS = TOOLS_PROVIDER / "tools"
SOURCE_INFO = TOOLS_PROVIDER / "source.json"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load maintenance module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.check_call(args, cwd=cwd)


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def git_status(path: Path) -> str:
    return output("git", "status", "--porcelain", "--", path.relative_to(ROOT).as_posix())


def restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def check() -> int:
    commands = (
        DEPENDENCY / "sync_ggml.py",
        DEPENDENCY / "sync_conversion.py",
    )
    for script in commands:
        result = subprocess.run([sys.executable, str(script), "--check"], cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


def update(target_ref: str | None, force: bool) -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    remote = str(config["remote"])
    target = target_ref or str(config["default_ref"])
    if output("git", "status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("llama.cpp submodule has local modifications")
    dirty = [
        name
        for name, path in (("root GGML", ACTIVE_GGML), ("copied conversion tools", ACTIVE_TOOLS))
        if path.exists() and git_status(path)
    ]
    if dirty and not force:
        raise RuntimeError(f"{', '.join(dirty)} have local changes; commit them or use --force")

    old_commit = output("git", "rev-parse", "HEAD", cwd=UPSTREAM)
    ggml_sync = load_module("forge_sync_llama_ggml", DEPENDENCY / "sync_ggml.py")
    conversion_sync = load_module("forge_sync_llama_conversion", DEPENDENCY / "sync_conversion.py")
    staged_ggml: Path | None = None
    staged_tools: Path | None = None
    transaction: Path | None = None
    installed = False
    old_lock = LOCK.read_bytes() if LOCK.exists() else None
    old_source = SOURCE_INFO.read_bytes() if SOURCE_INFO.exists() else None

    try:
        run("git", "fetch", "--tags", remote, cwd=UPSTREAM)
        run("git", "checkout", "--detach", target, cwd=UPSTREAM)
        staged_ggml, ggml_info = ggml_sync.stage()
        staged_tools, conversion_info = conversion_sync.stage()

        transaction = Path(tempfile.mkdtemp(prefix="llama-update-", dir=ROOT / "tmp"))
        previous_ggml = transaction / "ggml.previous"
        previous_tools = transaction / "tools.previous"
        ACTIVE_GGML.rename(previous_ggml)
        staged_ggml.rename(ACTIVE_GGML)
        if ACTIVE_TOOLS.exists():
            ACTIVE_TOOLS.rename(previous_tools)
        TOOLS_PROVIDER.mkdir(parents=True, exist_ok=True)
        staged_tools.rename(ACTIVE_TOOLS)

        combined_lock = json.loads(old_lock.decode("utf-8")) if old_lock else {}
        combined_lock.update(ggml_info)
        combined_lock.update({
            "conversion_scripts": conversion_info["scripts"],
            "conversion_tools_tree_sha256": conversion_info["tree_sha256"],
        })
        LOCK.write_text(json.dumps(combined_lock, indent=2) + "\n", encoding="utf-8")
        SOURCE_INFO.write_text(json.dumps(conversion_info, indent=2) + "\n", encoding="utf-8")
        installed = True
        shutil.rmtree(transaction, ignore_errors=True)
        transaction = None
        print(f"Updated llama.cpp, root GGML, patches, and conversion tools to {conversion_info['llama_cpp_commit']}")
        return 0
    except Exception:
        if transaction is not None:
            previous_ggml = transaction / "ggml.previous"
            previous_tools = transaction / "tools.previous"
            if previous_ggml.exists():
                if ACTIVE_GGML.exists():
                    shutil.rmtree(ACTIVE_GGML)
                previous_ggml.rename(ACTIVE_GGML)
            if previous_tools.exists():
                if ACTIVE_TOOLS.exists():
                    shutil.rmtree(ACTIVE_TOOLS)
                previous_tools.rename(ACTIVE_TOOLS)
        restore_file(LOCK, old_lock)
        restore_file(SOURCE_INFO, old_source)
        if not installed:
            run("git", "checkout", "--detach", old_commit, cwd=UPSTREAM)
        raise
    finally:
        if staged_ggml is not None:
            shutil.rmtree(staged_ggml.parent, ignore_errors=True)
        if staged_tools is not None:
            shutil.rmtree(staged_tools.parent, ignore_errors=True)
        if transaction is not None:
            shutil.rmtree(transaction, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="commit, tag, or remote ref; defaults to dependency config")
    parser.add_argument("--check", action="store_true", help="verify without fetching or changing files")
    parser.add_argument("--force", action="store_true", help="replace locally changed synchronized outputs")
    args = parser.parse_args()
    try:
        return check() if args.check else update(args.ref, args.force)
    except (RuntimeError, OSError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"llama.cpp update failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
