#!/usr/bin/env python3
"""Update stable-diffusion.cpp, validate root GGML reuse, and mirror converters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[3]
DEPENDENCY = ROOT / "scripts/maintenance/dependencies/stable_diffusion_cpp"
CONFIG = DEPENDENCY / "config.json"
LOCK = DEPENDENCY / "lock.json"
LLAMA_LOCK = ROOT / "scripts/maintenance/dependencies/llama_cpp/lock.json"
UPSTREAM = ROOT / "src/categories/visual_generation/providers/stable_diffusion_cpp/stable-diffusion.cpp"
PATCH = ROOT / "scripts/maintenance/patches/stable_diffusion_cpp/patch.py"
TOOLS_PROVIDER = ROOT / "scripts/conversion/categories/visual_generation/providers/stable_diffusion_cpp"
ACTIVE_TOOLS = TOOLS_PROVIDER / "tools"
SOURCE_INFO = TOOLS_PROVIDER / "source.json"
BRIDGE_REVISION = 1
GGML_MAX_NAME = 160
NESTED_SUBMODULES = (
    "ggml",
    "thirdparty/libwebp",
    "thirdparty/libwebm",
    "examples/server/frontend",
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load maintenance module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def run(*args: str, cwd: Path = ROOT) -> None:
    subprocess.check_call(args, cwd=cwd)


def gitlink(path: str) -> str:
    entry = output("git", "ls-tree", "HEAD", "--", path, cwd=UPSTREAM).split()
    if len(entry) < 3 or entry[1] != "commit":
        raise RuntimeError(f"expected nested submodule gitlink is missing: {path}")
    return entry[2]


def dependency_info(patch_module: ModuleType, conversion_info: dict[str, object]) -> dict[str, object]:
    llama_lock = json.loads(LLAMA_LOCK.read_text(encoding="utf-8"))
    return {
        "schema": 1,
        "stable_diffusion_cpp_commit": output("git", "rev-parse", "HEAD", cwd=UPSTREAM),
        "stable_diffusion_cpp_describe": output("git", "describe", "--always", "--tags", cwd=UPSTREAM),
        "bundled_ggml_tree": gitlink("ggml"),
        "nested_submodules": {path: gitlink(path) for path in NESTED_SUBMODULES},
        "root_llama_cpp_commit": llama_lock["llama_cpp_commit"],
        "root_ggml_tree_sha256": llama_lock["ggml_forge_tree_sha256"],
        "ggml_max_name": GGML_MAX_NAME,
        "bridge_revision": BRIDGE_REVISION,
        "patch_revision": patch_module.PATCH_REVISION,
        "conversion_scripts": conversion_info["scripts"],
        "conversion_tools_tree_sha256": conversion_info["tree_sha256"],
    }


def check() -> int:
    patch_module = load_module("forge_stable_diffusion_patch", PATCH)
    patch_module.verify(UPSTREAM)
    if subprocess.run([sys.executable, str(DEPENDENCY / "sync_conversion.py"), "--check"], cwd=ROOT).returncode:
        return 1
    source = json.loads(SOURCE_INFO.read_text(encoding="utf-8"))
    expected = dependency_info(patch_module, source)
    current = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    if current != expected:
        print("stable-diffusion.cpp lock metadata is stale", file=sys.stderr)
        return 1
    print("stable-diffusion.cpp is synchronized with the root GGML contract")
    return 0


def update(target_ref: str | None, force: bool) -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    remote = str(config["remote"])
    target = target_ref or str(config["default_ref"])
    if output("git", "status", "--porcelain", cwd=UPSTREAM):
        raise RuntimeError("stable-diffusion.cpp submodule has local modifications")
    if ACTIVE_TOOLS.exists():
        status = output("git", "status", "--porcelain", "--", ACTIVE_TOOLS.relative_to(ROOT).as_posix())
        if status and not force:
            raise RuntimeError("copied stable-diffusion.cpp conversion tools have local changes; use --force")

    old_commit = output("git", "rev-parse", "HEAD", cwd=UPSTREAM)
    old_lock = LOCK.read_bytes() if LOCK.exists() else None
    old_source = SOURCE_INFO.read_bytes() if SOURCE_INFO.exists() else None
    patch_module = load_module("forge_stable_diffusion_patch", PATCH)
    sync_module = load_module("forge_sync_stable_diffusion_conversion", DEPENDENCY / "sync_conversion.py")
    staged: Path | None = None
    backup: Path | None = None
    try:
        run("git", "fetch", "--tags", remote, cwd=UPSTREAM)
        run("git", "checkout", "--detach", target, cwd=UPSTREAM)
        patch_module.apply(UPSTREAM)
        staged, conversion_info = sync_module.stage()
        info = dependency_info(patch_module, conversion_info)

        TOOLS_PROVIDER.mkdir(parents=True, exist_ok=True)
        backup = staged.parent / "tools.previous"
        if ACTIVE_TOOLS.exists():
            ACTIVE_TOOLS.rename(backup)
        staged.rename(ACTIVE_TOOLS)
        SOURCE_INFO.write_text(json.dumps(conversion_info, indent=2) + "\n", encoding="utf-8")
        LOCK.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(backup, ignore_errors=True)
        backup = None
        print(f"Updated stable-diffusion.cpp and conversion tools to {info['stable_diffusion_cpp_commit']}")
        return 0
    except Exception:
        if backup is not None and backup.exists():
            if ACTIVE_TOOLS.exists():
                shutil.rmtree(ACTIVE_TOOLS)
            backup.rename(ACTIVE_TOOLS)
        if old_lock is None:
            LOCK.unlink(missing_ok=True)
        else:
            LOCK.write_bytes(old_lock)
        if old_source is None:
            SOURCE_INFO.unlink(missing_ok=True)
        else:
            SOURCE_INFO.write_bytes(old_source)
        run("git", "checkout", "--detach", old_commit, cwd=UPSTREAM)
        raise
    finally:
        if staged is not None:
            shutil.rmtree(staged.parent, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="commit, tag, or remote ref; defaults to dependency config")
    parser.add_argument("--check", action="store_true", help="verify without fetching or changing files")
    parser.add_argument("--force", action="store_true", help="replace locally changed synchronized outputs")
    args = parser.parse_args()
    try:
        return check() if args.check else update(args.ref, args.force)
    except (OSError, RuntimeError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"stable-diffusion.cpp update failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
