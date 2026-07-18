#!/usr/bin/env python3
"""Mirror llama.cpp's Python conversion toolchain into the LLM provider tools."""

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
UPSTREAM = ROOT / "src/categories/llm/providers/llama_cpp/llama.cpp"
PROVIDER = ROOT / "scripts/conversion/categories/llm/providers/llama_cpp"
ACTIVE_TOOLS = PROVIDER / "tools"
SOURCE_INFO = PROVIDER / "source.json"
LOCK = Path(__file__).with_name("lock.json")
TOOL_DIRECTORIES = ("conversion", "gguf-py", "requirements")
TOOL_FILES = ("requirements.txt",)
REQUIRED_SCRIPTS = {
    "convert_hf_to_gguf.py",
    "convert_llama_ggml_to_gguf.py",
    "convert_lora_to_gguf.py",
}


def run(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, encoding="utf-8").strip()


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def conversion_scripts() -> list[Path]:
    return sorted(UPSTREAM.glob("convert_*.py"), key=lambda path: path.name)


def ensure_layout() -> None:
    if not UPSTREAM.is_dir():
        raise RuntimeError(f"llama.cpp submodule is missing: {UPSTREAM}")
    missing = [name for name in TOOL_DIRECTORIES + TOOL_FILES if not (UPSTREAM / name).exists()]
    if missing:
        raise RuntimeError(f"llama.cpp conversion dependencies are missing: {', '.join(missing)}")
    names = {path.name for path in conversion_scripts()}
    missing_scripts = sorted(REQUIRED_SCRIPTS - names)
    if missing_scripts:
        raise RuntimeError(f"required llama.cpp converters are missing: {', '.join(missing_scripts)}")


def validate(staged: Path) -> None:
    for path in staged.rglob("*.py"):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeError) as error:
            raise RuntimeError(f"invalid copied Python source {path.relative_to(staged)}: {error}") from error
    for script in REQUIRED_SCRIPTS:
        if not (staged / script).is_file():
            raise RuntimeError(f"staged converter is missing: {script}")


def stage() -> tuple[Path, dict[str, object]]:
    ensure_layout()
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="llama-conversion-stage-", dir=ROOT / "tmp"))
    staged = temporary / "tools"
    staged.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for directory in TOOL_DIRECTORIES:
        shutil.copytree(UPSTREAM / directory, staged / directory, ignore=ignore)
    for source in (*conversion_scripts(), *(UPSTREAM / name for name in TOOL_FILES)):
        shutil.copy2(source, staged / source.name)
    validate(staged)

    commit = run("git", "rev-parse", "HEAD", cwd=UPSTREAM)
    info: dict[str, object] = {
        "schema": 1,
        "llama_cpp_commit": commit,
        "scripts": [path.name for path in conversion_scripts()],
        "tree_sha256": digest_tree(staged),
    }
    return staged, info


def compare_trees(left: Path, right: Path) -> list[str]:
    left_files = {path.relative_to(left).as_posix(): path for path in left.rglob("*") if path.is_file()}
    right_files = {path.relative_to(right).as_posix(): path for path in right.rglob("*") if path.is_file()}
    differences: list[str] = []
    for relative in sorted(left_files.keys() | right_files.keys()):
        if relative not in left_files:
            differences.append(f"unexpected copied file: {relative}")
        elif relative not in right_files:
            differences.append(f"missing copied file: {relative}")
        elif left_files[relative].read_bytes() != right_files[relative].read_bytes():
            differences.append(f"content differs: {relative}")
    return differences


def write_metadata(info: dict[str, object]) -> None:
    PROVIDER.mkdir(parents=True, exist_ok=True)
    SOURCE_INFO.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    lock.update({
        "llama_cpp_commit": info["llama_cpp_commit"],
        "conversion_scripts": info["scripts"],
        "conversion_tools_tree_sha256": info["tree_sha256"],
    })
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def install(staged: Path, info: dict[str, object], force: bool) -> None:
    relative = ACTIVE_TOOLS.relative_to(ROOT).as_posix()
    status = run("git", "status", "--porcelain", "--", relative)
    if status and not force:
        raise RuntimeError("copied llama.cpp conversion tools have local changes; use --force to replace them")

    PROVIDER.mkdir(parents=True, exist_ok=True)
    backup = staged.parent / "tools.previous"
    try:
        if ACTIVE_TOOLS.exists():
            ACTIVE_TOOLS.rename(backup)
        staged.rename(ACTIVE_TOOLS)
        write_metadata(info)
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
    mode.add_argument("--apply", action="store_true", help="replace the copied conversion toolchain")
    mode.add_argument("--check", action="store_true", help="verify the copied toolchain")
    mode.add_argument("--stage-only", action="store_true", help="print a validated temporary toolchain path")
    parser.add_argument("--force", action="store_true", help="replace locally changed copied tools")
    args = parser.parse_args()

    try:
        staged, info = stage()
        if args.stage_only:
            print(staged)
            return 0
        if args.check:
            differences = compare_trees(staged, ACTIVE_TOOLS) if ACTIVE_TOOLS.exists() else ["tools directory is missing"]
            shutil.rmtree(staged.parent, ignore_errors=True)
            current_info = json.loads(SOURCE_INFO.read_text(encoding="utf-8")) if SOURCE_INFO.exists() else {}
            if differences:
                print("llama.cpp conversion tools are out of sync:", file=sys.stderr)
                for difference in differences[:100]:
                    print(f"  - {difference}", file=sys.stderr)
                return 1
            if current_info != info:
                print("conversion tools match but source metadata is stale", file=sys.stderr)
                return 1
            print("llama.cpp conversion tools are synchronized")
            return 0
        install(staged, info, args.force)
        print(f"Copied llama.cpp conversion tools from {info['llama_cpp_commit']}")
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"conversion sync failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
