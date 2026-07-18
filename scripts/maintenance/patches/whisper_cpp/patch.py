"""Validate and apply Forge's whisper.cpp integration patch set."""

from __future__ import annotations

from pathlib import Path


PATCH_REVISION = 0


class PatchConflict(RuntimeError):
    pass


def apply(upstream: Path) -> None:
    """Apply source patches in place. Revision zero currently needs no edits."""
    verify(upstream)


def verify(upstream: Path) -> None:
    cmake = upstream / "CMakeLists.txt"
    if not cmake.is_file():
        raise PatchConflict(f"missing whisper.cpp CMakeLists.txt: {cmake}")
    text = cmake.read_text(encoding="utf-8")
    required = (
        "if (NOT TARGET ggml)",
        "add_subdirectory(ggml)",
        "otherwise assume ggml is added by a parent CMakeLists.txt",
    )
    missing = [contract for contract in required if contract not in text]
    if missing:
        raise PatchConflict(
            "whisper.cpp no longer guarantees parent GGML reuse: " + ", ".join(missing)
        )
