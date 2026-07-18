"""Validate and apply Forge's stable-diffusion.cpp integration patch set."""

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
        raise PatchConflict(f"missing stable-diffusion.cpp CMakeLists.txt: {cmake}")
    text = cmake.read_text(encoding="utf-8")
    required = (
        "if (NOT TARGET ggml)",
        "add_subdirectory(ggml)",
        "set(SD_LIB stable-diffusion)",
    )
    missing = [contract for contract in required if contract not in text]
    if missing:
        raise PatchConflict(
            "stable-diffusion.cpp no longer supports the Forge parent-GGML bridge: "
            + ", ".join(missing)
        )

    private_headers = ("src/ggml-impl.h",)
    missing_headers = [path for path in private_headers if not (upstream / "ggml" / path).is_file()]
    # An uninitialized bundled GGML is intentional. Validate its tree entry in
    # the updater; only validate headers when the nested submodule is present.
    if (upstream / "ggml" / ".git").exists() and missing_headers:
        raise PatchConflict(
            "stable-diffusion.cpp bundled GGML private-header contract changed: "
            + ", ".join(missing_headers)
        )
