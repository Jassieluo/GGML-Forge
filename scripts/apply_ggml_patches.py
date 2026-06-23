#!/usr/bin/env python3
"""
apply_ggml_patches.py — GPT-SoVITS.cpp ggml Bridge 注入脚本

Phases:
  COPY    — 拷贝 bridge 源文件到 ggml/src/ 对应目录
  INJECT  — 在 ggml 后端文件中注入 include + supports_op + graph_compute 调度
  VERIFY  — 验证所有锚点和标记

用法:
  python scripts/apply_ggml_patches.py            # 正常应用
  python scripts/apply_ggml_patches.py --revert   # 还原到 git-clean 状态
"""

import os
import sys
import shutil
import subprocess
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GGML_ROOT = os.path.join(PROJECT_ROOT, "ggml", "src")
BRIDGE_SRC = os.path.join(os.path.dirname(__file__), "ggml-bridge")

# ────────────────────────────────────────────────────────────────
# Injection rules
# ────────────────────────────────────────────────────────────────

# Format: list of (target_relative_path, injections)
# Each injection: {
#   "name": str, "anchor": str (exact line match),
#   "insert_after": True, "lines": [str, ...],
#   "marker": str (for idempotency check)
# }
# Target paths are relative to GGML_ROOT.
# For ggml/src/CMakeLists.txt, target is relative to ggml/ (parent of GGML_ROOT).

INJECTIONS = {
    # ── Inject bridge implementation into ggml.cpp (ggml-base target) ──
    # ggml-base.dll is the common dependency of all backend DLLs.
    os.path.join(GGML_ROOT, "ggml.cpp"): [
        {
            "name": "bridge_impl",
            "anchor": '#include "ggml-impl.h"',
            "insert_after": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: bridge_impl",
                '#include "ggml-ops-ext-bridge.cpp"',
            ],
            "marker": "ggml-ops-ext-bridge.cpp",
        },
    ],
    # ── ggml-cpu ──
    os.path.join(GGML_ROOT, "ggml-cpu", "ggml-cpu.cpp"): [
        {
            "name": "cpu_include_bridge",
            "anchor": '#include "ggml-impl.h"',
            "insert_after": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: cpu_include_bridge",
                '#include "../ggml-ops-ext-bridge.h"',
            ],
            "marker": 'ggml-ops-ext-bridge.h',
        },
        {
            "name": "cpu_supports_op",
            "anchor": "static bool ggml_backend_cpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {",
            "insert_after": True,
            "lines": [
                "    // @GGML_BRIDGE_INJECT: cpu_supports_op",
                "    if (op->op >= GGML_OP_EXT_BASE) return true;",
            ],
            "marker": "GGML_OP_EXT_BASE",
        },
        {
            "name": "cpu_graph_compute_dispatch",
            "anchor": "static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {",
            "insert_after": True,
            "lines": [
                "    // @GGML_BRIDGE_INJECT: cpu_graph_compute_dispatch",
                "    for (int _i = 0; _i < cgraph->n_nodes; ++_i) {",
                "        struct ggml_tensor * _node = cgraph->nodes[_i];",
                "        if (_node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {",
                "            if (g_ggml_bridge_hook(backend, _node)) {",
                "                _node->op = GGML_OP_NONE;",
                "            }",
                "        }",
                "    }",
            ],
            "marker": "g_ggml_bridge_hook",
        },
    ],
    # ── ggml-cuda ──
    os.path.join(GGML_ROOT, "ggml-cuda", "ggml-cuda.cu"): [
        {
            "name": "cuda_include_bridge",
            "anchor": '#include "ggml-cuda/common.cuh"',
            "insert_after": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: cuda_include_bridge",
                '#include "../ggml-ops-ext-bridge.h"',
            ],
            "marker": 'ggml-ops-ext-bridge.h',
        },
        {
            "name": "cuda_supports_op",
            "anchor": "static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {",
            "insert_after": True,
            "lines": [
                "    // @GGML_BRIDGE_INJECT: cuda_supports_op",
                "    if (op->op >= GGML_OP_EXT_BASE) return true;",
            ],
            "marker": "GGML_OP_EXT_BASE",
        },
        {
            "name": "cuda_graph_compute_dispatch",
            "anchor": "ggml_cuda_compute_forward(*cuda_ctx, node);",
            "insert_after": False,
            "lines": [
                "                // @GGML_BRIDGE_INJECT: cuda_graph_compute_dispatch",
                "                if (node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {",
                "                    if (g_ggml_bridge_hook(backend, node)) {",
                "                        continue;",
                "                    }",
                "                }",
            ],
            "marker": "g_ggml_bridge_hook",
        },
    ],
    # ── ggml-sycl ──
    os.path.join(GGML_ROOT, "ggml-sycl", "ggml-sycl.cpp"): [
        {
            "name": "sycl_include_bridge",
            "anchor": '#include "ggml-sycl/common.hpp"',
            "insert_after": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: sycl_include_bridge",
                '#include "../ggml-ops-ext-bridge.h"',
            ],
            "marker": 'ggml-ops-ext-bridge.h',
        },
        {
            "name": "sycl_supports_op",
            "anchor": "static bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {",
            "insert_after": True,
            "lines": [
                "    // @GGML_BRIDGE_INJECT: sycl_supports_op",
                "    if (op->op >= GGML_OP_EXT_BASE) return true;",
            ],
            "marker": "GGML_OP_EXT_BASE",
        },
        {
            "name": "sycl_graph_compute_dispatch",
            "anchor": "ggml_sycl_compute_forward(*sycl_ctx, node);",
            "insert_after": False,
            "lines": [
                "        // @GGML_BRIDGE_INJECT: sycl_graph_compute_dispatch",
                "        if (node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {",
                "            if (g_ggml_bridge_hook(backend, node)) {",
                "                continue;",
                "            }",
                "        }",
            ],
            "marker": "g_ggml_bridge_hook",
        },
    ],
}

# ────────────────────────────────────────────────────────────────
# Copy rules: (source_relative_to_bridge_src, dest_relative_to_project_root)
# ────────────────────────────────────────────────────────────────

COPY_RULES = [
    ("ggml-ops-ext-bridge.h",     "ggml/src/ggml-ops-ext-bridge.h"),
    ("ggml-ops-ext-bridge.cpp",   "ggml/src/ggml-ops-ext-bridge.cpp"),  # included inline by ggml-backend-reg.cpp
    ("ggml-ops-ext-bridge-cuda.cu", "ggml/src/ggml-cuda/ggml-ops-ext-bridge-cuda.cu"),
    ("ggml-ops-ext-bridge-sycl.cpp", "ggml/src/ggml-sycl/ggml-ops-ext-bridge-sycl.cpp"),
]

# Marker prefix for injected lines (used by --revert)
INJECT_MARKER = "@GGML_BRIDGE_INJECT:"

# ────────────────────────────────────────────────────────────────
# Implementation
# ────────────────────────────────────────────────────────────────

def error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print(f"[WARN]  {msg}")


def info(msg):
    print(f"[INFO]  {msg}")


def phase(header):
    print(f"\n{'='*60}")
    print(f"  {header}")
    print(f"{'='*60}")


def copy_bridge_files():
    """Phase: COPY"""
    for src_rel, dst_rel in COPY_RULES:
        src = os.path.join(BRIDGE_SRC, src_rel)
        dst = os.path.join(PROJECT_ROOT, dst_rel)

        if not os.path.exists(src):
            error(f"Bridge source file missing: {src}")

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        info(f"COPY  {src_rel}  →  {dst_rel}")


def inject_file(target_rel, injections):
    """Phase: INJECT - process one file"""
    target = os.path.join(PROJECT_ROOT, target_rel)

    if not os.path.exists(target):
        error(f"Target file not found: {target_rel}\n"
              f"       (file: {target})")

    with open(target, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    modified = False

    for inj in injections:
        name = inj["name"]
        anchor = inj["anchor"]
        marker = inj["marker"]

        # Idempotency check
        if any(marker in line for line in lines):
            info(f"SKIP  {target_rel} :: {name} (marker already present)")
            continue

        # Find anchor line
        anchor_idx = None
        for i, line in enumerate(lines):
            if anchor in line:
                anchor_idx = i
                break

        if anchor_idx is None:
            error(
                f"Anchor not found in {target_rel}\n"
                f"  Injection: {name}\n"
                f"  Anchor:    {repr(anchor)}\n"
                f"  The ggml upstream code may have changed. "
                f"Update the injection rules in this script."
            )

        if inj.get("replace_anchor"):
            # Replace the anchor line with the new lines
            insert_lines = [line + "\n" for line in inj["lines"]]
            lines = lines[:anchor_idx] + insert_lines + lines[anchor_idx+1:]
        else:
            insert_idx = anchor_idx + 1 if inj.get("insert_after", True) else anchor_idx
            insert_lines = [line + "\n" for line in inj["lines"]]
            lines = lines[:insert_idx] + insert_lines + lines[insert_idx:]
        modified = True
        info(f"PATCH {target_rel} :: {name}  (line {anchor_idx+1})")

    if modified:
        with open(target, "w", encoding="utf-8") as f:
            f.writelines(lines)


def inject_all():
    """Phase: INJECT"""
    for target_rel, injections in INJECTIONS.items():
        inject_file(target_rel, injections)


def verify_all():
    """Phase: VERIFY"""
    errors = []

    for target_rel, injections in INJECTIONS.items():
        target = os.path.join(PROJECT_ROOT, target_rel)

        if not os.path.exists(target):
            errors.append(f"Target file missing: {target_rel}")
            continue

        with open(target, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        for inj in injections:
            marker = inj["marker"]
            if marker not in content:
                errors.append(f"Marker '{marker}' not found in {target_rel}")

    if errors:
        error("VERIFY failed:\n  " + "\n  ".join(errors))

    info("VERIFY passed: all markers present")


def revert_all():
    """Phase: REVERT — use git checkout for reliability"""
    # 1. Remove copied bridge files
    for _, dst_rel in COPY_RULES:
        dst = os.path.join(PROJECT_ROOT, dst_rel)
        if os.path.exists(dst):
            os.remove(dst)
            info(f"REMOVE {dst_rel}")

    # 2. Use git to restore ggml files to clean state
    ggml_dir = os.path.join(PROJECT_ROOT, "ggml")
    if os.path.exists(os.path.join(ggml_dir, ".git")):
        result = subprocess.run(
            ["git", "-C", ggml_dir, "checkout", "--", "src/", "CMakeLists.txt"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            warn(f"git checkout may have failed: {result.stderr}")
        else:
            info("git checkout restored ggml to clean state")
    else:
        warn("ggml/.git not found; skipping git checkout")

    info("REVERT complete — ggml is now git-clean")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--revert":
        phase("REVERT: Removing all bridge injections")
        revert_all()
        return

    # Normal flow
    phase("COPY: Bridge source files → ggml/src/")
    copy_bridge_files()

    phase("INJECT: Insert include + supports_op + graph_compute dispatch")
    inject_all()

    phase("VERIFY: Check all markers present")
    verify_all()

    print("\n[DONE] ggml bridge patches applied successfully.")
    print("       Run with --revert to restore ggml to git-clean state.")


if __name__ == "__main__":
    main()
