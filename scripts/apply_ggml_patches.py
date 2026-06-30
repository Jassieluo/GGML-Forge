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
    # ── Export ggml_graph_view from ggml.c ──
    os.path.join(GGML_ROOT, "ggml.c"): [
        {
            "name": "export_graph_view",
            "anchor": "struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {",
            "replace_anchor": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: export_graph_view",
                "GGML_API struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {",
            ],
            "marker": "GGML_API struct ggml_cgraph ggml_graph_view",
        },
    ],
    # ── Add GGML_API declaration to ggml-impl.h ──
    os.path.join(GGML_ROOT, "ggml-impl.h"): [
        {
            "name": "decl_graph_view",
            "anchor": "struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph, int i0, int i1);",
            "replace_anchor": True,
            "lines": [
                "// @GGML_BRIDGE_INJECT: decl_graph_view",
                "GGML_API struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph, int i0, int i1);",
            ],
            "marker": "GGML_API struct ggml_cgraph ggml_graph_view",
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
                "    struct ggml_backend_cpu_context * overall_ctx = (struct ggml_backend_cpu_context *)backend->context;",
                "    std::vector<enum ggml_op> overall_ops(cgraph->n_nodes);",
                "    for (int i = 0; i < cgraph->n_nodes; ++i) {",
                "        overall_ops[i] = cgraph->nodes[i]->op;",
                "        if (cgraph->nodes[i]->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {",
                "            cgraph->nodes[i]->op = GGML_OP_NONE;",
                "        }",
                "    }",
                "    struct ggml_cplan overall_plan = ggml_graph_plan(cgraph, overall_ctx->n_threads, overall_ctx->threadpool);",
                "    if (overall_ctx->work_size < overall_plan.work_size) {",
                "        delete[] overall_ctx->work_data;",
                "        overall_ctx->work_data = new uint8_t[overall_plan.work_size];",
                "        if (overall_ctx->work_data == NULL) {",
                "            overall_ctx->work_size = 0;",
                "            for (int i = 0; i < cgraph->n_nodes; ++i) {",
                "                cgraph->nodes[i]->op = overall_ops[i];",
                "            }",
                "            return GGML_STATUS_ALLOC_FAILED;",
                "        }",
                "        overall_ctx->work_size = overall_plan.work_size;",
                "    }",
                "    for (int i = 0; i < cgraph->n_nodes; ++i) {",
                "        cgraph->nodes[i]->op = overall_ops[i];",
                "    }",
                "    int last_computed_idx = 0;",
                "    int n_nodes = cgraph->n_nodes;",
                "    for (int i = 0; i < n_nodes; ++i) {",
                "        struct ggml_tensor * node = cgraph->nodes[i];",
                "        if (node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {",
                "            if (i > last_computed_idx) {",
                "                struct ggml_cgraph sub_graph = ggml_graph_view(cgraph, last_computed_idx, i);",
                "                struct ggml_cplan sub_plan = ggml_graph_plan(&sub_graph, overall_ctx->n_threads, overall_ctx->threadpool);",
                "                sub_plan.work_data = (uint8_t *)overall_ctx->work_data;",
                "                sub_plan.abort_callback      = overall_ctx->abort_callback;",
                "                sub_plan.abort_callback_data = overall_ctx->abort_callback_data;",
                "                sub_plan.use_ref             = overall_ctx->use_ref;",
                "                enum ggml_status status = ggml_graph_compute(&sub_graph, &sub_plan);",
                "                if (status != GGML_STATUS_SUCCESS) {",
                "                    return status;",
                "                }",
                "            }",
                "            bool computed_by_hook = false;",
                "            if (g_ggml_bridge_hook(backend, node)) {",
                "                computed_by_hook = true;",
                "            }",
                "            if (computed_by_hook) {",
                "                last_computed_idx = i + 1;",
                "            }",
                "        }",
                "    }",
                "    if (last_computed_idx < n_nodes) {",
                "        struct ggml_cgraph sub_graph = ggml_graph_view(cgraph, last_computed_idx, n_nodes);",
                "        struct ggml_cplan sub_plan = ggml_graph_plan(&sub_graph, overall_ctx->n_threads, overall_ctx->threadpool);",
                "        sub_plan.work_data = (uint8_t *)overall_ctx->work_data;",
                "        sub_plan.abort_callback      = overall_ctx->abort_callback;",
                "        sub_plan.abort_callback_data = overall_ctx->abort_callback_data;",
                "        sub_plan.use_ref             = overall_ctx->use_ref;",
                "        enum ggml_status status = ggml_graph_compute(&sub_graph, &sub_plan);",
                "        if (status != GGML_STATUS_SUCCESS) {",
                "            return status;",
                "        }",
                "    }",
                "    return GGML_STATUS_SUCCESS;",
            ],
            "marker": "ggml_graph_view",
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
        {
            "name": "cuda_graph_evaluate_def",
            "anchor": "static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx,",
            "replace_anchor": True,
            "lines": [
                "static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_t backend, ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {"
            ],
            "marker": "ggml_backend_t backend, ggml_backend_cuda_context",
        },
        {
            "name": "cuda_graph_evaluate_call",
            "anchor": "    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);",
            "replace_anchor": True,
            "lines": [
                "    ggml_cuda_graph_evaluate_and_capture(backend, cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);"
            ],
            "marker": "ggml_cuda_graph_evaluate_and_capture(backend,",
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
        {
            "name": "sycl_graph_evaluate_def",
            "anchor": "static void ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {",
            "replace_anchor": True,
            "lines": [
                "static void ggml_backend_sycl_graph_compute_impl(ggml_backend_t backend, ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {"
            ],
            "marker": "ggml_backend_t backend, ggml_backend_sycl_context",
        },
        {
            "name": "sycl_graph_evaluate_call",
            "anchor": "ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);",
            "replace_all": True,
            "lines": [
                "ggml_backend_sycl_graph_compute_impl(backend, sycl_ctx, cgraph);"
            ],
            "marker": "ggml_backend_sycl_graph_compute_impl(backend,",
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

        if inj.get("replace_all"):
            new_lines = []
            for line in lines:
                if anchor in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(indent + inj["lines"][0] + "\n")
                    modified = True
                else:
                    new_lines.append(line)
            lines = new_lines
            info(f"PATCH {target_rel} :: {name} (replace_all)")
        else:
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
