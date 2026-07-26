#!/usr/bin/env python3
"""Structural, idempotent GGML ops-bridge transformation.

The transformer locates C/C++ functions by name and balanced scopes instead of
matching complete source lines. Every edit requires one unambiguous structural
target and every generated block carries a stable marker.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


MARKER = "@GGML_FORGE_BRIDGE:"


class BridgeConflict(RuntimeError):
    pass


def _code_mask(text: str) -> str:
    out = list(text)
    i = 0
    state = "code"
    quote = ""
    while i < len(text):
        c = text[i]
        n = text[i + 1] if i + 1 < len(text) else ""
        if state == "code":
            if c == "/" and n == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "line"
                continue
            if c == "/" and n == "*":
                out[i] = out[i + 1] = " "
                i += 2
                state = "block"
                continue
            if c in ('"', "'"):
                quote = c
                out[i] = " "
                i += 1
                state = "string"
                continue
        elif state == "line":
            if c == "\n":
                state = "code"
            else:
                out[i] = " "
        elif state == "block":
            if c == "*" and n == "/":
                out[i] = out[i + 1] = " "
                i += 2
                state = "code"
                continue
            if c != "\n":
                out[i] = " "
        else:
            out[i] = " "
            if c == "\\" and i + 1 < len(text):
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            if c == quote:
                state = "code"
        i += 1
    return "".join(out)


def _matching(mask: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    for i in range(start, len(mask)):
        if mask[i] == opening:
            depth += 1
        elif mask[i] == closing:
            depth -= 1
            if depth == 0:
                return i
    raise BridgeConflict(f"unbalanced {opening}{closing} scope at byte {start}")


@dataclass
class FunctionScope:
    signature_start: int
    brace: int
    end: int


class SourceEditor:
    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8")

    def save(self) -> None:
        self.path.write_text(self.text, encoding="utf-8", newline="\n")

    def _function(self, name: str) -> FunctionScope:
        mask = _code_mask(self.text)
        candidates: list[FunctionScope] = []
        for match in re.finditer(rf"\b{re.escape(name)}\s*\(", mask):
            open_paren = mask.find("(", match.start())
            close_paren = _matching(mask, open_paren, "(", ")")
            brace = mask.find("{", close_paren + 1)
            semicolon = mask.find(";", close_paren + 1)
            if brace < 0 or (semicolon >= 0 and semicolon < brace):
                continue
            line_start = self.text.rfind("\n", 0, match.start()) + 1
            # Include a preceding template/static return-type line only when the
            # function name is on the same source line, as GGML functions are.
            end = _matching(mask, brace, "{", "}")
            candidates.append(FunctionScope(line_start, brace, end))
        if len(candidates) != 1:
            raise BridgeConflict(f"{self.path}: expected one definition of {name}, found {len(candidates)}")
        return candidates[0]

    def insert_after_include(self, include_pattern: str, marker: str, lines: str) -> None:
        if marker in self.text:
            return
        matches = list(re.finditer(include_pattern, self.text, re.MULTILINE))
        if len(matches) != 1:
            raise BridgeConflict(f"{self.path}: include anchor matched {len(matches)} times")
        end = self.text.find("\n", matches[0].end())
        end = len(self.text) if end < 0 else end + 1
        self.text = self.text[:end] + lines.rstrip() + "\n" + self.text[end:]

    def insert_function_entry(self, function: str, marker: str, code: str) -> None:
        if marker in self.text:
            return
        scope = self._function(function)
        self.text = self.text[:scope.brace + 1] + "\n" + code.rstrip() + self.text[scope.brace + 1:]

    def insert_before_call(self, function: str, callee: str, marker: str, code: str) -> None:
        if marker in self.text:
            return
        scope = self._function(function)
        body = self.text[scope.brace + 1:scope.end]
        matches = list(re.finditer(rf"\b{re.escape(callee)}\s*\(", _code_mask(body)))
        if len(matches) != 1:
            raise BridgeConflict(f"{self.path}: {callee} in {function} matched {len(matches)} times")
        absolute = scope.brace + 1 + matches[0].start()
        line_start = self.text.rfind("\n", 0, absolute) + 1
        self.text = self.text[:line_start] + code.rstrip() + "\n" + self.text[line_start:]

    def replace_signature(self, function: str, marker: str, signature: str) -> None:
        if marker in self.text:
            return
        scope = self._function(function)
        self.text = self.text[:scope.signature_start] + signature.rstrip() + " {" + self.text[scope.brace + 1:]

    def insert_before_function_end(self, function: str, marker: str, code: str) -> None:
        if marker in self.text:
            return
        scope = self._function(function)
        self.text = self.text[:scope.end] + code.rstrip() + "\n" + self.text[scope.end:]

    def replace_call_statements(self, function: str, callee: str, marker: str, replacement: str) -> None:
        if marker in self.text:
            return
        scope = self._function(function)
        body = self.text[scope.brace + 1:scope.end]
        pattern = re.compile(rf"^(?P<indent>[ \t]*).*\b{re.escape(callee)}\s*\([^;]*\);[ \t]*$", re.MULTILINE)
        matches = list(pattern.finditer(body))
        if not matches:
            raise BridgeConflict(f"{self.path}: no {callee} call statements in {function}")
        updated = pattern.sub(lambda m: m.group("indent") + replacement, body)
        self.text = self.text[:scope.brace + 1] + updated + self.text[scope.end:]


CPU_DISPATCH = f"""
    // {MARKER} cpu_graph_compute_dispatch
    // Forge extension nodes execute in graph order inside the single compute
    // pass below (cpu_thread_ext_dispatch in ggml-cpu.c); the plan just
    // carries the backend handle the kernels need.
    plan.forge_ext_backend = backend;
"""

CPLAN_EXT_BACKEND = f"""
        // {MARKER} cplan_ext_backend
        // Backend handle for in-pass dispatch of Forge extension nodes
        // (op >= GGML_OP_EXT_BASE). NULL (the ggml_graph_plan default) keeps
        // extension nodes as no-ops, matching stock ggml behavior.
        void * forge_ext_backend;
"""

CPU_THREAD_DISPATCH = f"""
        // {MARKER} cpu_thread_ext_dispatch
        // Forge extension nodes run inline inside this single graph pass, in
        // graph order: thread 0 executes the registered kernel while every
        // thread meets at the same per-node barrier, which also publishes the
        // result before any thread starts the next node.
        if ((int)node->op >= GGML_OP_EXT_BASE) {{
            if (params.ith == 0 && g_ggml_cpu_op_vtable[node->op] != NULL &&
                cplan->forge_ext_backend != NULL) {{
                g_ggml_cpu_op_vtable[node->op]((ggml_backend_t) cplan->forge_ext_backend, node);
            }}
            if (state->ith == 0 && cplan->abort_callback &&
                    cplan->abort_callback(cplan->abort_callback_data)) {{
                atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
                tp->ec    = GGML_STATUS_ABORTED;
            }}
            if (node_n + 1 < cgraph->n_nodes) {{
                ggml_barrier(state->threadpool);
            }}
            continue;
        }}
"""


def _inject_enum(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "GGML_OP_EXT_RESERVED_MAX" in text:
        return
    pattern = re.compile(r"^(?P<i>\s*)GGML_OP_COUNT,\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise BridgeConflict(f"{path}: GGML_OP_COUNT matched {len(matches)} times")
    m = matches[0]
    insertion = (
        m.group(0) + "\n\n" + m.group("i") + f"// {MARKER} reserve_ext_op_range\n" +
        m.group("i") + "GGML_OP_EXT_RESERVED_MAX = 4095,"
    )
    path.write_text(text[:m.start()] + insertion + text[m.end():], encoding="utf-8", newline="\n")


def _export_graph_view(root: Path) -> None:
    for relative in ("src/ggml.c", "src/ggml-impl.h"):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if f"{MARKER} export_graph_view" in text:
            continue
        pattern = re.compile(r"^(?P<i>\s*)(?!GGML_API)(struct\s+ggml_cgraph\s+ggml_graph_view\s*\()", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise BridgeConflict(f"{path}: graph_view declaration/definition matched {len(matches)} times")
        m = matches[0]
        replacement = m.group("i") + f"// {MARKER} export_graph_view\n" + m.group("i") + "GGML_API " + m.group(2)
        path.write_text(text[:m.start()] + replacement + text[m.end():], encoding="utf-8", newline="\n")


def apply_bridge(ggml_root: Path, bridge_assets: Path) -> None:
    ggml_root = ggml_root.resolve()
    assets = {
        "ggml-ops-ext-bridge.h": ggml_root / "src/ggml-ops-ext-bridge.h",
        "ggml-ops-ext-bridge.cpp": ggml_root / "src/ggml-ops-ext-bridge.cpp",
        "ggml-ops-ext-bridge-cuda.cu": ggml_root / "src/ggml-cuda/ggml-ops-ext-bridge-cuda.cu",
        "ggml-ops-ext-bridge-sycl.cpp": ggml_root / "src/ggml-sycl/ggml-ops-ext-bridge-sycl.cpp",
    }
    for source, target in assets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bridge_assets / source, target)

    _inject_enum(ggml_root / "include/ggml.h")
    _export_graph_view(ggml_root)

    cpu_h = SourceEditor(ggml_root / "include/ggml-cpu.h")
    cpu_h.insert_after_include(r'^\s*bool use_ref;\s*$', "cplan_ext_backend", CPLAN_EXT_BACKEND)
    cpu_h.save()

    base = SourceEditor(ggml_root / "src/ggml.cpp")
    base.insert_after_include(
        r'^\s*#\s*include\s+"ggml-impl\.h"\s*$',
        "ggml-ops-ext-bridge.cpp",
        f'// {MARKER} bridge_impl\n#include "ggml-ops-ext-bridge.cpp"',
    )
    base.save()

    cpu_c = SourceEditor(ggml_root / "src/ggml-cpu/ggml-cpu.c")
    cpu_c.insert_after_include(r'^\s*#\s*include\s+"ggml-backend-impl\.h"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} cpu_c_include\n#include "../ggml-ops-ext-bridge.h"')
    cpu_c.insert_function_entry("ggml_get_n_tasks", "cpu_n_tasks_ext", f"""
    // {MARKER} cpu_n_tasks_ext
    if ((int)node->op >= GGML_OP_EXT_BASE) return 1;
""")
    cpu_c.insert_before_call("ggml_graph_compute_thread", "ggml_cpu_try_fuse_ops",
                             "cpu_thread_ext_dispatch", CPU_THREAD_DISPATCH)
    cpu_c.save()

    cpu = SourceEditor(ggml_root / "src/ggml-cpu/ggml-cpu.cpp")
    cpu.insert_after_include(r'^\s*#\s*include\s+"ggml-impl\.h"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} cpu_include\n#include "../ggml-ops-ext-bridge.h"')
    cpu.insert_function_entry("ggml_backend_cpu_device_supports_op", "cpu_supports_op", f"""
    // {MARKER} cpu_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    # The dispatch needs `plan` in scope and must run right before the final
    # full-graph compute, so it anchors on that call rather than the entry.
    cpu.insert_before_call("ggml_backend_cpu_graph_compute", "ggml_graph_compute", "cpu_graph_compute_dispatch", CPU_DISPATCH)
    cpu.insert_after_include(
        r'^\s*cpu_plan->cplan = ggml_graph_plan\(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool\);\s*$',
        "cpu_graph_plan_ext_backend",
        f"    // {MARKER} cpu_graph_plan_ext_backend\n    cpu_plan->cplan.forge_ext_backend = backend;",
    )
    cpu.save()

    cuda = SourceEditor(ggml_root / "src/ggml-cuda/ggml-cuda.cu")
    cuda.insert_after_include(r'^\s*#\s*include\s+"ggml-cuda/common\.cuh"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} cuda_include\n#include "../ggml-ops-ext-bridge.h"')
    cuda.insert_function_entry("ggml_backend_cuda_device_supports_op", "cuda_supports_op", f"""
    // {MARKER} cuda_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    cuda.insert_before_call("ggml_cuda_graph_evaluate_and_capture", "ggml_cuda_compute_forward", "cuda_graph_compute_dispatch", f"""
                // {MARKER} cuda_graph_compute_dispatch
                if (node->op >= GGML_OP_EXT_BASE && g_ggml_cuda_op_vtable[node->op]) {{
                    g_ggml_cuda_op_vtable[node->op](backend, node);
                    continue;
                }}
""")
    cuda.save()

    sycl = SourceEditor(ggml_root / "src/ggml-sycl/ggml-sycl.cpp")
    sycl.insert_after_include(r'^\s*#\s*include\s+"ggml-sycl/common\.hpp"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} sycl_include\n#include "../ggml-ops-ext-bridge.h"')
    sycl.insert_function_entry("do_ggml_backend_sycl_device_supports_op", "sycl_supports_op", f"""
    // {MARKER} sycl_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    sycl.insert_before_call("ggml_backend_sycl_graph_compute_impl", "ggml_sycl_compute_forward", "sycl_graph_compute_dispatch", f"""
        // {MARKER} sycl_graph_compute_dispatch
        if (node->op >= GGML_OP_EXT_BASE && g_ggml_sycl_op_vtable[node->op]) {{
            g_ggml_sycl_op_vtable[node->op](backend, node);
            continue;
        }}
""")
    sycl.save()


def verify_bridge(ggml_root: Path) -> None:
    requirements = {
        "include/ggml.h": ["GGML_OP_EXT_RESERVED_MAX"],
        "include/ggml-cpu.h": ["cplan_ext_backend"],
        "src/ggml.cpp": ["ggml-ops-ext-bridge.cpp"],
        "src/ggml-cpu/ggml-cpu.c": ["cpu_n_tasks_ext", "cpu_thread_ext_dispatch"],
        "src/ggml-cpu/ggml-cpu.cpp": ["cpu_graph_compute_dispatch", "cpu_supports_op",
                                      "cpu_graph_plan_ext_backend"],
        "src/ggml-cuda/ggml-cuda.cu": ["cuda_graph_compute_dispatch", "cuda_supports_op"],
        "src/ggml-sycl/ggml-sycl.cpp": ["sycl_graph_compute_dispatch", "sycl_supports_op"],
    }
    for relative, markers in requirements.items():
        text = (ggml_root / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise BridgeConflict(f"{relative}: missing bridge contract {marker}")
