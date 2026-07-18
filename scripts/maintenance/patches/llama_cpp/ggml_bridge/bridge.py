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
    struct ggml_backend_cpu_context * overall_ctx = (struct ggml_backend_cpu_context *) backend->context;
    std::vector<enum ggml_op> overall_ops(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {{
        overall_ops[i] = cgraph->nodes[i]->op;
        if (cgraph->nodes[i]->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) cgraph->nodes[i]->op = GGML_OP_NONE;
    }}
    struct ggml_cplan overall_plan = ggml_graph_plan(cgraph, overall_ctx->n_threads, overall_ctx->threadpool);
    if (overall_ctx->work_size < overall_plan.work_size) {{
        delete[] overall_ctx->work_data;
        overall_ctx->work_data = new uint8_t[overall_plan.work_size];
        if (overall_ctx->work_data == NULL) {{
            overall_ctx->work_size = 0;
            for (int i = 0; i < cgraph->n_nodes; ++i) cgraph->nodes[i]->op = overall_ops[i];
            return GGML_STATUS_ALLOC_FAILED;
        }}
        overall_ctx->work_size = overall_plan.work_size;
    }}
    for (int i = 0; i < cgraph->n_nodes; ++i) cgraph->nodes[i]->op = overall_ops[i];
    int last_computed_idx = 0;
    const int n_nodes = cgraph->n_nodes;
    for (int i = 0; i < n_nodes; ++i) {{
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op < GGML_OP_EXT_BASE || !g_ggml_bridge_hook) continue;
        if (i > last_computed_idx) {{
            struct ggml_cgraph sub_graph = ggml_graph_view(cgraph, last_computed_idx, i);
            struct ggml_cplan sub_plan = ggml_graph_plan(&sub_graph, overall_ctx->n_threads, overall_ctx->threadpool);
            sub_plan.work_data = (uint8_t *) overall_ctx->work_data;
            sub_plan.abort_callback = overall_ctx->abort_callback;
            sub_plan.abort_callback_data = overall_ctx->abort_callback_data;
            sub_plan.use_ref = overall_ctx->use_ref;
            enum ggml_status status = ggml_graph_compute(&sub_graph, &sub_plan);
            if (status != GGML_STATUS_SUCCESS) return status;
        }}
        const int ext_result = g_ggml_bridge_hook(backend, node);
        if (ext_result != GGML_OPS_EXT_SUCCESS) return GGML_STATUS_FAILED;
        last_computed_idx = i + 1;
    }}
    if (last_computed_idx < n_nodes) {{
        struct ggml_cgraph sub_graph = ggml_graph_view(cgraph, last_computed_idx, n_nodes);
        struct ggml_cplan sub_plan = ggml_graph_plan(&sub_graph, overall_ctx->n_threads, overall_ctx->threadpool);
        sub_plan.work_data = (uint8_t *) overall_ctx->work_data;
        sub_plan.abort_callback = overall_ctx->abort_callback;
        sub_plan.abort_callback_data = overall_ctx->abort_callback_data;
        sub_plan.use_ref = overall_ctx->use_ref;
        enum ggml_status status = ggml_graph_compute(&sub_graph, &sub_plan);
        if (status != GGML_STATUS_SUCCESS) return status;
    }}
    return GGML_STATUS_SUCCESS;
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

    base = SourceEditor(ggml_root / "src/ggml.cpp")
    base.insert_after_include(
        r'^\s*#\s*include\s+"ggml-impl\.h"\s*$',
        "ggml-ops-ext-bridge.cpp",
        f'// {MARKER} bridge_impl\n#include "ggml-ops-ext-bridge.cpp"',
    )
    base.save()

    cpu = SourceEditor(ggml_root / "src/ggml-cpu/ggml-cpu.cpp")
    cpu.insert_after_include(r'^\s*#\s*include\s+"ggml-impl\.h"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} cpu_include\n#include "../ggml-ops-ext-bridge.h"')
    cpu.insert_function_entry("ggml_backend_cpu_device_supports_op", "cpu_supports_op", f"""
    // {MARKER} cpu_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    cpu.insert_function_entry("ggml_backend_cpu_graph_compute", "cpu_graph_compute_dispatch", CPU_DISPATCH)
    cpu.save()

    cuda = SourceEditor(ggml_root / "src/ggml-cuda/ggml-cuda.cu")
    cuda.insert_after_include(r'^\s*#\s*include\s+"ggml-cuda/common\.cuh"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} cuda_include\n#include "../ggml-ops-ext-bridge.h"')
    cuda.insert_function_entry("ggml_backend_cuda_device_supports_op", "cuda_supports_op", f"""
    // {MARKER} cuda_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    cuda.replace_signature("ggml_cuda_graph_evaluate_and_capture", "ggml_backend_t backend, ggml_backend_cuda_context", f"""
// {MARKER} cuda_status_signature
static enum ggml_status ggml_cuda_graph_evaluate_and_capture(ggml_backend_t backend, ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key)
""")
    cuda.insert_before_call("ggml_cuda_graph_evaluate_and_capture", "ggml_cuda_compute_forward", "cuda_graph_compute_dispatch", f"""
                // {MARKER} cuda_graph_compute_dispatch
                if (node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {{
                    const int ext_result = g_ggml_bridge_hook(backend, node);
                    if (ext_result == GGML_OPS_EXT_SUCCESS) continue;
                    return GGML_STATUS_FAILED;
                }}
""")
    cuda.insert_before_function_end("ggml_cuda_graph_evaluate_and_capture", "cuda_status_return", f"""
    // {MARKER} cuda_status_return
    return GGML_STATUS_SUCCESS;
""")
    cuda.replace_call_statements("ggml_backend_cuda_graph_compute", "ggml_cuda_graph_evaluate_and_capture", "ggml_cuda_graph_evaluate_and_capture(backend", f"""// {MARKER} cuda_status_call
    return ggml_cuda_graph_evaluate_and_capture(backend, cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);""")
    cuda.save()

    sycl = SourceEditor(ggml_root / "src/ggml-sycl/ggml-sycl.cpp")
    sycl.insert_after_include(r'^\s*#\s*include\s+"ggml-sycl/common\.hpp"\s*$', "ggml-ops-ext-bridge.h", f'// {MARKER} sycl_include\n#include "../ggml-ops-ext-bridge.h"')
    sycl.insert_function_entry("do_ggml_backend_sycl_device_supports_op", "sycl_supports_op", f"""
    // {MARKER} sycl_supports_op
    if (op->op >= GGML_OP_EXT_BASE) return g_ggml_bridge_supports_hook && g_ggml_bridge_supports_hook(dev, op);
""")
    sycl.replace_signature("ggml_backend_sycl_graph_compute_impl", "ggml_backend_t backend, ggml_backend_sycl_context", f"""
// {MARKER} sycl_status_signature
static ggml_status ggml_backend_sycl_graph_compute_impl(ggml_backend_t backend, ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph)
""")
    sycl.insert_before_call("ggml_backend_sycl_graph_compute_impl", "ggml_sycl_compute_forward", "sycl_graph_compute_dispatch", f"""
        // {MARKER} sycl_graph_compute_dispatch
        if (node->op >= GGML_OP_EXT_BASE && g_ggml_bridge_hook) {{
            const int ext_result = g_ggml_bridge_hook(backend, node);
            if (ext_result == GGML_OPS_EXT_SUCCESS) continue;
            return GGML_STATUS_FAILED;
        }}
""")
    sycl.insert_before_function_end("ggml_backend_sycl_graph_compute_impl", "sycl_status_return", f"""
    // {MARKER} sycl_status_return
    return GGML_STATUS_SUCCESS;
""")
    sycl.replace_call_statements("ggml_backend_sycl_graph_compute", "ggml_backend_sycl_graph_compute_impl", "ggml_backend_sycl_graph_compute_impl(backend", f"""// {MARKER} sycl_status_call
        if (ggml_backend_sycl_graph_compute_impl(backend, sycl_ctx, cgraph) != GGML_STATUS_SUCCESS) return GGML_STATUS_FAILED;""")
    sycl.save()


def verify_bridge(ggml_root: Path) -> None:
    requirements = {
        "include/ggml.h": ["GGML_OP_EXT_RESERVED_MAX"],
        "src/ggml.cpp": ["ggml-ops-ext-bridge.cpp"],
        "src/ggml-cpu/ggml-cpu.cpp": ["cpu_graph_compute_dispatch", "cpu_supports_op"],
        "src/ggml-cuda/ggml-cuda.cu": ["cuda_graph_compute_dispatch", "cuda_status_call"],
        "src/ggml-sycl/ggml-sycl.cpp": ["sycl_graph_compute_dispatch", "sycl_status_call"],
    }
    for relative, markers in requirements.items():
        text = (ggml_root / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise BridgeConflict(f"{relative}: missing bridge contract {marker}")
