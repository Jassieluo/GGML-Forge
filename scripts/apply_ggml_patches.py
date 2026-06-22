import os
import re
import sys

def patch_file(filepath, patch_rules):
    """
    Applies search-and-replace rules to a file.
    Returns True if matches were found and modified, False otherwise.
    """
    if not os.path.exists(filepath):
        print(f"[GGML Patch] Error: File {filepath} does not exist!")
        return False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    modified = False
    for rule in patch_rules:
        if len(rule) == 4:
            rule_name, pattern, replacement, skip_marker = rule
        else:
            rule_name, pattern, replacement = rule
            skip_marker = replacement

        # Check if already patched to maintain idempotency
        if skip_marker in content:
            print(f"[GGML Patch] Rule '{rule_name}' already applied to {os.path.basename(filepath)}")
            continue

        if re.search(pattern, content, re.DOTALL):
            new_content, count = re.subn(pattern, replacement, content, flags=re.DOTALL)
            if count > 0:
                content = new_content
                modified = True
                print(f"[GGML Patch] Successfully applied rule '{rule_name}' to {os.path.basename(filepath)} ({count} occurrences)")
            else:
                print(f"[GGML Patch] Error: Rule '{rule_name}' pattern found but substitution failed!")
                sys.exit(1)
        else:
            print(f"[GGML Patch] Error: Could not find pattern for rule '{rule_name}' in {os.path.basename(filepath)}!")
            print(f"Pattern expected: {pattern}")
            sys.exit(1)

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    return True

def main():
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    ggml_dir = os.path.join(base_dir, "ggml")

    print("[GGML Patch] Running semantic patch checks on upstream ggml...")

    # 1. Rules for ggml-backend.cpp
    backend_path = os.path.join(ggml_dir, "src", "ggml-backend.cpp")
    backend_rules = [
        (
            "add_global_hook_api",
            r"\Z",  # Matches end of file
            "\n\n// Added by GPT-SoVITS.cpp Custom Operator Framework\n"
            'extern "C" {\n'
            "    typedef bool (*ggml_ops_ext_hook_t)(ggml_backend_t backend, struct ggml_tensor * node);\n"
            "    GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook = nullptr;\n"
            "    GGML_API void ggml_backend_set_ops_ext_hook(ggml_ops_ext_hook_t hook) {\n"
            "        g_ggml_ops_ext_hook = hook;\n"
            "    }\n"
            "}\n",
            "g_ggml_ops_ext_hook = hook;"
        )
    ]

    # 2. Rules for ggml-cpu.cpp
    cpu_path = os.path.join(ggml_dir, "src", "ggml-cpu", "ggml-cpu.cpp")
    cpu_rules = [
        (
            "cpu_add_global_hook_decl",
            r"(#include \"ggml-impl.h\")",
            r"\1" + "\n\n"
            'extern "C" {\n'
            "    typedef bool (*ggml_ops_ext_hook_t)(ggml_backend_t backend, struct ggml_tensor * node);\n"
            "    GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;\n"
            "}\n",
            "GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;"
        ),
        (
            "cpu_supports_op_virtual",
            r"(static bool ggml_backend_cpu_device_supports_op\([^{]*\{[^}]*?if \(op->op == GGML_OP_NONE \|\| op->op == GGML_OP_RESHAPE \|\| op->op == GGML_OP_VIEW \|\| op->op == GGML_OP_PERMUTE \|\| op->op == GGML_OP_TRANSPOSE)",
            r"\1 || op->op >= 2000",
            "|| op->op >= 2000"
        ),
        (
            "cpu_graph_compute_hook",
            r"(static enum ggml_status ggml_backend_cpu_graph_compute\(ggml_backend_t backend, struct ggml_cgraph \* cgraph\) \{)",
            r"\1" + "\n"
            "    if (g_ggml_ops_ext_hook) {\n"
            "        for (int i = 0; i < cgraph->n_nodes; ++i) {\n"
            "            struct ggml_tensor * node = cgraph->nodes[i];\n"
            "            if (node->op >= 2000) {\n"
            "                g_ggml_ops_ext_hook(backend, node);\n"
            "                node->op = GGML_OP_NONE;\n"
            "            }\n"
            "        }\n"
            "    }",
            "if (g_ggml_ops_ext_hook) {\n        for (int i = 0; i < cgraph->n_nodes; ++i) {"
        )
    ]

    # 3. Rules for ggml-cuda.cu
    cuda_path = os.path.join(ggml_dir, "src", "ggml-cuda", "ggml-cuda.cu")
    cuda_rules = [
        (
            "cuda_add_global_hook_decl",
            r"(#include \"ggml-backend-impl.h\")",
            r"\1" + "\n\n"
            'extern "C" {\n'
            "    typedef bool (*ggml_ops_ext_hook_t)(ggml_backend_t backend, struct ggml_tensor * node);\n"
            "    GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;\n"
            "}\n",
            "GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;"
        ),
        (
            "cuda_supports_op_virtual",
            r"(static bool ggml_backend_cuda_device_supports_op\([^{]*\{.*?switch \(op->op\) \{.*?default:\n\s+)return false;",
            r"\1if (op->op >= 2000) return true; return false;",
            "if (op->op >= 2000) return true; return false;"
        ),
        (
            "cuda_capture_sig_update",
            r"static void ggml_cuda_graph_evaluate_and_capture\(ggml_backend_cuda_context \* cuda_ctx,",
            r"static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_t backend, ggml_backend_cuda_context * cuda_ctx,",
            "static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_t backend"
        ),
        (
            "cuda_capture_call_update",
            r"ggml_cuda_graph_evaluate_and_capture\(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key\);",
            r"ggml_cuda_graph_evaluate_and_capture(backend, cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);",
            "ggml_cuda_graph_evaluate_and_capture(backend, cuda_ctx"
        ),
        (
            "cuda_compute_hook",
            r"(\s+bool ok = ggml_cuda_compute_forward\(\*cuda_ctx, node\);)",
            "\n                if (g_ggml_ops_ext_hook && g_ggml_ops_ext_hook(backend, node)) {\n"
            "                    continue;\n"
            "                }\n" + r"\1",
            "if (g_ggml_ops_ext_hook && g_ggml_ops_ext_hook(backend, node)) {\n                    continue;"
        )
    ]

    # 4. Rules for ggml-sycl.cpp
    sycl_path = os.path.join(ggml_dir, "src", "ggml-sycl", "ggml-sycl.cpp")
    sycl_rules = [
        (
            "sycl_add_global_hook_decl",
            r"(#include \"ggml-backend-impl.h\")",
            r"\1" + "\n\n"
            'extern "C" {\n'
            "    typedef bool (*ggml_ops_ext_hook_t)(ggml_backend_t backend, struct ggml_tensor * node);\n"
            "    GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;\n"
            "}\n",
            "GGML_API ggml_ops_ext_hook_t g_ggml_ops_ext_hook;"
        ),
        (
            "sycl_supports_op_virtual",
            r"(static bool ggml_backend_sycl_device_supports_op\(ggml_backend_dev_t dev, const ggml_tensor \* op\) \{)",
            r"\1\n    if (op->op >= 2000) return true;",
            "if (op->op >= 2000) return true;"
        ),
        (
            "sycl_compute_impl_sig_update",
            r"static void ggml_backend_sycl_graph_compute_impl\(ggml_backend_sycl_context \* sycl_ctx,",
            r"static void ggml_backend_sycl_graph_compute_impl(ggml_backend_t backend, ggml_backend_sycl_context * sycl_ctx,",
            "static void ggml_backend_sycl_graph_compute_impl(ggml_backend_t backend"
        ),
        (
            "sycl_compute_impl_call_update",
            r"ggml_backend_sycl_graph_compute_impl\(sycl_ctx, cgraph\);",
            r"ggml_backend_sycl_graph_compute_impl(backend, sycl_ctx, cgraph);",
            "ggml_backend_sycl_graph_compute_impl(backend, sycl_ctx"
        ),
        (
            "sycl_compute_hook",
            r"(\s+bool ok = ggml_sycl_compute_forward\(\*sycl_ctx, node\);)",
            "\n        if (g_ggml_ops_ext_hook && g_ggml_ops_ext_hook(backend, node)) {\n"
            "            continue;\n"
            "        }\n" + r"\1",
            "if (g_ggml_ops_ext_hook && g_ggml_ops_ext_hook(backend, node)) {\n            continue;"
        )
    ]

    # Apply all rules
    patch_file(backend_path, backend_rules)
    patch_file(cpu_path, cpu_rules)
    patch_file(cuda_path, cuda_rules)
    patch_file(sycl_path, sycl_rules)

    print("[GGML Patch] Semantic patching completed successfully!")

if __name__ == '__main__':
    main()
