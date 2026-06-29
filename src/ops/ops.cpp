#include "ops/ops.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>
#include <iostream>

namespace ggml_ops_ext {

static std::vector<ops_backend_interface> g_registered_backends;
static std::recursive_mutex g_hooks_mutex;

#include "ggml-ops-ext-bridge.h"

static bool ops_ext_hook_impl(ggml_backend_t backend, struct ggml_tensor* node) {
    if (!node || node->op < 2000) {
        return false;
    }

    const ops_backend_interface* ops_backend = find_ops_backend(backend);
    ops_op_handler_t handler = nullptr;
    if (ops_backend) {
        for (int j = 0; j < ops_backend->n_handlers; ++j) {
            if (ops_backend->handlers[j].op_id == (int)node->op) {
                handler = ops_backend->handlers[j].handler;
                break;
            }
        }
    }

    if (handler) {
        bool ok = handler(backend, node);
        return ok;
    }
    return false;
}

void register_ops_backend(const ops_backend_interface& iface) {
    std::lock_guard<std::recursive_mutex> lock(g_hooks_mutex);
    g_registered_backends.push_back(iface);
}

const ops_backend_interface* find_ops_backend(ggml_backend_t backend) {
    const char* bname = ggml_backend_name(backend);
    if (!bname) return nullptr;

    for (const auto& iface : g_registered_backends) {
        size_t prefix_len = std::strlen(iface.backend_name_prefix);
        if (std::strncmp(bname, iface.backend_name_prefix, prefix_len) == 0) {
            return &iface;
        }
    }
    return nullptr;
}

ops_op_builder_t find_ops_builder(ggml_backend_t backend, int op_id) {
    if (!backend) return nullptr;
    const ops_backend_interface* ops_backend = find_ops_backend(backend);
    if (ops_backend && ops_backend->builders) {
        for (int i = 0; i < ops_backend->n_builders; ++i) {
            if (ops_backend->builders[i].op_id == op_id) {
                return ops_backend->builders[i].builder;
            }
        }
    }
    return nullptr;
}

void install_ops_hook(ggml_backend_t backend) {
    (void)backend;
    ggml_ops_ext_bridge_set_hook(ops_ext_hook_impl);
}

void uninstall_ops_hook(ggml_backend_t backend) {
    (void)backend;
    ggml_ops_ext_bridge_set_hook(nullptr);
}

struct ggml_tensor* ops_new_virtual_node(
    struct ggml_context* ctx,
    ops_virt_op_type op,
    ggml_type type,
    int n_dims,
    const int64_t* ne,
    int n_srcs,
    struct ggml_tensor** srcs
) {
    struct ggml_tensor* result = ggml_new_tensor(ctx, type, n_dims, ne);
    result->op = (enum ggml_op)op;
    GGML_ASSERT(n_srcs <= GGML_MAX_SRC);
    for (int i = 0; i < n_srcs; ++i) {
        result->src[i] = srcs[i];
    }
    return result;
}

} // namespace ggml_ops_ext

// Global namespace custom operator wrapper functions
bool ggml_ops_backend_supports_op(ggml_backend_t backend, int op_id) {
    if (!backend) return false;
    const ggml_ops_ext::ops_backend_interface* ops_backend = ggml_ops_ext::find_ops_backend(backend);
    if (ops_backend) {
        for (int i = 0; i < ops_backend->n_handlers; ++i) {
            if (ops_backend->handlers[i].op_id == op_id) {
                return true;
            }
        }
    }
    return false;
}

