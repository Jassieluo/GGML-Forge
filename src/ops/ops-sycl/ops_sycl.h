#pragma once

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

// ────────────────────────────────────────────────────────────
// Bridge function pointer — resolved at runtime from
// ggml-sycl.dll (MODULE_LIBRARY, no import library).
// ────────────────────────────────────────────────────────────

typedef void* (*pfn_bridge_sycl_get_queue_t)(ggml_backend_t);

extern pfn_bridge_sycl_get_queue_t g_bridge_sycl_get_queue;

inline void * ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend) {
    return g_bridge_sycl_get_queue ? g_bridge_sycl_get_queue(backend) : nullptr;
}

namespace ggml_ops_ext {
namespace sycl {

void register_backend();

} // namespace sycl
} // namespace ggml_ops_ext
