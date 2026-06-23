#include "ggml-ops-ext-bridge.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

// This file is compiled inside ggml-sycl target and has
// legal access to ggml_backend_sycl_context.

void * ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;
    if (!sycl_ctx) return nullptr;

    return (void *)sycl_ctx->stream();
}
