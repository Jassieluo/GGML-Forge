#ifndef GGML_OPS_EXT_BRIDGE_H
#define GGML_OPS_EXT_BRIDGE_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OP_EXT_BASE  2000

enum ggml_ops_ext_result {
    GGML_OPS_EXT_NOT_HANDLED = 0,
    GGML_OPS_EXT_SUCCESS = 1,
    GGML_OPS_EXT_FAILED = -1,
};

typedef int (*ggml_ops_ext_handler_t)(
    ggml_backend_t backend,
    struct ggml_tensor * node
);

typedef bool (*ggml_ops_ext_supports_t)(
    ggml_backend_dev_t device,
    const struct ggml_tensor * node
);

// ---- Hook pointer (the ONLY exported symbol) ----
// Defined in ggml-ops-ext-bridge.cpp (included inline by ggml-backend-reg.cpp).
// Backend files reference this directly via extern declaration.
GGML_API ggml_ops_ext_handler_t g_ggml_bridge_hook;
GGML_API ggml_ops_ext_supports_t g_ggml_bridge_supports_hook;

// Install both hooks under the ops registry lifecycle lock.
GGML_API void ggml_ops_ext_bridge_set_hooks(
    ggml_ops_ext_handler_t handler,
    ggml_ops_ext_supports_t supports
);

// ---- Native resource getters (compiled inside ggml-cuda / ggml-sycl) ----
GGML_API int  ggml_ops_ext_bridge_cuda_get_device(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_cuda_get_stream(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_cuda_get_cublas(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend);
GGML_API bool ggml_ops_ext_bridge_cuda_dequantize(ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type);
GGML_API bool ggml_ops_ext_bridge_sycl_dequantize(ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type);
GGML_API void* ggml_ops_ext_bridge_sycl_pool_alloc(ggml_backend_t backend, size_t size, size_t* actual_size);
GGML_API void ggml_ops_ext_bridge_sycl_pool_free(ggml_backend_t backend, void* ptr, size_t actual_size);

#ifdef __cplusplus
}
#endif

#endif // GGML_OPS_EXT_BRIDGE_H
