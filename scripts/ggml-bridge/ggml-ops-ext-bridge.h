#ifndef GGML_OPS_EXT_BRIDGE_H
#define GGML_OPS_EXT_BRIDGE_H

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OP_EXT_BASE  2000

typedef bool (*ggml_ops_ext_handler_t)(
    ggml_backend_t backend,
    struct ggml_tensor * node
);

// ---- Hook pointer (the ONLY exported symbol) ----
// Defined in ggml-ops-ext-bridge.cpp (included inline by ggml-backend-reg.cpp).
// Backend files reference this directly via extern declaration.
GGML_API ggml_ops_ext_handler_t g_ggml_bridge_hook;

// Register a handler. Called from ops layer (tts.dll).
GGML_API void ggml_ops_ext_bridge_set_hook(ggml_ops_ext_handler_t hook);

// ---- Native resource getters (compiled inside ggml-cuda / ggml-sycl) ----
GGML_API int  ggml_ops_ext_bridge_cuda_get_device(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_cuda_get_stream(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_cuda_get_cublas(ggml_backend_t backend);
GGML_API void* ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif

#endif // GGML_OPS_EXT_BRIDGE_H
