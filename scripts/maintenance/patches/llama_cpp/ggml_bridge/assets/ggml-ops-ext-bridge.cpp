// This file is #included inline by ggml-backend-reg.cpp.
#include "ggml-ops-ext-bridge.h"

// ---- Hook pointer (exported from ggml.dll) ----
GGML_API ggml_ops_ext_handler_t g_ggml_bridge_hook = nullptr;
GGML_API ggml_ops_ext_supports_t g_ggml_bridge_supports_hook = nullptr;

GGML_API void ggml_ops_ext_bridge_set_hooks(
    ggml_ops_ext_handler_t handler,
    ggml_ops_ext_supports_t supports
) {
    g_ggml_bridge_supports_hook = supports;
    g_ggml_bridge_hook = handler;
}
