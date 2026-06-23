// This file is #included inline by ggml-backend-reg.cpp.
#include "ggml-ops-ext-bridge.h"

// ---- Hook pointer (exported from ggml.dll) ----
GGML_API ggml_ops_ext_handler_t g_ggml_bridge_hook = nullptr;

GGML_API void ggml_ops_ext_bridge_set_hook(ggml_ops_ext_handler_t hook) {
    g_ggml_bridge_hook = hook;
}
