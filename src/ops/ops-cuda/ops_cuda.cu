#include "ops_cuda_common.cuh"
#include "ops_cuda.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifndef GGML_CUDA_MAX_DEVICES
#define GGML_CUDA_MAX_DEVICES 8
#endif

#ifdef GGML_USE_CUDNN
static cudnnHandle_t g_cudnn_handles[GGML_CUDA_MAX_DEVICES] = { nullptr };
static std::mutex g_cudnn_mutex;
#endif

// ────────────────────────────────────────────────────────────
// Bridge function pointers — resolved at runtime from
// ggml-cuda.dll (which is a MODULE_LIBRARY, no import lib).
// ────────────────────────────────────────────────────────────

pfn_bridge_cuda_get_device_t g_bridge_cuda_get_device = nullptr;
pfn_bridge_cuda_get_stream_t g_bridge_cuda_get_stream = nullptr;
pfn_bridge_cuda_get_cublas_t g_bridge_cuda_get_cublas = nullptr;

static void resolve_bridge_functions() {
    static bool resolved = false;
    if (resolved) return;
    resolved = true;

#ifdef _WIN32
    HMODULE dll = GetModuleHandleW(L"ggml-cuda.dll");
    if (!dll) dll = GetModuleHandleW(L"ggml-cuda");
    if (!dll) dll = LoadLibraryW(L"ggml-cuda.dll");
    if (!dll) dll = LoadLibraryW(L"ggml-cuda");
    if (!dll) return;

    g_bridge_cuda_get_device = (pfn_bridge_cuda_get_device_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_cuda_get_device");
    g_bridge_cuda_get_stream = (pfn_bridge_cuda_get_stream_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_cuda_get_stream");
    g_bridge_cuda_get_cublas = (pfn_bridge_cuda_get_cublas_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_cuda_get_cublas");
#else
    void * dll = dlopen("libggml-cuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dll) return;

    g_bridge_cuda_get_device = (pfn_bridge_cuda_get_device_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_device");
    g_bridge_cuda_get_stream = (pfn_bridge_cuda_get_stream_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_stream");
    g_bridge_cuda_get_cublas = (pfn_bridge_cuda_get_cublas_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_cublas");
#endif
}

namespace ggml_ops_ext {
namespace cuda {

#ifdef GGML_USE_CUDNN
cudnnHandle_t get_cudnn_handle(int device) {
    std::lock_guard<std::mutex> lock(g_cudnn_mutex);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!g_cudnn_handles[device]) {
        CUDNN_CHECK(cudnnCreate(&g_cudnn_handles[device]));
    }
    return g_cudnn_handles[device];
}
#endif

// Forward declarations of entrypoint handlers implemented in separate files
bool ggml_cuda_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_double_swish_entry(ggml_backend_t backend, struct ggml_tensor* node);

static const ops_handler_entry CUDA_HANDLERS[] = {
    { GGML_OP_OPS_VIRT_CONV_1D,           ggml_cuda_op_conv_1d_entry },
    { GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, ggml_cuda_op_conv_transpose_1d_entry },
    { GGML_OP_OPS_VIRT_MISH,               ggml_cuda_op_mish_entry },
    { GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, ggml_cuda_op_gated_tanh_sigmoid_entry },
    { GGML_OP_OPS_VIRT_LAYER_NORM,         ggml_cuda_op_layer_norm_entry },
    { GGML_OP_OPS_VIRT_DOUBLE_SWISH,       ggml_cuda_op_double_swish_entry },
};

void register_backend() {
    resolve_bridge_functions();

    ops_backend_interface iface = {
        /* backend_name_prefix */ "CUDA",
        /* handlers            */ CUDA_HANDLERS,
        /* n_handlers          */ sizeof(CUDA_HANDLERS) / sizeof(CUDA_HANDLERS[0]),
        /* builders            */ nullptr,
        /* n_builders          */ 0
    };
    register_ops_backend(iface);
}

struct RegisterCuda {
    RegisterCuda() {
        register_backend();
    }
} g_register_cuda;

} // namespace cuda
} // namespace ggml_ops_ext

// Note: ggml_cuda_set_device / ggml_cuda_error are now inline-defined
// in ops_cuda_common.cuh before the ggml-cuda/common.cuh include.
// No separate definitions needed here.

