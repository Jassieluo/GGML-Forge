#include "ops_cuda_common.cuh"
#include "ops_cuda.h"

#ifdef GGML_USE_CUDNN
static cudnnHandle_t g_cudnn_handles[GGML_CUDA_MAX_DEVICES] = { nullptr };
static std::mutex g_cudnn_mutex;
#endif

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

// Global namespace definitions to resolve internal ggml-cuda link dependencies
void ggml_cuda_set_device(int device) {
    cudaSetDevice(device);
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s in %s at %s:%d - %s\n", stmt, func, file, line, msg);
}

