#include "ops_cuda.h"
#include "ops_cuda_common.cuh"
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Runtime bridge symbols resolved from ggml-cuda.dll.

pfn_bridge_cuda_get_device_t g_bridge_cuda_get_device = nullptr;
pfn_bridge_cuda_get_stream_t g_bridge_cuda_get_stream = nullptr;
pfn_bridge_cuda_get_cublas_t g_bridge_cuda_get_cublas = nullptr;
pfn_bridge_cuda_dequantize_t g_bridge_cuda_dequantize = nullptr;

static void resolve_bridge_functions() {
    static std::once_flag resolve_once;
    std::call_once(resolve_once, []() {

#ifdef _WIN32
        HMODULE dll = GetModuleHandleW(L"ggml-cuda.dll");
        if (!dll)
            dll = GetModuleHandleW(L"ggml-cuda");
        if (!dll)
            dll = LoadLibraryW(L"ggml-cuda.dll");
        if (!dll)
            dll = LoadLibraryW(L"ggml-cuda");
        if (!dll)
            return;

        g_bridge_cuda_get_device = (pfn_bridge_cuda_get_device_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_cuda_get_device");
        g_bridge_cuda_get_stream = (pfn_bridge_cuda_get_stream_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_cuda_get_stream");
        g_bridge_cuda_get_cublas = (pfn_bridge_cuda_get_cublas_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_cuda_get_cublas");
        g_bridge_cuda_dequantize = (pfn_bridge_cuda_dequantize_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_cuda_dequantize");
#else
    void * dll = dlopen("libggml-cuda.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dll) return;

    g_bridge_cuda_get_device = (pfn_bridge_cuda_get_device_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_device");
    g_bridge_cuda_get_stream = (pfn_bridge_cuda_get_stream_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_stream");
    g_bridge_cuda_get_cublas = (pfn_bridge_cuda_get_cublas_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_get_cublas");
    g_bridge_cuda_dequantize = (pfn_bridge_cuda_dequantize_t)
        dlsym(dll, "ggml_ops_ext_bridge_cuda_dequantize");
#endif
    });
}

namespace ggml_ops_ext {
namespace cuda {

// Forward declarations of entrypoint handlers implemented in separate files
bool ggml_cuda_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_conv_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_pool_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_pad_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_adaptive_pool_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_resize_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_double_swish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_attention_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_kv_cache_update_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_gated_activation_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_relative_pe_values_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_snake_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_snake_beta_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_cuda_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node);

static ops_probe_result supports_conv(const ops_request& request) {
    if (!ops_validate_conv_request(request)) {
        return false;
    }
    const ops_quantization_desc weight =
        ops_describe_quantization(request.srcs[0], ops_weight_layout::channel_rows);
    const ggml_type x_type = request.srcs[1]->type;
    const bool weight_ok =
        weight.storage_type == GGML_TYPE_F32 || weight.storage_type == GGML_TYPE_F16 ||
        weight.storage_type == GGML_TYPE_BF16 || weight.scheme == ops_quant_scheme::q4_0 ||
        weight.scheme == ops_quant_scheme::q8_0 || weight.scheme == ops_quant_scheme::q4_k;
    const bool activation_ok = x_type == GGML_TYPE_F32 || x_type == GGML_TYPE_F16;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] ||
                         request.srcs[2]->type == GGML_TYPE_F32 ||
                         request.srcs[2]->type == GGML_TYPE_F16;
    return weight_ok && activation_ok && bias_ok;
}

static ops_probe_result supports_standard(const ops_request& request) {
    return ops_validate_request_contract(ops_support_profile::gpu, request);
}

static ops_probe_result supports_conv_nd(const ops_request& request) {
    if (!ops_validate_conv_request(request)) {
        return false;
    }
    const ggml_type weight = request.srcs[0]->type;
    const ggml_type input = request.srcs[1]->type;
    const bool weight_ok =
        weight == GGML_TYPE_F32 || weight == GGML_TYPE_F16 || weight == GGML_TYPE_BF16 ||
        weight == GGML_TYPE_Q4_0 || weight == GGML_TYPE_Q4_1 || weight == GGML_TYPE_Q5_0 ||
        weight == GGML_TYPE_Q5_1 || weight == GGML_TYPE_Q8_0 || weight == GGML_TYPE_Q2_K ||
        weight == GGML_TYPE_Q3_K || weight == GGML_TYPE_Q4_K || weight == GGML_TYPE_Q5_K ||
        weight == GGML_TYPE_Q6_K;
    const bool extended_weight_ok =
        weight == GGML_TYPE_IQ4_NL || weight == GGML_TYPE_IQ4_XS || weight == GGML_TYPE_MXFP4;
    const bool input_ok =
        input == GGML_TYPE_F32 || input == GGML_TYPE_F16 || input == GGML_TYPE_BF16;
    const bool bias_ok =
        request.n_srcs < 3 || !request.srcs[2] || request.srcs[2]->type == GGML_TYPE_F32 ||
        request.srcs[2]->type == GGML_TYPE_F16 || request.srcs[2]->type == GGML_TYPE_BF16;
    return (weight_ok || extended_weight_ok) && input_ok && bias_ok;
}

static ops_probe_result supports_pool_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_POOL_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_POOL_2D ? 2
                                                                         : 3;
    if (!ops_validate_pool_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static ops_probe_result supports_pad_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_PAD_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_PAD_2D ? 2
                                                                        : 3;
    if (!ops_validate_pad_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static ops_probe_result supports_adaptive_pool_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                                  : 3;
    if (!ops_validate_adaptive_pool_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}
static ops_probe_result supports_resize_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_RESIZE_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_RESIZE_2D ? 2
                                                                           : 3;
    if (!ops_validate_resize_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static const ops_kernel_entry CUDA_KERNELS[] = {
    make_ops_kernel<ggml_cuda_op_conv_1d_entry>(GGML_OP_OPS_VIRT_CONV_1D, "cuda.conv1d",
                                                supports_conv, 100),
    make_ops_kernel<ggml_cuda_op_conv_transpose_1d_entry>(
        GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, "cuda.conv_transpose1d", supports_conv, 100),
    make_ops_kernel<ggml_cuda_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_2D, "cuda.conv2d",
                                                supports_conv_nd, 100),
    make_ops_kernel<ggml_cuda_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D,
                                                "cuda.conv_transpose2d", supports_conv_nd, 100),
    make_ops_kernel<ggml_cuda_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_3D, "cuda.conv3d",
                                                supports_conv_nd, 100),
    make_ops_kernel<ggml_cuda_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D,
                                                "cuda.conv_transpose3d", supports_conv_nd, 100),
    make_ops_kernel<ggml_cuda_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_1D, "cuda.pool1d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_2D, "cuda.pool2d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_3D, "cuda.pool3d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_1D, "cuda.pad1d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_cuda_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_2D, "cuda.pad2d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_cuda_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_3D, "cuda.pad3d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_cuda_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D, "cuda.adaptive_pool1d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D, "cuda.adaptive_pool2d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_3D, "cuda.adaptive_pool3d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_cuda_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_1D, "cuda.resize1d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_cuda_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_2D, "cuda.resize2d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_cuda_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_3D, "cuda.resize3d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_cuda_op_mish_entry>(GGML_OP_OPS_VIRT_MISH, "cuda.mish", supports_standard,
                                             100),
    make_ops_kernel<ggml_cuda_op_gated_tanh_sigmoid_entry>(
        GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, "cuda.gated_tanh_sigmoid", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_layer_norm_entry>(GGML_OP_OPS_VIRT_LAYER_NORM, "cuda.layer_norm",
                                                   supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_double_swish_entry>(GGML_OP_OPS_VIRT_DOUBLE_SWISH,
                                                     "cuda.double_swish", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_attention_entry>(GGML_OP_OPS_VIRT_FUSED_ATTN, "cuda.attention",
                                                  supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_kv_cache_update_entry>(
        GGML_OP_OPS_VIRT_KV_CACHE_UPDATE, "cuda.kv_cache_update", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_glu_entry>(GGML_OP_OPS_VIRT_GLU, "cuda.glu", supports_standard,
                                            100),
    make_ops_kernel<ggml_cuda_op_gated_activation_entry>(GGML_OP_OPS_VIRT_GATED_ACTIVATION,
                                                         "cuda.gated_activation", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_relative_pe_keys_entry>(
        GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS, "cuda.relative_pe_keys", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_relative_pe_values_entry>(
        GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, "cuda.relative_pe_values", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_instance_norm_entry>(GGML_OP_OPS_VIRT_INSTANCE_NORM,
                                                      "cuda.instance_norm", supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_snake_entry>(GGML_OP_OPS_VIRT_SNAKE, "cuda.snake",
                                              supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_snake_beta_entry>(GGML_OP_OPS_VIRT_SNAKE_BETA, "cuda.snake_beta",
                                                   supports_standard, 100),
    make_ops_kernel<ggml_cuda_op_ada_ln_entry>(GGML_OP_OPS_VIRT_ADA_LN, "cuda.ada_ln",
                                               supports_standard, 100),
};

void register_backend() {
    resolve_bridge_functions();

    ops_backend_registration iface = {"CUDA", CUDA_KERNELS,
                                      sizeof(CUDA_KERNELS) / sizeof(CUDA_KERNELS[0])};
    register_ops_backend(iface);
}

struct RegisterCuda {
    RegisterCuda() { register_backend(); }
} g_register_cuda;

} // namespace cuda
} // namespace ggml_ops_ext

#ifdef _WIN32
extern "C" __declspec(dllexport) void ggml_ops_ext_cuda_init() {
#else
extern "C" void ggml_ops_ext_cuda_init() {
#endif
    // Force loading of DLL
}

// Note: ggml_cuda_set_device / ggml_cuda_error are now inline-defined
// in ops_cuda_common.cuh before the ggml-cuda/common.cuh include.
// No separate definitions needed here.
