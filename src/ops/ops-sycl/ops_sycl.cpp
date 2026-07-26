#include "ops_sycl.h"
#include "ops/ops.h"
#include <iostream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Runtime bridge symbols resolved from ggml-sycl.dll.

pfn_bridge_sycl_get_queue_t g_bridge_sycl_get_queue = nullptr;
pfn_bridge_sycl_dequantize_t g_bridge_sycl_dequantize = nullptr;
pfn_bridge_sycl_pool_alloc_t g_bridge_sycl_pool_alloc = nullptr;
pfn_bridge_sycl_pool_free_t g_bridge_sycl_pool_free = nullptr;

static void resolve_bridge_sycl_functions() {
    static std::once_flag resolve_once;
    std::call_once(resolve_once, []() {

#ifdef _WIN32
        HMODULE dll = GetModuleHandleW(L"ggml-sycl.dll");
        if (!dll)
            dll = GetModuleHandleW(L"ggml-sycl");
        if (!dll)
            dll = LoadLibraryW(L"ggml-sycl.dll");
        if (!dll)
            dll = LoadLibraryW(L"ggml-sycl");
        if (!dll) {
            std::cerr << "[SYCL Bridge] Error: Failed to load ggml-sycl.dll!\n";
            return;
        }

        g_bridge_sycl_get_queue =
            (pfn_bridge_sycl_get_queue_t)GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_get_queue");
        g_bridge_sycl_dequantize = (pfn_bridge_sycl_dequantize_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_sycl_dequantize");
        g_bridge_sycl_pool_alloc = (pfn_bridge_sycl_pool_alloc_t)GetProcAddress(
            dll, "ggml_ops_ext_bridge_sycl_pool_alloc");
        g_bridge_sycl_pool_free =
            (pfn_bridge_sycl_pool_free_t)GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_pool_free");
        if (!g_bridge_sycl_get_queue) {
            std::cerr << "[SYCL Bridge] Error: Failed to resolve "
                         "ggml_ops_ext_bridge_sycl_get_queue from ggml-sycl.dll!\n";
        } else {
            std::cout << "[SYCL Bridge] Successfully loaded ggml-sycl.dll and "
                         "resolved get_queue "
                         "symbol!\n";
        }
#else
    void * dll = dlopen("libggml-sycl.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dll) {
        std::cerr << "[SYCL Bridge] Error: Failed to dlopen libggml-sycl.so!\n";
        return;
    }

    g_bridge_sycl_get_queue = (pfn_bridge_sycl_get_queue_t)
        dlsym(dll, "ggml_ops_ext_bridge_sycl_get_queue");
    g_bridge_sycl_dequantize = (pfn_bridge_sycl_dequantize_t)
        dlsym(dll, "ggml_ops_ext_bridge_sycl_dequantize");
    g_bridge_sycl_pool_alloc = (pfn_bridge_sycl_pool_alloc_t)
        dlsym(dll, "ggml_ops_ext_bridge_sycl_pool_alloc");
    g_bridge_sycl_pool_free = (pfn_bridge_sycl_pool_free_t)
        dlsym(dll, "ggml_ops_ext_bridge_sycl_pool_free");
    if (!g_bridge_sycl_get_queue) {
        std::cerr << "[SYCL Bridge] Error: Failed to resolve ggml_ops_ext_bridge_sycl_get_queue from libggml-sycl.so!\n";
    } else {
        std::cout << "[SYCL Bridge] Successfully loaded libggml-sycl.so and resolved get_queue symbol!\n";
    }
#endif
    });
}

namespace ggml_ops_ext {
namespace sycl {

// Declarations of entrypoints implemented in other files
bool ggml_sycl_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_conv_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_pool_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_pad_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_adaptive_pool_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_resize_nd_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_double_swish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_attention_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_kv_cache_update_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_gated_activation_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_relative_pe_values_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_snake_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_snake_beta_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_alias_free_activation_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_fused_norm_act_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_pos_encoding_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_sample_dist_entry(ggml_backend_t backend, struct ggml_tensor* node);

static ops_probe_result supports_conv(const ops_request& request) {
    if (!ops_validate_conv_request(request)) {
        return false;
    }
    const ops_quantization_desc weight =
        ops_describe_quantization(request.srcs[0], ops_weight_layout::channel_rows);
    const ggml_type x_type = request.srcs[1]->type;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] ||
                         request.srcs[2]->type == GGML_TYPE_F32 ||
                         request.srcs[2]->type == GGML_TYPE_F16;
    if (!bias_ok) {
        return false;
    }
    return (x_type == GGML_TYPE_F32 || x_type == GGML_TYPE_F16) &&
           (weight.storage_type == GGML_TYPE_F32 || weight.storage_type == GGML_TYPE_F16 ||
            weight.scheme == ops_quant_scheme::q4_0 || weight.scheme == ops_quant_scheme::q8_0 ||
            weight.scheme == ops_quant_scheme::q4_k);
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
        weight == GGML_TYPE_F32 || weight == GGML_TYPE_F16 || weight == GGML_TYPE_Q4_0 ||
        weight == GGML_TYPE_Q4_1 || weight == GGML_TYPE_Q5_0 || weight == GGML_TYPE_Q5_1 ||
        weight == GGML_TYPE_Q8_0 || weight == GGML_TYPE_Q2_K || weight == GGML_TYPE_Q3_K ||
        weight == GGML_TYPE_Q4_K || weight == GGML_TYPE_Q5_K || weight == GGML_TYPE_Q6_K;
    const bool extended_weight_ok =
        weight == GGML_TYPE_IQ4_NL || weight == GGML_TYPE_IQ4_XS || weight == GGML_TYPE_MXFP4;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] ||
                         request.srcs[2]->type == GGML_TYPE_F32 ||
                         request.srcs[2]->type == GGML_TYPE_F16;
    return (weight_ok || extended_weight_ok) &&
           (input == GGML_TYPE_F32 || input == GGML_TYPE_F16) && bias_ok;
}

static ops_probe_result supports_pool_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_POOL_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_POOL_2D ? 2
                                                                         : 3;
    if (!ops_validate_pool_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static ops_probe_result supports_pad_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_PAD_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_PAD_2D ? 2
                                                                        : 3;
    if (!ops_validate_pad_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static ops_probe_result supports_adaptive_pool_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                                  : 3;
    if (!ops_validate_adaptive_pool_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}
static ops_probe_result supports_resize_nd(const ops_request& request) {
    const int spatial_dims = request.op_id == GGML_OP_OPS_VIRT_RESIZE_1D   ? 1
                             : request.op_id == GGML_OP_OPS_VIRT_RESIZE_2D ? 2
                                                                           : 3;
    if (!ops_validate_resize_nd_contract(request, spatial_dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static ops_probe_result supports_sample_dist(const ops_request& request) {
    if (!ops_validate_sample_dist(request)) {
        return false;
    }
    // The kernel bitonic-sorts the whole vocabulary in shared local memory in
    // one work-group; honestly reject sizes beyond the SLM budget.
    return request.srcs[0]->ne[0] <= ops_sycl_sample_dist_max_vocab;
}

static const ops_kernel_entry SYCL_KERNELS[] = {
    make_ops_kernel<ggml_sycl_op_conv_1d_entry>(GGML_OP_OPS_VIRT_CONV_1D, "sycl.conv1d",
                                                supports_conv, 100),
    make_ops_kernel<ggml_sycl_op_conv_transpose_1d_entry>(
        GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, "sycl.conv_transpose1d", supports_conv, 100),
    make_ops_kernel<ggml_sycl_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_2D, "sycl.conv2d",
                                                supports_conv_nd, 100),
    make_ops_kernel<ggml_sycl_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D,
                                                "sycl.conv_transpose2d", supports_conv_nd, 100),
    make_ops_kernel<ggml_sycl_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_3D, "sycl.conv3d",
                                                supports_conv_nd, 100),
    make_ops_kernel<ggml_sycl_op_conv_nd_entry>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D,
                                                "sycl.conv_transpose3d", supports_conv_nd, 100),
    make_ops_kernel<ggml_sycl_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_1D, "sycl.pool1d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_2D, "sycl.pool2d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_pool_nd_entry>(GGML_OP_OPS_VIRT_POOL_3D, "sycl.pool3d",
                                                supports_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_1D, "sycl.pad1d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_sycl_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_2D, "sycl.pad2d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_sycl_op_pad_nd_entry>(GGML_OP_OPS_VIRT_PAD_3D, "sycl.pad3d",
                                               supports_pad_nd, 100),
    make_ops_kernel<ggml_sycl_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D, "sycl.adaptive_pool1d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D, "sycl.adaptive_pool2d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_adaptive_pool_nd_entry>(
        GGML_OP_OPS_VIRT_ADAPTIVE_POOL_3D, "sycl.adaptive_pool3d", supports_adaptive_pool_nd, 100),
    make_ops_kernel<ggml_sycl_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_1D, "sycl.resize1d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_sycl_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_2D, "sycl.resize2d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_sycl_op_resize_nd_entry>(GGML_OP_OPS_VIRT_RESIZE_3D, "sycl.resize3d",
                                                  supports_resize_nd, 100),
    make_ops_kernel<ggml_sycl_op_mish_entry>(GGML_OP_OPS_VIRT_MISH, "sycl.mish", supports_standard,
                                             100),
    make_ops_kernel<ggml_sycl_op_gated_tanh_sigmoid_entry>(
        GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, "sycl.gated_tanh_sigmoid", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_layer_norm_entry>(GGML_OP_OPS_VIRT_LAYER_NORM, "sycl.layer_norm",
                                                   supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_double_swish_entry>(GGML_OP_OPS_VIRT_DOUBLE_SWISH,
                                                     "sycl.double_swish", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_attention_entry>(GGML_OP_OPS_VIRT_FUSED_ATTN, "sycl.attention",
                                                  supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_kv_cache_update_entry>(
        GGML_OP_OPS_VIRT_KV_CACHE_UPDATE, "sycl.kv_cache_update", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_glu_entry>(GGML_OP_OPS_VIRT_GLU, "sycl.glu", supports_standard,
                                            100),
    make_ops_kernel<ggml_sycl_op_gated_activation_entry>(GGML_OP_OPS_VIRT_GATED_ACTIVATION,
                                                         "sycl.gated_activation", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_relative_pe_keys_entry>(
        GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS, "sycl.relative_pe_keys", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_relative_pe_values_entry>(
        GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, "sycl.relative_pe_values", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_instance_norm_entry>(GGML_OP_OPS_VIRT_INSTANCE_NORM,
                                                      "sycl.instance_norm", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_snake_entry>(GGML_OP_OPS_VIRT_SNAKE, "sycl.snake",
                                              supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_snake_beta_entry>(GGML_OP_OPS_VIRT_SNAKE_BETA, "sycl.snake_beta",
                                                   supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_ada_ln_entry>(GGML_OP_OPS_VIRT_ADA_LN, "sycl.ada_ln",
                                               supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_alias_free_activation_entry>(
        GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION, "sycl.alias_free_activation", supports_standard,
        100),
    make_ops_kernel<ggml_sycl_op_fused_norm_act_entry>(
        GGML_OP_OPS_VIRT_FUSED_NORM_ACT, "sycl.fused_norm_act", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_pos_encoding_entry>(GGML_OP_OPS_VIRT_POS_ENCODING,
                                                     "sycl.pos_encoding", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_sample_dist_entry>(GGML_OP_OPS_VIRT_SAMPLE_DIST,
                                                    "sycl.sample_dist", supports_sample_dist, 100),
};

void register_backend() {
    resolve_bridge_sycl_functions();

    ops_backend_registration iface = {"SYCL", SYCL_KERNELS,
                                      sizeof(SYCL_KERNELS) / sizeof(SYCL_KERNELS[0])};
    register_ops_backend(iface);
}

struct RegisterSycl {
    RegisterSycl() { register_backend(); }
} g_register_sycl;

} // namespace sycl
} // namespace ggml_ops_ext

#ifdef _WIN32
extern "C" __declspec(dllexport) void ggml_ops_ext_sycl_init() {
#else
extern "C" void ggml_ops_ext_sycl_init() {
#endif
    // Force loading of DLL
}
