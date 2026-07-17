#include "ops/ops.h"
#include "ops_sycl.h"
#include <iostream>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// ────────────────────────────────────────────────────────────
// Bridge function pointer — resolved at runtime from ggml-sycl.dll
// ────────────────────────────────────────────────────────────

pfn_bridge_sycl_get_queue_t g_bridge_sycl_get_queue = nullptr;
pfn_bridge_sycl_dequantize_t g_bridge_sycl_dequantize = nullptr;
pfn_bridge_sycl_pool_alloc_t g_bridge_sycl_pool_alloc = nullptr;
pfn_bridge_sycl_pool_free_t g_bridge_sycl_pool_free = nullptr;

static void resolve_bridge_sycl_functions() {
    static std::once_flag resolve_once;
    std::call_once(resolve_once, []() {

#ifdef _WIN32
    HMODULE dll = GetModuleHandleW(L"ggml-sycl.dll");
    if (!dll) dll = GetModuleHandleW(L"ggml-sycl");
    if (!dll) dll = LoadLibraryW(L"ggml-sycl.dll");
    if (!dll) dll = LoadLibraryW(L"ggml-sycl");
    if (!dll) {
        std::cerr << "[SYCL Bridge] Error: Failed to load ggml-sycl.dll!\n";
        return;
    }

    g_bridge_sycl_get_queue = (pfn_bridge_sycl_get_queue_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_get_queue");
    g_bridge_sycl_dequantize = (pfn_bridge_sycl_dequantize_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_dequantize");
    g_bridge_sycl_pool_alloc = (pfn_bridge_sycl_pool_alloc_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_pool_alloc");
    g_bridge_sycl_pool_free = (pfn_bridge_sycl_pool_free_t)
        GetProcAddress(dll, "ggml_ops_ext_bridge_sycl_pool_free");
    if (!g_bridge_sycl_get_queue) {
        std::cerr << "[SYCL Bridge] Error: Failed to resolve ggml_ops_ext_bridge_sycl_get_queue from ggml-sycl.dll!\n";
    } else {
        std::cout << "[SYCL Bridge] Successfully loaded ggml-sycl.dll and resolved get_queue symbol!\n";
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
bool ggml_sycl_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_double_swish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_attention_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_glu_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_relative_pe_values_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_snake_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_snake_beta_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node);

static ops_probe_result supports_conv(
    const ops_request& request
) {
    if (!ops_validate_conv_request(request)) return false;
    const ops_quantization_desc weight = ops_describe_quantization(request.srcs[0], ops_weight_layout::channel_rows);
    const ggml_type x_type = request.srcs[1]->type;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] ||
                         request.srcs[2]->type == GGML_TYPE_F32 || request.srcs[2]->type == GGML_TYPE_F16;
    if (!bias_ok) return false;
    return (x_type == GGML_TYPE_F32 || x_type == GGML_TYPE_F16) &&
           (weight.storage_type == GGML_TYPE_F32 || weight.storage_type == GGML_TYPE_F16 ||
            weight.scheme == ops_quant_scheme::q4_0 ||
            weight.scheme == ops_quant_scheme::q8_0 ||
            weight.scheme == ops_quant_scheme::q4_k);
}

static ops_probe_result supports_standard(
    const ops_request& request
) {
    return ops_validate_request_contract(ops_support_profile::gpu, request);
}

static const ops_kernel_entry SYCL_KERNELS[] = {
    make_ops_kernel<ggml_sycl_op_conv_1d_entry>          (GGML_OP_OPS_VIRT_CONV_1D,           "sycl.conv1d",           supports_conv, 100),
    make_ops_kernel<ggml_sycl_op_conv_transpose_1d_entry>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, "sycl.conv_transpose1d", supports_conv, 100),
    make_ops_kernel<ggml_sycl_op_mish_entry>             (GGML_OP_OPS_VIRT_MISH,               "sycl.mish",               supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_gated_tanh_sigmoid_entry>(GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, "sycl.gated_tanh_sigmoid", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_layer_norm_entry>       (GGML_OP_OPS_VIRT_LAYER_NORM,         "sycl.layer_norm",         supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_double_swish_entry>     (GGML_OP_OPS_VIRT_DOUBLE_SWISH,       "sycl.double_swish",       supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_attention_entry>        (GGML_OP_OPS_VIRT_FUSED_ATTN,         "sycl.attention",          supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_glu_entry>              (GGML_OP_OPS_VIRT_GLU,                "sycl.glu",                supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_relative_pe_keys_entry> (GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS,   "sycl.relative_pe_keys",   supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_relative_pe_values_entry>(GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, "sycl.relative_pe_values", supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_instance_norm_entry>    (GGML_OP_OPS_VIRT_INSTANCE_NORM,      "sycl.instance_norm",      supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_snake_entry>            (GGML_OP_OPS_VIRT_SNAKE,              "sycl.snake",              supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_snake_beta_entry>       (GGML_OP_OPS_VIRT_SNAKE_BETA,         "sycl.snake_beta",         supports_standard, 100),
    make_ops_kernel<ggml_sycl_op_ada_ln_entry>           (GGML_OP_OPS_VIRT_ADA_LN,             "sycl.ada_ln",             supports_standard, 100),
};

void register_backend() {
    resolve_bridge_sycl_functions();

    ops_backend_registration iface = {
        "SYCL", SYCL_KERNELS, sizeof(SYCL_KERNELS) / sizeof(SYCL_KERNELS[0])
    };
    register_ops_backend(iface);
}

struct RegisterSycl {
    RegisterSycl() {
        register_backend();
    }
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
