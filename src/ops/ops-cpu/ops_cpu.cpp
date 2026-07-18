#include "ops/ops.h"
#define GGML_OPS_EXT_CPU_API extern "C" __declspec(dllexport)
#include "ops/cpu.h"
#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ggml_ops_ext {
namespace cpu {

using thread_count_map = std::unordered_map<ggml_backend_t, int>;

static std::mutex g_thread_counts_write_mutex;
static std::shared_ptr<const thread_count_map> g_thread_counts =
    std::make_shared<const thread_count_map>();

int backend_thread_count(ggml_backend_t backend) {
    const auto snapshot = std::atomic_load_explicit(&g_thread_counts, std::memory_order_acquire);
    const auto it = snapshot->find(backend);
    if (it != snapshot->end()) return it->second;
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads > 0 ? static_cast<int>(hardware_threads) : 1;
}

static void set_backend_thread_count(ggml_backend_t backend, int n_threads) {
    if (!backend || n_threads < 1) return;
    std::lock_guard<std::mutex> lock(g_thread_counts_write_mutex);
    const auto current = std::atomic_load_explicit(&g_thread_counts, std::memory_order_acquire);
    auto next = std::make_shared<thread_count_map>(*current);
    (*next)[backend] = n_threads;
    std::atomic_store_explicit(
        &g_thread_counts,
        std::static_pointer_cast<const thread_count_map>(next),
        std::memory_order_release);
}

bool ops_cpu_op_conv_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_conv_transpose_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_mish(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_gated_tanh_sigmoid(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_double_swish(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_attention(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_glu(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_relative_pe_keys(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_relative_pe_values(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_instance_norm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_snake(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_snake_beta(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_ada_ln(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_kv_cache_update(ggml_backend_t backend, struct ggml_tensor* node);

static ops_probe_result supports_conv(
    const ops_request& request
) {
    if (!ops_validate_conv_request(request)) return false;
    const ops_quantization_desc weight = ops_describe_quantization(request.srcs[0], ops_weight_layout::channel_rows);
    const ggml_type x_type = request.srcs[1]->type;
    const bool weight_ok = weight.storage_type == GGML_TYPE_F32 || weight.storage_type == GGML_TYPE_F16 ||
                           weight.scheme == ops_quant_scheme::q4_0 ||
                           weight.scheme == ops_quant_scheme::q8_0 ||
                           weight.scheme == ops_quant_scheme::q4_k;
    const bool activation_ok = x_type == GGML_TYPE_F32 || x_type == GGML_TYPE_F16;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] ||
                         request.srcs[2]->type == GGML_TYPE_F32 || request.srcs[2]->type == GGML_TYPE_F16;
    if (!weight_ok || !activation_ok || !bias_ok) return false;

    return true;
}

static ops_probe_result supports_standard(
    const ops_request& request
) {
    return ops_validate_request_contract(ops_support_profile::cpu, request);
}

static const ops_kernel_entry CPU_KERNELS[] = {
    make_ops_kernel<ops_cpu_op_conv_1d>          (GGML_OP_OPS_VIRT_CONV_1D,           "cpu.conv1d",           supports_conv, 100),
    make_ops_kernel<ops_cpu_op_conv_transpose_1d>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, "cpu.conv_transpose1d", supports_conv, 100),

    make_ops_kernel<ops_cpu_op_mish>             (GGML_OP_OPS_VIRT_MISH,               "cpu.mish",               supports_standard, 100),
    make_ops_kernel<ops_cpu_op_gated_tanh_sigmoid>(GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, "cpu.gated_tanh_sigmoid", supports_standard, 100),
    // { GGML_OP_OPS_VIRT_LAYER_NORM,         ops_cpu_op_layer_norm },
    make_ops_kernel<ops_cpu_op_double_swish>     (GGML_OP_OPS_VIRT_DOUBLE_SWISH,       "cpu.double_swish",       supports_standard, 100),
    make_ops_kernel<ops_cpu_op_attention>        (GGML_OP_OPS_VIRT_FUSED_ATTN,         "cpu.attention",          supports_standard, 100),
    make_ops_kernel<ops_cpu_op_glu>              (GGML_OP_OPS_VIRT_GLU,                "cpu.glu",                supports_standard, 100),
    make_ops_kernel<ops_cpu_op_relative_pe_keys> (GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS,   "cpu.relative_pe_keys",   supports_standard, 100),
    make_ops_kernel<ops_cpu_op_relative_pe_values>(GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, "cpu.relative_pe_values", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_instance_norm>    (GGML_OP_OPS_VIRT_INSTANCE_NORM,      "cpu.instance_norm",      supports_standard, 100),
    make_ops_kernel<ops_cpu_op_snake>            (GGML_OP_OPS_VIRT_SNAKE,              "cpu.snake",              supports_standard, 100),
    make_ops_kernel<ops_cpu_op_snake_beta>       (GGML_OP_OPS_VIRT_SNAKE_BETA,         "cpu.snake_beta",         supports_standard, 100),
    make_ops_kernel<ops_cpu_op_ada_ln>           (GGML_OP_OPS_VIRT_ADA_LN,             "cpu.ada_ln",             supports_standard, 100),
    make_ops_kernel<ops_cpu_op_kv_cache_update>  (GGML_OP_OPS_VIRT_KV_CACHE_UPDATE,    "cpu.kv_cache_update",    supports_standard, 100),
};

void register_backend() {
    ops_backend_registration iface = {
        "CPU", CPU_KERNELS, sizeof(CPU_KERNELS) / sizeof(CPU_KERNELS[0])
    };
    register_ops_backend(iface);

    ops_backend_registration iface_blas = {
        "BLAS", CPU_KERNELS, sizeof(CPU_KERNELS) / sizeof(CPU_KERNELS[0])
    };
    register_ops_backend(iface_blas);
}

struct RegisterCpu {
    RegisterCpu() {
        register_backend();
    }
} g_register_cpu;

} // namespace cpu
} // namespace ggml_ops_ext

GGML_OPS_EXT_CPU_API void ggml_ops_ext_cpu_set_n_threads(ggml_backend_t backend, int n_threads) {
    ggml_ops_ext::cpu::set_backend_thread_count(backend, n_threads);
}

#ifdef _WIN32
extern "C" __declspec(dllexport) void ggml_ops_ext_cpu_init() {
#else
extern "C" void ggml_ops_ext_cpu_init() {
#endif
    // Force loading of DLL
}
