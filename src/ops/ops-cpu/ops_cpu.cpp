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
static std::shared_ptr<const thread_count_map> g_thread_counts = std::make_shared<const thread_count_map>();

int backend_thread_count(ggml_backend_t backend) {
    const auto snapshot = std::atomic_load_explicit(&g_thread_counts, std::memory_order_acquire);
    const auto it = snapshot->find(backend);
    if (it != snapshot->end()) {
        return it->second;
    }
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads > 0 ? static_cast<int>(hardware_threads) : 1;
}

static void set_backend_thread_count(ggml_backend_t backend, int n_threads) {
    if (!backend || n_threads < 1) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_thread_counts_write_mutex);
    const auto current = std::atomic_load_explicit(&g_thread_counts, std::memory_order_acquire);
    auto next = std::make_shared<thread_count_map>(*current);
    (*next)[backend] = n_threads;
    std::atomic_store_explicit(&g_thread_counts, std::static_pointer_cast<const thread_count_map>(next),
                               std::memory_order_release);
}

bool ops_cpu_op_conv_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_conv_transpose_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_conv_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_pool_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_pad_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_adaptive_pool_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_resize_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_mish(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_gated_tanh_sigmoid(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_double_swish(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_attention(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_glu(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_gated_activation(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_alias_free_activation(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_relative_pe_keys(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_relative_pe_values(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_instance_norm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_snake(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_snake_beta(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_ada_ln(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_kv_cache_update(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_fused_norm_act(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_pos_encoding(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_sample_dist(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_stft(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_istft(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_fft(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_complex_abs(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_length_regulate(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_gru(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_lstm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_reduce_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_arg_reduce_nd(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_selection(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_grid_sample_2d(ggml_backend_t backend, struct ggml_tensor* node);

static ops_probe_result supports_conv(const ops_request& request) {
    if (!ops_validate_conv_request(request)) {
        return false;
    }
    const ops_quantization_desc weight = ops_describe_quantization(request.srcs[0], ops_weight_layout::channel_rows);
    const ggml_type x_type = request.srcs[1]->type;
    const bool weight_ok = weight.storage_type == GGML_TYPE_F32 || weight.storage_type == GGML_TYPE_F16 ||
                           weight.scheme == ops_quant_scheme::q4_0 || weight.scheme == ops_quant_scheme::q8_0 ||
                           weight.scheme == ops_quant_scheme::q4_k;
    const bool activation_ok = x_type == GGML_TYPE_F32 || x_type == GGML_TYPE_F16;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] || request.srcs[2]->type == GGML_TYPE_F32 ||
                         request.srcs[2]->type == GGML_TYPE_F16;
    if (!weight_ok || !activation_ok || !bias_ok) {
        return false;
    }

    // Quantized rows must be whole blocks: conv_1d reduces over C_in/groups,
    // conv_transpose_1d over C_out/groups. Misalignment would trip
    // ggml_row_size's GGML_ASSERT (process abort) or over-read the 32-wide
    // microtiles, so reject it here instead.
    if (ggml_is_quantized(request.srcs[0]->type)) {
        ops_conv_1d_contract_params params;
        std::memcpy(&params, request.params, sizeof(params));
        ops_conv_weight_desc desc;
        if (!ops_describe_conv_weight(request.op_id, request.srcs[0], request.srcs[1],
                                      params.groups, desc)) {
            return false;
        }
        const int64_t block_size = ggml_blck_size(request.srcs[0]->type);
        const int64_t row_length = request.op_id == GGML_OP_OPS_VIRT_CONV_1D
                                       ? desc.input_channels_per_group
                                       : desc.output_channels_per_group;
        if (block_size <= 0 || row_length % block_size != 0) {
            return false;
        }
    }

    return true;
}

static ops_probe_result supports_standard(const ops_request& request) {
    return ops_validate_request_contract(ops_support_profile::cpu, request);
}

static ops_probe_result supports_conv_nd(const ops_request& request) {
    if (!ops_validate_conv_request(request)) {
        return false;
    }
    const ggml_tensor* weight = request.srcs[0];
    const ggml_tensor* input = request.srcs[1];
    const ggml_type_traits* traits = ggml_get_type_traits(weight->type);
    const bool weight_ok =
        weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_BF16 ||
        (ggml_is_quantized(weight->type) && traits && traits->to_float && weight->ne[0] % traits->blck_size == 0);
    const bool activation_ok =
        input->type == GGML_TYPE_F32 || input->type == GGML_TYPE_F16 || input->type == GGML_TYPE_BF16;
    const bool bias_ok = request.n_srcs < 3 || !request.srcs[2] || request.srcs[2]->type == GGML_TYPE_F32 ||
                         request.srcs[2]->type == GGML_TYPE_F16 || request.srcs[2]->type == GGML_TYPE_BF16;
    return weight_ok && activation_ok && bias_ok;
}

static ops_probe_result supports_pool_nd(const ops_request& request) {
    const int dims = request.op_id == GGML_OP_OPS_VIRT_POOL_1D ? 1 : request.op_id == GGML_OP_OPS_VIRT_POOL_2D ? 2 : 3;
    if (!ops_validate_pool_nd_contract(request, dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static ops_probe_result supports_pad_nd(const ops_request& request) {
    const int dims = request.op_id == GGML_OP_OPS_VIRT_PAD_1D ? 1 : request.op_id == GGML_OP_OPS_VIRT_PAD_2D ? 2 : 3;
    if (!ops_validate_pad_nd_contract(request, dims)) {
        return false;
    }
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static ops_probe_result supports_adaptive_pool_nd(const ops_request& request) {
    const int dims = request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                     : request.op_id == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                          : 3;
    if (!ops_validate_adaptive_pool_nd_contract(request, dims)) {
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

static ops_probe_result supports_reduce_nd(const ops_request& request) {
    const bool valid = request.op_id == GGML_OP_OPS_VIRT_ARG_REDUCE_ND
                           ? static_cast<bool>(ops_validate_arg_reduce_nd_contract(request))
                           : static_cast<bool>(ops_validate_reduce_nd_contract(request));
    if (!valid) return false;
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

static ops_probe_result supports_selection(const ops_request& request) {
    const bool valid = request.op_id == GGML_OP_OPS_VIRT_COMPARE ? static_cast<bool>(ops_validate_compare(request))
                     : request.op_id == GGML_OP_OPS_VIRT_LOGICAL ? static_cast<bool>(ops_validate_logical(request))
                                                                 : static_cast<bool>(ops_validate_where(request));
    if (!valid) return false;
    const ggml_type type = request.op_id == GGML_OP_OPS_VIRT_WHERE ? request.srcs[1]->type : request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_I32;
}
static ops_probe_result supports_grid_sample(const ops_request& request) {
    if (!ops_validate_grid_sample_2d(request)) return false;
    const ggml_type input = request.srcs[0]->type, grid = request.srcs[1]->type;
    return (input == GGML_TYPE_F32 || input == GGML_TYPE_F16 || input == GGML_TYPE_BF16) &&
           (grid == GGML_TYPE_F32 || grid == GGML_TYPE_F16);
}

static const ops_kernel_entry CPU_KERNELS[] = {
    make_ops_kernel<ops_cpu_op_conv_1d>(GGML_OP_OPS_VIRT_CONV_1D, "cpu.conv1d", supports_conv, 100),
    make_ops_kernel<ops_cpu_op_conv_transpose_1d>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, "cpu.conv_transpose1d",
                                                  supports_conv, 100),
    make_ops_kernel<ops_cpu_op_conv_nd>(GGML_OP_OPS_VIRT_CONV_2D, "cpu.conv2d", supports_conv_nd, 100),
    make_ops_kernel<ops_cpu_op_conv_nd>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D, "cpu.conv_transpose2d", supports_conv_nd,
                                        100),
    make_ops_kernel<ops_cpu_op_conv_nd>(GGML_OP_OPS_VIRT_CONV_3D, "cpu.conv3d", supports_conv_nd, 100),
    make_ops_kernel<ops_cpu_op_conv_nd>(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D, "cpu.conv_transpose3d", supports_conv_nd,
                                        100),
    make_ops_kernel<ops_cpu_op_pool_nd>(GGML_OP_OPS_VIRT_POOL_1D, "cpu.pool1d", supports_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_pool_nd>(GGML_OP_OPS_VIRT_POOL_2D, "cpu.pool2d", supports_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_pool_nd>(GGML_OP_OPS_VIRT_POOL_3D, "cpu.pool3d", supports_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_pad_nd>(GGML_OP_OPS_VIRT_PAD_1D, "cpu.pad1d", supports_pad_nd, 100),
    make_ops_kernel<ops_cpu_op_pad_nd>(GGML_OP_OPS_VIRT_PAD_2D, "cpu.pad2d", supports_pad_nd, 100),
    make_ops_kernel<ops_cpu_op_pad_nd>(GGML_OP_OPS_VIRT_PAD_3D, "cpu.pad3d", supports_pad_nd, 100),
    make_ops_kernel<ops_cpu_op_adaptive_pool_nd>(GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D, "cpu.adaptive_pool1d",
                                                 supports_adaptive_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_adaptive_pool_nd>(GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D, "cpu.adaptive_pool2d",
                                                 supports_adaptive_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_adaptive_pool_nd>(GGML_OP_OPS_VIRT_ADAPTIVE_POOL_3D, "cpu.adaptive_pool3d",
                                                 supports_adaptive_pool_nd, 100),
    make_ops_kernel<ops_cpu_op_resize_nd>(GGML_OP_OPS_VIRT_RESIZE_1D, "cpu.resize1d", supports_resize_nd, 100),
    make_ops_kernel<ops_cpu_op_resize_nd>(GGML_OP_OPS_VIRT_RESIZE_2D, "cpu.resize2d", supports_resize_nd, 100),
    make_ops_kernel<ops_cpu_op_resize_nd>(GGML_OP_OPS_VIRT_RESIZE_3D, "cpu.resize3d", supports_resize_nd, 100),
    make_ops_kernel<ops_cpu_op_reduce_nd>(GGML_OP_OPS_VIRT_REDUCE_ND, "cpu.reduce_nd",
                                          supports_reduce_nd, 100),
    make_ops_kernel<ops_cpu_op_arg_reduce_nd>(GGML_OP_OPS_VIRT_ARG_REDUCE_ND, "cpu.arg_reduce_nd",
                                              supports_reduce_nd, 100),
    make_ops_kernel<ops_cpu_op_selection>(GGML_OP_OPS_VIRT_COMPARE, "cpu.compare", supports_selection, 100),
    make_ops_kernel<ops_cpu_op_selection>(GGML_OP_OPS_VIRT_LOGICAL, "cpu.logical", supports_selection, 100),
    make_ops_kernel<ops_cpu_op_selection>(GGML_OP_OPS_VIRT_WHERE, "cpu.where", supports_selection, 100),
    make_ops_kernel<ops_cpu_op_grid_sample_2d>(GGML_OP_OPS_VIRT_GRID_SAMPLE_2D, "cpu.grid_sample2d",
                                               supports_grid_sample, 100),

    make_ops_kernel<ops_cpu_op_mish>(GGML_OP_OPS_VIRT_MISH, "cpu.mish", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_gated_tanh_sigmoid>(GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, "cpu.gated_tanh_sigmoid",
                                                   supports_standard, 100),
    make_ops_kernel<ops_cpu_op_layer_norm>(GGML_OP_OPS_VIRT_LAYER_NORM, "cpu.layer_norm", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_double_swish>(GGML_OP_OPS_VIRT_DOUBLE_SWISH, "cpu.double_swish", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_attention>(GGML_OP_OPS_VIRT_FUSED_ATTN, "cpu.attention", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_glu>(GGML_OP_OPS_VIRT_GLU, "cpu.glu", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_gated_activation>(GGML_OP_OPS_VIRT_GATED_ACTIVATION,
                                                 "cpu.gated_activation", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_alias_free_activation>(
        GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION, "cpu.alias_free_activation", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_relative_pe_keys>(GGML_OP_OPS_VIRT_RELATIVE_PE_KEYS, "cpu.relative_pe_keys",
                                                 supports_standard, 100),
    make_ops_kernel<ops_cpu_op_relative_pe_values>(GGML_OP_OPS_VIRT_RELATIVE_PE_VALUES, "cpu.relative_pe_values",
                                                   supports_standard, 100),
    make_ops_kernel<ops_cpu_op_instance_norm>(GGML_OP_OPS_VIRT_INSTANCE_NORM, "cpu.instance_norm", supports_standard,
                                              100),
    make_ops_kernel<ops_cpu_op_snake>(GGML_OP_OPS_VIRT_SNAKE, "cpu.snake", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_snake_beta>(GGML_OP_OPS_VIRT_SNAKE_BETA, "cpu.snake_beta", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_ada_ln>(GGML_OP_OPS_VIRT_ADA_LN, "cpu.ada_ln", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_kv_cache_update>(GGML_OP_OPS_VIRT_KV_CACHE_UPDATE, "cpu.kv_cache_update",
                                                supports_standard, 100),
    make_ops_kernel<ops_cpu_op_fused_norm_act>(GGML_OP_OPS_VIRT_FUSED_NORM_ACT, "cpu.fused_norm_act",
                                               supports_standard, 100),
    make_ops_kernel<ops_cpu_op_pos_encoding>(GGML_OP_OPS_VIRT_POS_ENCODING, "cpu.pos_encoding",
                                             supports_standard, 100),
    make_ops_kernel<ops_cpu_op_sample_dist>(GGML_OP_OPS_VIRT_SAMPLE_DIST, "cpu.sample_dist",
                                            supports_standard, 100),
    make_ops_kernel<ops_cpu_op_stft>(GGML_OP_OPS_VIRT_STFT, "cpu.stft", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_istft>(GGML_OP_OPS_VIRT_ISTFT, "cpu.istft", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_fft>(GGML_OP_OPS_VIRT_FFT, "cpu.fft", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_complex_abs>(GGML_OP_OPS_VIRT_COMPLEX_ABS, "cpu.complex_abs",
                                            supports_standard, 100),
    make_ops_kernel<ops_cpu_op_length_regulate>(GGML_OP_OPS_VIRT_LENGTH_REGULATE,
                                                "cpu.length_regulate", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_gru>(GGML_OP_OPS_VIRT_GRU, "cpu.gru", supports_standard, 100),
    make_ops_kernel<ops_cpu_op_lstm>(GGML_OP_OPS_VIRT_LSTM, "cpu.lstm", supports_standard, 100),
};

void register_backend() {
    ops_backend_registration iface = {"CPU", CPU_KERNELS, sizeof(CPU_KERNELS) / sizeof(CPU_KERNELS[0])};
    register_ops_backend(iface);

    ops_backend_registration iface_blas = {"BLAS", CPU_KERNELS, sizeof(CPU_KERNELS) / sizeof(CPU_KERNELS[0])};
    register_ops_backend(iface_blas);
}

struct RegisterCpu {
    RegisterCpu() { register_backend(); }
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
