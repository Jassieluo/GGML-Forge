#include "ops/ops.h"
#include <cstdio>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_conv_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_conv_transpose_1d(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_mish(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_gated_tanh_sigmoid(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* node);
bool ops_cpu_op_double_swish(ggml_backend_t backend, struct ggml_tensor* node);

static const ops_handler_entry CPU_HANDLERS[] = {
    // ── Conv1D ──
    // 已回退到 ggml 原生实现（im2col + mul_mat）。
    // 我们的 custom Direct Conv 方案（ops_cpu_op_conv_1d）因性能不及 ggml 原生已临时禁用。
    // 详见 docs/cpu_conv_optimization.md
    // { GGML_OP_OPS_VIRT_CONV_1D,            ops_cpu_op_conv_1d },

    { GGML_OP_OPS_VIRT_MISH,               ops_cpu_op_mish },
    { GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, ops_cpu_op_gated_tanh_sigmoid },
    // { GGML_OP_OPS_VIRT_LAYER_NORM,         ops_cpu_op_layer_norm },
    { GGML_OP_OPS_VIRT_DOUBLE_SWISH,       ops_cpu_op_double_swish },
};

void register_backend() {
    ops_backend_interface iface = {
        /* backend_name_prefix */ "CPU",
        /* handlers            */ CPU_HANDLERS,
        /* n_handlers          */ sizeof(CPU_HANDLERS) / sizeof(CPU_HANDLERS[0]),
        /* builders            */ nullptr,
        /* n_builders          */ 0
    };
    register_ops_backend(iface);
}

struct RegisterCpu {
    RegisterCpu() {
        register_backend();
    }
} g_register_cpu;

} // namespace cpu
} // namespace ggml_ops_ext

#ifdef _WIN32
extern "C" __declspec(dllexport) void ggml_ops_ext_cpu_init() {
#else
extern "C" void ggml_ops_ext_cpu_init() {
#endif
    // Force loading of DLL
}
