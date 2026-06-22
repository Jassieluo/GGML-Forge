#include "ops/ops.h"
#include "ops_sycl.h"

namespace ggml_ops_ext {
namespace sycl {

// Declarations of builders and entrypoints implemented in other files
struct ggml_tensor* sycl_conv_1d_builder(
    struct ggml_context* ctx,
    int op_id,
    struct ggml_tensor** srcs,
    int n_srcs,
    const int32_t* params,
    int n_params,
    ggml_backend_t backend
);

bool ggml_sycl_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_softmax(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_mish_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_gated_tanh_sigmoid_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node);
bool ggml_sycl_op_double_swish_entry(ggml_backend_t backend, struct ggml_tensor* node);

static const ops_handler_entry SYCL_HANDLERS[] = {
    { GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, ggml_sycl_op_conv_transpose_1d_entry },
    { GGML_OP_OPS_VIRT_SOFTMAX,           ggml_sycl_op_softmax },
    { GGML_OP_OPS_VIRT_MISH,               ggml_sycl_op_mish_entry },
    { GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID, ggml_sycl_op_gated_tanh_sigmoid_entry },
    { GGML_OP_OPS_VIRT_LAYER_NORM,         ggml_sycl_op_layer_norm_entry },
    { GGML_OP_OPS_VIRT_DOUBLE_SWISH,       ggml_sycl_op_double_swish_entry },
};

static const ops_builder_entry SYCL_BUILDERS[] = {
    { GGML_OP_OPS_VIRT_CONV_1D,           sycl_conv_1d_builder },
};

void register_backend() {
    ops_backend_interface iface = {
        /* backend_name_prefix */ "SYCL",
        /* handlers            */ SYCL_HANDLERS,
        /* n_handlers          */ sizeof(SYCL_HANDLERS) / sizeof(SYCL_HANDLERS[0]),
        /* builders            */ SYCL_BUILDERS,
        /* n_builders          */ sizeof(SYCL_BUILDERS) / sizeof(SYCL_BUILDERS[0])
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
