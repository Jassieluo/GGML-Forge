#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend,
    struct ggml_tensor* bias
) {
    struct ggml_tensor* srcs[] = { w, x, bias };
    const ggml_ops_ext::ops_conv_1d_contract_params params = {
        stride, padding, dilation, groups
    };
    int64_t out_w = 0;
    ggml_ops_ext::ops_conv_weight_desc weight_desc = {};
    if (!ggml_ops_ext::ops_validate_conv_contract(
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D,
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D,
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
            srcs,
            bias ? 3 : 2,
            &params,
            sizeof(params),
            &out_w,
            &weight_desc)) {
        return nullptr;
    }

    // Build a semantic node only when the selected backend has a compatible kernel.
    if (ggml_ops_backend_supports_op(
            backend,
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D,
            srcs,
            bias ? 3 : 2,
            &params,
            sizeof(params))) {
        const int64_t ne[4] = { out_w, weight_desc.output_channels, x->ne[2], 1 };

        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, x->type, 4, ne, bias ? 3 : 2, srcs);

        ggml_set_op_params(result, &params, sizeof(params));
        return result;
    }

    // A selected backend must execute the project implementation or fail.
    if (backend) return nullptr;

    // Reference path used only when no backend is selected.
    struct ggml_tensor* conv = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    if (bias) {
        struct ggml_tensor* conv_reshaped = ggml_reshape_3d(ctx, conv, conv->ne[0], conv->ne[2], conv->ne[1]);
        struct ggml_tensor* b_reshaped = ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]);
        struct ggml_tensor* added = ggml_add(ctx, conv_reshaped, ggml_repeat(ctx, b_reshaped, conv_reshaped));
        return ggml_reshape_3d(ctx, added, conv->ne[0], conv->ne[1], conv->ne[2]);
    }
    return conv;
}
