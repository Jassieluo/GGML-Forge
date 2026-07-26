#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_conv_transpose_1d(
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
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
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
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
            srcs,
            bias ? 3 : 2,
            &params,
            sizeof(params))) {
        const int64_t ne[4] = { out_w, weight_desc.output_channels, x->ne[2], 1 };

        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, x->type, 4, ne, bias ? 3 : 2, srcs);

        ggml_set_op_params(result, &params, sizeof(params));
        return result;
    }

    // A selected backend must execute the project implementation or fail.
    if (backend) return nullptr;

    // Reference path used only when no backend is selected. ggml's
    // conv_transpose_1d asserts dilation == 1 and has no grouped variant.
    if (dilation != 1 || groups != 1) return nullptr;
    struct ggml_tensor* w_f32 = w;
    if (w->type != GGML_TYPE_F32) {
        w_f32 = ggml_cast(ctx, w, GGML_TYPE_F32);
    }
    struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d(ctx, w_f32, x, stride, 0, dilation);
    struct ggml_tensor* out = conv_t_raw;
    if (padding > 0) {
        int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
        size_t offset_bytes = padding * conv_t_raw->nb[0];
        struct ggml_tensor* cropped_view = ggml_view_3d(
            ctx,
            conv_t_raw,
            cropped_seq_len,
            conv_t_raw->ne[1],
            conv_t_raw->ne[2],
            conv_t_raw->nb[1],
            conv_t_raw->nb[2],
            offset_bytes
        );
        out = ggml_cont(ctx, cropped_view);
    }
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        return ggml_add(ctx, out, ggml_repeat(ctx, b_reshaped, out));
    }
    return out;
}
