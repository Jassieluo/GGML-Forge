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
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D);
    if (builder) {
        struct ggml_tensor* srcs[] = { w, x, bias };
        int32_t params[] = { stride, padding, dilation, groups };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, srcs, bias ? 3 : 2, params, 4, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D)) {
        int64_t out_w = (x->ne[0] - 1) * stride - 2 * padding + dilation * (w->ne[0] - 1) + 1;
        const int64_t ne[4] = { out_w, w->ne[1] * groups, x->ne[2], 1 };

        struct ggml_tensor* srcs[] = { w, x, bias };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, x->type, 4, ne, bias ? 3 : 2, srcs);

        int32_t params[] = { stride, padding, dilation, groups };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback
    struct ggml_tensor* w_f32 = w;
    if (w->type != GGML_TYPE_F32) {
        w_f32 = ggml_cast(ctx, w, GGML_TYPE_F32);
    }
    struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d(ctx, w_f32, x, stride, 0, dilation);
    struct ggml_tensor* out = conv_t_raw;
    if (padding > 0) {
        int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
        size_t offset_bytes = padding * conv_t_raw->nb[0];
        struct ggml_tensor* cropped_view = ggml_view_2d(
            ctx,
            conv_t_raw,
            cropped_seq_len,
            conv_t_raw->ne[1],
            conv_t_raw->nb[1],
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
