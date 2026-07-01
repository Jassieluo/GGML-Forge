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
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D);
    if (builder) {
        struct ggml_tensor* srcs[] = { w, x, bias };
        int32_t params[] = { stride, padding, dilation, groups };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, srcs, bias ? 3 : 2, params, 4, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D)) {
        int64_t out_w = (x->ne[0] + 2 * padding - dilation * (w->ne[0] - 1) - 1) / stride + 1;
        const int64_t ne[4] = { out_w, w->ne[2], x->ne[2], 1 };

        struct ggml_tensor* srcs[] = { w, x, bias };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, x->type, 4, ne, bias ? 3 : 2, srcs);

        int32_t params[] = { stride, padding, dilation, groups };
        ggml_set_op_params(result, params, sizeof(params));
        return result;
    }

    // 3. Default fallback
    struct ggml_tensor* conv = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    if (bias) {
        struct ggml_tensor* conv_reshaped = ggml_reshape_3d(ctx, conv, conv->ne[0], conv->ne[2], conv->ne[1]);
        struct ggml_tensor* b_reshaped = ggml_reshape_3d(ctx, bias, 1, 1, bias->ne[0]);
        struct ggml_tensor* added = ggml_add(ctx, conv_reshaped, ggml_repeat(ctx, b_reshaped, conv_reshaped));
        return ggml_reshape_3d(ctx, added, conv->ne[0], conv->ne[1], conv->ne[2]);
    }
    return conv;
}
