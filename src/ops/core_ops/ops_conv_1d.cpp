#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_conv_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation,
    ggml_backend_t backend
) {
    // 1. Check if the backend registers a custom builder
    ggml_ops_ext::ops_op_builder_t builder = ggml_ops_ext::find_ops_builder(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D);
    if (builder) {
        struct ggml_tensor* srcs[] = { w, x };
        int32_t params[] = { stride, padding, dilation };
        return builder(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, srcs, 2, params, 3, backend);
    }

    // 2. Otherwise check if it supports direct handler execution (virtual node)
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D)) {
        int64_t out_w = (x->ne[0] + 2 * padding - dilation * (w->ne[0] - 1) - 1) / stride + 1;
        const int64_t ne[4] = { out_w, w->ne[2], x->ne[2], 1 };
        
        int kW = w->ne[0];
        if (kW > 1 || stride > 1 || padding > 0 || dilation > 1) {
            int64_t col_elements = x->ne[2] * w->ne[1] * w->ne[0] * out_w; // batch * C_in * kW * out_w
            struct ggml_tensor* col_buf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, col_elements);
            struct ggml_tensor* srcs[] = { w, x, col_buf };
            struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, GGML_TYPE_F32, 4, ne, 3, srcs);
            
            int32_t params[] = { stride, padding, dilation };
            ggml_set_op_params(result, params, sizeof(params));
            return result;
        } else {
            struct ggml_tensor* srcs[] = { w, x };
            struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, GGML_TYPE_F32, 4, ne, 2, srcs);
            
            int32_t params[] = { stride, padding, dilation };
            ggml_set_op_params(result, params, sizeof(params));
            return result;
        }
    }

    // 3. Default fallback
    return ggml_conv_1d(ctx, w, x, stride, padding, dilation);
}
