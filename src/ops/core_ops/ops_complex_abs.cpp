#include "ops/ops.h"
#include "ggml-impl.h"
#include <cstring>

struct ggml_tensor* ggml_ops_complex_abs(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    bool squared,
    ggml_backend_t backend
) {
    if (!x || x->ne[2] != 2) return nullptr;
    ggml_ops_ext::ops_complex_abs_params params;
    params.squared = squared ? 1 : 0;
    struct ggml_tensor* srcs[] = { x };

    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_COMPLEX_ABS,
                                     srcs, 1, &params, sizeof(params))) {
        const int64_t ne[4] = { x->ne[0], x->ne[1], 1, x->ne[3] };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_COMPLEX_ABS, GGML_TYPE_F32, 4, ne, 1, srcs);
        ggml_set_op_params(result, &params, sizeof(params));
        return result;
    }

    // Native fallback: slice the real/imag planes and compose re^2 + im^2.
    struct ggml_tensor* re = ggml_cont(ctx, ggml_view_4d(
        ctx, x, x->ne[0], x->ne[1], 1, x->ne[3], x->nb[1], x->nb[2], x->nb[3], 0));
    struct ggml_tensor* im = ggml_cont(ctx, ggml_view_4d(
        ctx, x, x->ne[0], x->ne[1], 1, x->ne[3], x->nb[1], x->nb[2], x->nb[3], x->nb[2]));
    struct ggml_tensor* power = ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im));
    return squared ? power : ggml_sqrt(ctx, power);
}
