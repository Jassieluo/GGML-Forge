#include "ops/ops.h"
#include "ggml-impl.h"

struct ggml_tensor* ggml_ops_glu(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    struct ggml_tensor* srcs[] = { x };
    // 2. Check if backend registers virtual node support
    if (ggml_ops_backend_supports_op(backend, ggml_ops_ext::GGML_OP_OPS_VIRT_GLU, srcs, 1)) {
        // Output size along dimension 0 is halved: C = x->ne[0] / 2
        int64_t ne[GGML_MAX_DIMS] = { x->ne[0] / 2, x->ne[1], x->ne[2], x->ne[3] };
        struct ggml_tensor* result = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_GLU, x->type, ggml_n_dims(x), ne, 1, srcs);
        return result;
    }

    // 3. Fallback: split and execute GLU subgraph
    int64_t in_ch = x->ne[0] / 2;
    struct ggml_tensor* x1 = ggml_view_2d(ctx, x, in_ch, x->ne[1], x->nb[1], 0);
    struct ggml_tensor* x2 = ggml_view_2d(ctx, x, in_ch, x->ne[1], x->nb[1], in_ch * ggml_element_size(x));

    struct ggml_tensor* glu = ggml_mul(ctx, x1, ggml_sigmoid(ctx, x2));
    return glu;
}
