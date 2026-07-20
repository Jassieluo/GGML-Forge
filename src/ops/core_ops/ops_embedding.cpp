#include "ops/ops.h"

ggml_tensor* ggml_ops_embedding(ggml_context* ctx, ggml_tensor* weight,
                                ggml_tensor* indices) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(weight != nullptr);
    GGML_ASSERT(indices != nullptr);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(weight->ne[2] == 1 && weight->ne[3] == 1);
    GGML_ASSERT(indices->ne[3] == 1);

    ggml_tensor* flat_indices =
        ggml_reshape_1d(ctx, ggml_cont(ctx, indices), ggml_nelements(indices));
    ggml_tensor* rows = ggml_get_rows(ctx, weight, flat_indices);
    return ggml_reshape_4d(ctx, rows, weight->ne[0], indices->ne[0],
                           indices->ne[1], indices->ne[2]);
}
