#include "nn/layers/embedding.h"
#include "ops/ops.h"

namespace nn {

struct ggml_tensor* Embedding::forward(struct ggml_context* ctx,
                                       struct ggml_tensor* input_ids) {
    return ggml_ops_embedding(ctx, weight.tensor(), input_ids);
}

} // namespace nn
