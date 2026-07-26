#include "nn/layers/embedding.h"
#include "nn/core/context.h"
#include "ops/ops.h"

namespace nn {

struct ggml_tensor* Embedding::forward(Context& context,
                                       struct ggml_tensor* input_ids) {
    ggml_context* ctx = context.native_handle();
    return ggml_ops_embedding(ctx, weight.tensor(), input_ids);
}

} // namespace nn
