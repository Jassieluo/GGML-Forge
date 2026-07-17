#include "nn/nn.h"

namespace nn {


struct ggml_tensor* Embedding::forward(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
    return ggml_get_rows(ctx, weight.tensor(), input_ids);
}

} // namespace nn
