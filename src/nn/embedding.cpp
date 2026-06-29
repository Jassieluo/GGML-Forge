#include "nn/nn.h"

namespace nn {

Embedding::Embedding(struct ggml_tensor* w)
    : weight(w) {}

struct ggml_tensor* Embedding::forward(struct ggml_context* ctx, struct ggml_tensor* input_ids) {
    return ggml_get_rows(ctx, weight, input_ids);
}

} // namespace nn
