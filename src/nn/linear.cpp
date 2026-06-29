#include "nn/nn.h"

namespace nn {

Linear::Linear(struct ggml_tensor* w, struct ggml_tensor* b)
    : weight(w), bias(b) {}

struct ggml_tensor* Linear::forward(struct ggml_context* ctx, struct ggml_tensor* x) {
    struct ggml_tensor* out = ggml_mul_mat(ctx, weight, x);
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

} // namespace nn
