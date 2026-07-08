#include "nn/nn.h"

namespace nn {


struct ggml_tensor* ResBlock1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t b = backend ? backend : this->backend;
    struct ggml_tensor* current_x = x;
    for (int i = 0; i < 3; ++i) {
        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = convs1[i].forward(ctx, xt, b);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = convs2[i].forward(ctx, xt, b);
        current_x = ggml_add(ctx, xt, current_x);
    }
    return current_x;
}

} // namespace nn
