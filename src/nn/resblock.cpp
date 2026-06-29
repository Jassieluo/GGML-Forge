#include "nn/nn.h"

namespace nn {

// ResBlock1d
ResBlock1d::ResBlock1d(
    struct ggml_tensor* convs1_w[3], struct ggml_tensor* convs1_b[3],
    struct ggml_tensor* convs2_w[3], struct ggml_tensor* convs2_b[3],
    const std::vector<int>& dilations,
    int kernel_size
) {
    for (int i = 0; i < 3; ++i) {
        int dilation = dilations[i];
        int padding = (kernel_size - 1) * dilation / 2;
        convs1[i] = Conv1d(convs1_w[i], convs1_b[i], 1, padding, dilation);
        convs2[i] = Conv1d(convs2_w[i], convs2_b[i], 1, (kernel_size - 1) / 2, 1);
    }
}

struct ggml_tensor* ResBlock1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    struct ggml_tensor* current_x = x;
    for (int i = 0; i < 3; ++i) {
        struct ggml_tensor* xt = ggml_leaky_relu(ctx, current_x, 0.1f, false);
        xt = convs1[i].forward(ctx, xt, backend);
        xt = ggml_leaky_relu(ctx, xt, 0.1f, false);
        xt = convs2[i].forward(ctx, xt, backend);
        current_x = ggml_add(ctx, xt, current_x);
    }
    return current_x;
}

} // namespace nn
