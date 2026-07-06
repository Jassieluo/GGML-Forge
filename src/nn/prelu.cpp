#include "nn/nn.h"

namespace nn {

struct ggml_tensor* PReLU::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    (void)backend;
    if (!weight) {
        return ggml_relu(ctx, x);
    }
    
    // PReLU(x) = max(0, x) + weight * min(0, x)
    struct ggml_tensor* pos = ggml_relu(ctx, x);
    struct ggml_tensor* neg = ggml_sub(ctx, x, pos);
    
    int64_t dim = weight->ne[0];
    struct ggml_tensor* w_reshaped = weight;
    if (x->ne[0] == 1 && x->ne[1] == dim) {
        w_reshaped = ggml_reshape_2d(ctx, weight, 1, dim);
    } else if (x->ne[0] == dim && x->ne[1] == 1) {
        w_reshaped = ggml_reshape_2d(ctx, weight, dim, 1);
    }
    
    struct ggml_tensor* a_neg = ggml_mul(ctx, w_reshaped, neg);
    return ggml_add(ctx, pos, a_neg);
}

} // namespace nn
