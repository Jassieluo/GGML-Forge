#include "nn/nn.h"

namespace nn {

struct ggml_tensor* PReLU::forward(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    ggml_backend_t backend
) {
    (void)backend;
    struct ggml_tensor* parameter = weight.local_tensor();
    if (!parameter) {
        return ggml_relu(ctx, x);
    }
    
    // PReLU(x) = max(0, x) + weight * min(0, x)
    struct ggml_tensor* pos = ggml_relu(ctx, x);
    struct ggml_tensor* neg = ggml_sub(ctx, x, pos);
    
    int64_t dim = parameter->ne[0];
    struct ggml_tensor* w_reshaped = parameter;
    if (x->ne[0] == 1 && x->ne[1] == dim) {
        w_reshaped = ggml_reshape_2d(ctx, parameter, 1, dim);
    } else if (x->ne[0] == dim && x->ne[1] == 1) {
        w_reshaped = ggml_reshape_2d(ctx, parameter, dim, 1);
    }
    
    struct ggml_tensor* a_neg = ggml_mul(ctx, w_reshaped, neg);
    return ggml_add(ctx, pos, a_neg);
}

} // namespace nn
