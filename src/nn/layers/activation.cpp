#include "nn/layers/activation.h"

#include "nn/core/context.h"
#include "ops/ops.h"

namespace nn {

ggml_tensor* Snake::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return ggml_ops_snake(ctx, input, alpha, target);
}

ggml_tensor* PReLU::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    (void)selected_backend;
    ggml_tensor* parameter = weight.local_tensor();
    if (!parameter) return ggml_relu(ctx, input);

    ggml_tensor* positive = ggml_relu(ctx, input);
    ggml_tensor* negative = ggml_sub(ctx, input, positive);
    const int64_t dimension = parameter->ne[0];
    ggml_tensor* slope = parameter;
    if (input->ne[0] == 1 && input->ne[1] == dimension) {
        slope = ggml_reshape_2d(ctx, parameter, 1, dimension);
    } else if (input->ne[0] == dimension && input->ne[1] == 1) {
        slope = ggml_reshape_2d(ctx, parameter, dimension, 1);
    }
    return ggml_add(ctx, positive, ggml_mul(ctx, slope, negative));
}

ggml_tensor* GLU::forward(
    Context& context, ggml_tensor* input, ggml_backend_t selected_backend
) {
    ggml_context* ctx = context.native_handle();
    ggml_backend_t target = selected_backend ? selected_backend : backend;
    return ggml_ops_glu(ctx, input, target);
}

} // namespace nn
