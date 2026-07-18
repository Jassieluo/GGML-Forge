#include "nn/functional/elementwise.h"

#include "nn/core/context.h"

namespace nn::functional {

ggml_tensor* add_scalar(ggml_context* ctx, ggml_tensor* input, float value) {
    Context context = Context::borrow(ctx);
    ggml_tensor* scalar = context.empty<float>("nn.scalar", {1});
    scalar = ggml_fill(ctx, scalar, value);
    return ggml_add(ctx, input, ggml_repeat(ctx, scalar, input));
}

} // namespace nn::functional
