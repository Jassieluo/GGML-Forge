#include "nn/layers/linear.h"

#include "nn/core/context.h"
#include "nn/functional/linear.h"

namespace nn {

ggml_tensor* Linear::forward(Context& context, ggml_tensor* input) {
    ggml_context* ctx = context.native_handle();
    return functional::linear(ctx, input, weight.tensor(), bias.local_tensor(), backend);
}

} // namespace nn
