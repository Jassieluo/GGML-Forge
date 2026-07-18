#include "nn/layers/linear.h"

#include "nn/functional/linear.h"

namespace nn {

ggml_tensor* Linear::forward(ggml_context* ctx, ggml_tensor* input) {
    return functional::linear(ctx, input, weight.tensor(), bias.local_tensor(), backend);
}

} // namespace nn
