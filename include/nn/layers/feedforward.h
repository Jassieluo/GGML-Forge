#pragma once

#include "nn/layers/activation.h"
#include "nn/layers/linear.h"

namespace nn {

class FeedForward : public Module<FeedForward> {
public:
    Linear& w1 = submodule<Linear>("w1");
    Linear& w2 = submodule<Linear>("w2");
    ActivationType act_type = ActivationType::GELU;

    FeedForward() = default;
    FeedForward(
        ggml_tensor* w1_weight, ggml_tensor* w1_bias,
        ggml_tensor* w2_weight, ggml_tensor* w2_bias,
        ActivationType activation = ActivationType::GELU)
        : act_type(activation) {
        if (w1_weight) w1.weight.bind(w1_weight);
        if (w1_bias) w1.bias.bind(w1_bias);
        if (w2_weight) w2.weight.bind(w2_weight);
        if (w2_bias) w2.bias.bind(w2_bias);
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend = nullptr);
};

} // namespace nn
