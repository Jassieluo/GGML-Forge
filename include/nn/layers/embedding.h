#pragma once

#include "nn/core/module.h"

namespace nn {

class Embedding : public Module<Embedding> {
public:
    Parameter& weight = parameter("weight", Parameter::required(
        std::nullopt, Parameter::Usage::embedding_weight));

    Embedding() = default;
    explicit Embedding(ggml_tensor* value) : Embedding() {
        if (value) weight.bind(value);
    }

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input_ids);
};

} // namespace nn
