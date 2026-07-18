#include "nn/nn.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn::functional {

ggml_tensor* sinusoidal_position_embedding(
    Context& context,
    int64_t sequence_length,
    int64_t dimension,
    float theta
) {
    if (sequence_length <= 0 || dimension <= 0 || dimension % 2 != 0 || theta <= 0.0f) {
        throw std::invalid_argument("sinusoidal position embedding requires a positive even dimension");
    }

    const int64_t half = dimension / 2;
    std::vector<float> values(static_cast<size_t>(sequence_length * dimension));
    for (int64_t position = 0; position < sequence_length; ++position) {
        for (int64_t index = 0; index < half; ++index) {
            const double frequency = std::pow(
                static_cast<double>(theta), -2.0 * static_cast<double>(index) / dimension);
            const double angle = static_cast<double>(position) * frequency;
            values[static_cast<size_t>(position * dimension + index)] =
                static_cast<float>(std::cos(angle));
            values[static_cast<size_t>(position * dimension + half + index)] =
                static_cast<float>(std::sin(angle));
        }
    }
    return context.constant<float>(
        "nn.position.sinusoidal", {dimension, sequence_length}, nn::data::copy(values));
}

} // namespace nn::functional
