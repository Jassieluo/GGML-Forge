#include "nn/functional/interpolation.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace nn::functional {

ggml_tensor* interpolate_nearest_2x(ggml_context* ctx, ggml_tensor* input) {
    const int64_t channels = input->ne[0];
    const int64_t length = input->ne[1];
    ggml_tensor* reshaped = ggml_reshape_3d(ctx, input, channels, 1, length);
    ggml_tensor* target = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, 2, length);
    ggml_tensor* repeated = ggml_repeat(ctx, reshaped, target);
    return ggml_cont(ctx, ggml_reshape_2d(ctx, repeated, channels, length * 2));
}

ggml_tensor* interpolate_nearest(
    Context& context,
    ggml_tensor* input,
    int64_t target_length,
    double scale_factor
) {
    if (!input || target_length <= 0 || input->ne[1] <= 0) return nullptr;
    const int64_t source_length = input->ne[1];
    const double scale = scale_factor > 0.0
        ? scale_factor : static_cast<double>(target_length) / source_length;
    std::vector<int32_t> indices(static_cast<size_t>(target_length));
    for (int64_t index = 0; index < target_length; ++index) {
        const int64_t source = std::min<int64_t>(
            source_length - 1, static_cast<int64_t>(index / scale));
        indices[static_cast<size_t>(index)] = static_cast<int32_t>(source);
    }
    ggml_tensor* index_tensor = context.constant<int32_t>(
        "nn.interpolate.nearest.index", {target_length}, nn::data::copy(indices));
    return ggml_get_rows(context.native_handle(), input, index_tensor);
}

ggml_tensor* interpolate_linear(Context& context, ggml_tensor* input, int64_t target_length) {
    if (!input || target_length <= 0 || input->ne[1] <= 0) return nullptr;
    const int64_t source_length = input->ne[1];
    if (target_length == source_length) return input;

    std::vector<int32_t> left_indices(static_cast<size_t>(target_length));
    std::vector<int32_t> right_indices(static_cast<size_t>(target_length));
    std::vector<float> left_weights(static_cast<size_t>(target_length));
    std::vector<float> right_weights(static_cast<size_t>(target_length));
    const double scale = static_cast<double>(source_length) / target_length;
    for (int64_t index = 0; index < target_length; ++index) {
        const double position = std::max(0.0, (index + 0.5) * scale - 0.5);
        const int64_t left = std::min<int64_t>(
            source_length - 1, static_cast<int64_t>(std::floor(position)));
        const int64_t right = std::min<int64_t>(source_length - 1, left + 1);
        const float right_weight = static_cast<float>(
            std::max(0.0, position - std::floor(position)));
        left_indices[static_cast<size_t>(index)] = static_cast<int32_t>(left);
        right_indices[static_cast<size_t>(index)] = static_cast<int32_t>(right);
        left_weights[static_cast<size_t>(index)] = 1.0f - right_weight;
        right_weights[static_cast<size_t>(index)] = right_weight;
    }

    ggml_context* ctx = context.native_handle();
    ggml_tensor* left_index = context.constant<int32_t>(
        "nn.interpolate.linear.left_index", {target_length}, nn::data::copy(left_indices));
    ggml_tensor* right_index = context.constant<int32_t>(
        "nn.interpolate.linear.right_index", {target_length}, nn::data::copy(right_indices));
    ggml_tensor* left_weight = context.constant<float>(
        "nn.interpolate.linear.left_weight", {1, target_length}, nn::data::copy(left_weights));
    ggml_tensor* right_weight = context.constant<float>(
        "nn.interpolate.linear.right_weight", {1, target_length}, nn::data::copy(right_weights));
    ggml_tensor* left = ggml_get_rows(ctx, input, left_index);
    ggml_tensor* right = ggml_get_rows(ctx, input, right_index);
    left = ggml_mul(ctx, left, ggml_repeat(ctx, left_weight, left));
    right = ggml_mul(ctx, right, ggml_repeat(ctx, right_weight, right));
    return ggml_add(ctx, left, right);
}

} // namespace nn::functional
