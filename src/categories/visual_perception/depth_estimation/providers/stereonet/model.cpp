#include "providers/stereonet/model.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace visual_perception::depth::stereonet {
namespace {

ggml_tensor* leaky(ggml_context* ctx, ggml_tensor* value) {
    return ggml_leaky_relu(ctx, value, 0.2f, false);
}

ggml_tensor* resize(nn::Context& context, ggml_tensor* input, int width, int height,
                    ggml_backend_t backend) {
    ggml_ops_ext::ops_resize_nd_config config;
    config.input_size[0] = static_cast<int32_t>(input->ne[0]);
    config.input_size[1] = static_cast<int32_t>(input->ne[1]);
    config.output_size[0] = width;
    config.output_size[1] = height;
    config.mode = ggml_ops_ext::ops_resize_mode::linear;
    config.align_corners = true;
    return nn::functional::resize2d(context.native_handle(), input, config, backend);
}

} // namespace

ResidualBlock::ResidualBlock(int dilation)
    : first_(submodule<nn::Conv2d>("first", 1, dilation)),
      second_(submodule<nn::Conv2d>("second", 1, dilation)) {
    first_.dilation_width = first_.dilation_height = dilation;
    second_.dilation_width = second_.dilation_height = dilation;
}

ggml_tensor* ResidualBlock::forward(nn::Context& context, ggml_tensor* input,
                                    ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* value = leaky(ctx, first_.forward(context, input, backend));
    return leaky(ctx, ggml_add(ctx, second_.forward(context, value, backend), input));
}

FeatureExtractor::FeatureExtractor()
    : downsample_(submodule<nn::ModuleList<nn::Conv2d>>("downsample")),
      residual_(submodule<nn::ModuleList<ResidualBlock>>("residual")),
      output_(submodule<nn::Conv2d>("output", 1, 1)) {
    for (int i = 0; i < 3; ++i) downsample_.emplace_back(2, 2);
    for (int i = 0; i < 6; ++i) residual_.emplace_back(1);
}

ggml_tensor* FeatureExtractor::forward(nn::Context& context, ggml_tensor* input,
                                       ggml_backend_t backend) {
    ggml_tensor* value = input;
    for (auto& layer : downsample_) value = layer.forward(context, value, backend);
    for (auto& block : residual_) value = block.forward(context, value, backend);
    return output_.forward(context, value, backend);
}

CostVolume::CostVolume()
    : layers_(submodule<nn::ModuleList<nn::Conv3d>>("layers")),
      output_(submodule<nn::Conv3d>("output")) {
    for (int i = 0; i < 4; ++i) {
        nn::Conv3d& layer = layers_.emplace_back();
        layer.padding[0] = layer.padding[1] = layer.padding[2] = 1;
    }
    output_.padding[0] = output_.padding[1] = output_.padding[2] = 1;
}

ggml_tensor* CostVolume::forward(nn::Context& context, ggml_tensor* reference,
                                 ggml_tensor* target, ggml_backend_t backend) {
    constexpr int disparities = 32;
    ggml_context* ctx = context.native_handle();
    const int64_t width = reference->ne[0], height = reference->ne[1];
    if (width < disparities || target->ne[0] != width || target->ne[1] != height ||
        target->ne[2] != reference->ne[2]) {
        throw std::runtime_error("StereoNet feature geometry is invalid");
    }
    ggml_tensor* packed = nullptr;
    for (int d = 0; d < disparities; ++d) {
        ggml_tensor* slab = nullptr;
        if (d == 0) {
            slab = ggml_sub(ctx, reference, target);
        } else {
            ggml_tensor* ref_view = ggml_view_4d(
                ctx, reference, width - d, height, reference->ne[2], 1,
                reference->nb[1], reference->nb[2], reference->nb[3], d * reference->nb[0]);
            ggml_tensor* target_view = ggml_view_4d(
                ctx, target, width - d, height, target->ne[2], 1,
                target->nb[1], target->nb[2], target->nb[3], 0);
            ggml_tensor* difference = ggml_sub(ctx, ref_view, target_view);
            ggml_ops_ext::ops_pad_nd_config padding;
            padding.input_size[0] = static_cast<int32_t>(width - d);
            padding.input_size[1] = static_cast<int32_t>(height);
            padding.padding_before[0] = d;
            slab = nn::functional::pad2d(ctx, difference, padding, backend);
        }
        slab = ggml_reshape_2d(ctx, ggml_cont(ctx, slab), width * height, reference->ne[2]);
        packed = packed ? ggml_concat(ctx, packed, slab, 0) : slab;
    }
    ggml_tensor* value = packed;
    for (auto& layer : layers_) {
        layer.input_size[0] = static_cast<int>(width);
        layer.input_size[1] = static_cast<int>(height);
        layer.input_size[2] = disparities;
        value = leaky(ctx, layer.forward(context, value, backend));
    }
    output_.input_size[0] = static_cast<int>(width);
    output_.input_size[1] = static_cast<int>(height);
    output_.input_size[2] = disparities;
    value = output_.forward(context, value, backend);

    ggml_tensor* cost = ggml_reshape_4d(ctx, value, width, height, disparities, 1);
    cost = ggml_cont(ctx, ggml_permute(ctx, ggml_scale(ctx, cost, -1.0f), 1, 2, 0, 3));
    ggml_tensor* probability = ggml_soft_max(ctx, cost);
    std::vector<float> grid(disparities);
    for (int i = 0; i < disparities; ++i) grid[i] = 256.0f * i / (disparities - 1);
    ggml_tensor* values = context.constant<float>(
        "stereonet.disparity_grid", {disparities, 1, 1, 1}, nn::data::copy(grid));
    ggml_tensor* disparity = ggml_sum_rows(ctx, ggml_mul(ctx, probability,
                                                         ggml_repeat(ctx, values, probability)));
    return ggml_cont(ctx, ggml_permute(ctx, disparity, 2, 0, 1, 3));
}

Refiner::Refiner()
    : input_(submodule<nn::Conv2d>("input", 1, 1)),
      residual_(submodule<nn::ModuleList<ResidualBlock>>("residual")),
      output_(submodule<nn::Conv2d>("output", 1, 1)) {
    constexpr std::array<int, 6> dilations = {1, 2, 4, 8, 1, 1};
    for (int dilation : dilations) residual_.emplace_back(dilation);
}

ggml_tensor* Refiner::forward(nn::Context& context, ggml_tensor* reference,
                              ggml_tensor* disparity, int width, int height,
                              ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* reference_scaled = resize(context, reference, width, height, backend);
    ggml_tensor* disparity_scaled = resize(context, disparity, width, height, backend);
    ggml_tensor* value = input_.forward(
        context, ggml_concat(ctx, reference_scaled, disparity_scaled, 2), backend);
    for (auto& block : residual_) value = block.forward(context, value, backend);
    return ggml_relu(ctx, ggml_add(ctx, output_.forward(context, value, backend), disparity_scaled));
}

Model::Model()
    : features_(submodule<FeatureExtractor>("features")),
      cost_(submodule<CostVolume>("cost")),
      refiners_(submodule<nn::ModuleList<Refiner>>("refiners")) {
    for (int i = 0; i < 3; ++i) refiners_.emplace_back();
}

ggml_tensor* Model::forward(nn::Context& context, ggml_tensor* left,
                            ggml_tensor* right, ggml_backend_t backend) {
    ggml_tensor* left_features = features_.forward(context, left, backend);
    ggml_tensor* right_features = features_.forward(context, right, backend);
    ggml_tensor* disparity = cost_.forward(context, left_features, right_features, backend);
    for (int i = 0; i < 3; ++i) {
        const int divisor = 1 << (2 - i);
        disparity = refiners_[i].forward(context, left, disparity,
                                         static_cast<int>(left->ne[0]) / divisor,
                                         static_cast<int>(left->ne[1]) / divisor, backend);
    }
    return disparity;
}

} // namespace visual_perception::depth::stereonet
