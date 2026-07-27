#include "providers/lraspp/model.h"

#include "nn/functional/interpolation.h"

#include <stdexcept>

namespace visual_perception::segmentation::lraspp {

ConvAct::ConvAct(int kernel, int stride, int padding, int dilation, int groups,
                 Activation activation)
    : activation_(activation), conv_(submodule<nn::Conv2d>("conv", stride, padding)) {
    conv_.dilation_width = dilation;
    conv_.dilation_height = dilation;
    conv_.groups = groups;
}

ggml_tensor* ConvAct::forward(nn::Context& context, ggml_tensor* input,
                              ggml_backend_t backend) {
    ggml_tensor* value = conv_.forward(context, input, backend);
    if (!value) throw std::runtime_error("LR-ASPP convolution layout rejected");
    if (activation_ == Activation::relu) return ggml_relu(context.native_handle(), value);
    if (activation_ == Activation::hardswish) return ggml_hardswish(context.native_handle(), value);
    return value;
}

SqueezeExcitation::SqueezeExcitation()
    : fc1_(submodule<nn::Conv2d>("fc1", 1, 0)),
      fc2_(submodule<nn::Conv2d>("fc2", 1, 0)) {}

ggml_tensor* SqueezeExcitation::forward(nn::Context& context, ggml_tensor* input,
                                        ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* scale = ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG,
                                     input->ne[0], input->ne[1], 1, 1, 0, 0);
    scale = ggml_relu(ctx, fc1_.forward(context, scale, backend));
    scale = ggml_hardsigmoid(ctx, fc2_.forward(context, scale, backend));
    return ggml_mul(ctx, input, ggml_repeat(ctx, scale, input));
}

InvertedResidual::InvertedResidual(int input_channels, int expanded_channels,
                                   int output_channels, int kernel, int stride,
                                   int dilation, bool squeeze_excitation,
                                   Activation activation)
    : residual_(stride == 1 && input_channels == output_channels),
      depthwise_(submodule<ConvAct>("depthwise", kernel, stride,
                                   ((kernel - 1) / 2) * dilation, dilation,
                                   expanded_channels, activation)),
      project_(submodule<ConvAct>("project", 1, 1, 0, 1, 1, Activation::none)) {
    if (input_channels != expanded_channels) {
        expand_ = &submodule<ConvAct>("expand", 1, 1, 0, 1, 1, activation);
    }
    if (squeeze_excitation) {
        squeeze_excitation_ = &submodule<SqueezeExcitation>("se");
    }
}

ggml_tensor* InvertedResidual::forward(nn::Context& context, ggml_tensor* input,
                                       ggml_backend_t backend) {
    ggml_tensor* value = expand_ ? expand_->forward(context, input, backend) : input;
    value = depthwise_.forward(context, value, backend);
    if (squeeze_excitation_) value = squeeze_excitation_->forward(context, value, backend);
    value = project_.forward(context, value, backend);
    return residual_ ? ggml_add(context.native_handle(), input, value) : value;
}

Backbone::Backbone()
    : stem_(submodule<ConvAct>("stem", 3, 2, 1, 1, 1, Activation::hardswish)),
      blocks_(submodule<nn::ModuleList<InvertedResidual>>("blocks")),
      final_(submodule<ConvAct>("final", 1, 1, 0, 1, 1, Activation::hardswish)) {
    blocks_.emplace_back(16, 16, 16, 3, 1, 1, false, Activation::relu);
    blocks_.emplace_back(16, 64, 24, 3, 2, 1, false, Activation::relu);
    blocks_.emplace_back(24, 72, 24, 3, 1, 1, false, Activation::relu);
    blocks_.emplace_back(24, 72, 40, 5, 2, 1, true, Activation::relu);
    blocks_.emplace_back(40, 120, 40, 5, 1, 1, true, Activation::relu);
    blocks_.emplace_back(40, 120, 40, 5, 1, 1, true, Activation::relu);
    blocks_.emplace_back(40, 240, 80, 3, 2, 1, false, Activation::hardswish);
    blocks_.emplace_back(80, 200, 80, 3, 1, 1, false, Activation::hardswish);
    blocks_.emplace_back(80, 184, 80, 3, 1, 1, false, Activation::hardswish);
    blocks_.emplace_back(80, 184, 80, 3, 1, 1, false, Activation::hardswish);
    blocks_.emplace_back(80, 480, 112, 3, 1, 1, true, Activation::hardswish);
    blocks_.emplace_back(112, 672, 112, 3, 1, 1, true, Activation::hardswish);
    blocks_.emplace_back(112, 672, 160, 5, 1, 2, true, Activation::hardswish);
    blocks_.emplace_back(160, 960, 160, 5, 1, 2, true, Activation::hardswish);
    blocks_.emplace_back(160, 960, 160, 5, 1, 2, true, Activation::hardswish);
}

Backbone::Output Backbone::forward(nn::Context& context, ggml_tensor* input,
                                   ggml_backend_t backend) {
    ggml_tensor* value = stem_.forward(context, input, backend);
    ggml_tensor* low = nullptr;
    for (size_t index = 0; index < blocks_.size(); ++index) {
        value = blocks_[index].forward(context, value, backend);
        if (index == 3) low = value;
    }
    return {low, final_.forward(context, value, backend)};
}

Head::Head()
    : cbr_(submodule<ConvAct>("cbr", 1, 1, 0, 1, 1, Activation::relu)),
      scale_(submodule<nn::Conv2d>("scale", 1, 0)),
      low_classifier_(submodule<nn::Conv2d>("low_classifier", 1, 0)),
      high_classifier_(submodule<nn::Conv2d>("high_classifier", 1, 0)) {}

ggml_tensor* Head::forward(nn::Context& context, ggml_tensor* low,
                           ggml_tensor* high, ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* pooled = ggml_pool_2d(ctx, high, GGML_OP_POOL_AVG,
                                      high->ne[0], high->ne[1], 1, 1, 0, 0);
    ggml_tensor* scale = ggml_sigmoid(ctx, scale_.forward(context, pooled, backend));
    ggml_tensor* value = cbr_.forward(context, high, backend);
    value = ggml_mul(ctx, value, ggml_repeat(ctx, scale, value));
    value = high_classifier_.forward(context, value, backend);
    ggml_ops_ext::ops_resize_nd_config resize;
    resize.input_size[0] = static_cast<int32_t>(value->ne[0]);
    resize.input_size[1] = static_cast<int32_t>(value->ne[1]);
    resize.output_size[0] = static_cast<int32_t>(low->ne[0]);
    resize.output_size[1] = static_cast<int32_t>(low->ne[1]);
    resize.mode = ggml_ops_ext::ops_resize_mode::linear;
    resize.align_corners = false;
    value = nn::functional::resize2d(ctx, value, resize, backend);
    return ggml_add(ctx, low_classifier_.forward(context, low, backend), value);
}

Model::Model(int class_count)
    : class_count_(class_count), backbone_(submodule<Backbone>("backbone")),
      classifier_(submodule<Head>("classifier")) {}

ggml_tensor* Model::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    (void) class_count_;
    const Backbone::Output features = backbone_.forward(context, input, backend);
    ggml_tensor* logits = classifier_.forward(context, features.low, features.high, backend);
    ggml_ops_ext::ops_resize_nd_config resize;
    resize.input_size[0] = static_cast<int32_t>(logits->ne[0]);
    resize.input_size[1] = static_cast<int32_t>(logits->ne[1]);
    resize.output_size[0] = static_cast<int32_t>(input->ne[0]);
    resize.output_size[1] = static_cast<int32_t>(input->ne[1]);
    resize.mode = ggml_ops_ext::ops_resize_mode::linear;
    resize.align_corners = false;
    return nn::functional::resize2d(context.native_handle(), logits, resize, backend);
}

} // namespace visual_perception::segmentation::lraspp
