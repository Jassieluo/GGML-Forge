#include "providers/fastdepth/model.h"

#include <array>
#include <stdexcept>

namespace visual_perception::depth::fastdepth {

ConvAct::ConvAct(int kernel, int stride, int padding, int groups, Activation activation)
    : activation_(activation), conv_(submodule<nn::Conv2d>("conv", stride, padding)) {
    conv_.groups = groups;
}

ggml_tensor* ConvAct::forward(nn::Context& context, ggml_tensor* input,
                              ggml_backend_t backend) {
    ggml_tensor* value = conv_.forward(context, input, backend);
    if (!value) throw std::runtime_error("FastDepth convolution layout rejected");
    if (activation_ == Activation::relu) return ggml_relu(context.native_handle(), value);
    if (activation_ == Activation::relu6) return ggml_clamp(context.native_handle(), value, 0.0f, 6.0f);
    return value;
}

Separable::Separable(int channels, int kernel, int stride, int padding,
                     Activation activation)
    : depthwise_(submodule<ConvAct>("depthwise", kernel, stride, padding, channels, activation)),
      pointwise_(submodule<ConvAct>("pointwise", 1, 1, 0, 1, activation)) {}

ggml_tensor* Separable::forward(nn::Context& context, ggml_tensor* input,
                                ggml_backend_t backend) {
    return pointwise_.forward(context, depthwise_.forward(context, input, backend), backend);
}

Model::Model()
    : stem_(submodule<ConvAct>("stem", 3, 2, 1, 1, Activation::relu6)),
      encoder_(submodule<nn::ModuleList<Separable>>("encoder")),
      decoder_(submodule<nn::ModuleList<Separable>>("decoder")),
      output_(submodule<ConvAct>("output", 1, 1, 0, 1, Activation::none)) {
    const std::array<int, 13> input = {16,56,88,120,144,256,408,376,272,288,296,328,480};
    const std::array<int, 13> stride = {1,2,1,2,1,2,1,1,1,1,1,2,1};
    for (size_t i = 0; i < input.size(); ++i)
        encoder_.emplace_back(input[i], 3, stride[i], 1, Activation::relu6);
    decoder_.emplace_back(512, 5, 1, 2, Activation::relu);
    decoder_.emplace_back(200, 5, 1, 2, Activation::relu);
    decoder_.emplace_back(256, 5, 1, 2, Activation::relu);
    decoder_.emplace_back(120, 5, 1, 2, Activation::relu);
    decoder_.emplace_back(56, 5, 1, 2, Activation::relu);
}

ggml_tensor* Model::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* value = stem_.forward(context, input, backend);
    ggml_tensor* skip112 = nullptr;
    ggml_tensor* skip56 = nullptr;
    ggml_tensor* skip28 = nullptr;
    for (size_t i = 0; i < encoder_.size(); ++i) {
        value = encoder_[i].forward(context, value, backend);
        if (i == 0) skip112 = value;
        if (i == 2) skip56 = value;
        if (i == 4) skip28 = value;
    }
    value = ggml_upscale(ctx, decoder_[0].forward(context, value, backend), 2,
                         GGML_SCALE_MODE_NEAREST);
    value = ggml_upscale(ctx, decoder_[1].forward(context, value, backend), 2,
                         GGML_SCALE_MODE_NEAREST);
    value = ggml_add(ctx, value, skip28);
    value = ggml_upscale(ctx, decoder_[2].forward(context, value, backend), 2,
                         GGML_SCALE_MODE_NEAREST);
    value = ggml_add(ctx, value, skip56);
    value = ggml_upscale(ctx, decoder_[3].forward(context, value, backend), 2,
                         GGML_SCALE_MODE_NEAREST);
    value = ggml_add(ctx, value, skip112);
    value = ggml_upscale(ctx, decoder_[4].forward(context, value, backend), 2,
                         GGML_SCALE_MODE_NEAREST);
    return output_.forward(context, value, backend);
}

} // namespace visual_perception::depth::fastdepth
