#include "providers/yolo/v8/classification_model.h"

#include <sstream>
#include <utility>

namespace visual_perception::yolo::v8::classification {
namespace {

nn::Shape logical_shape(const nn::io::TensorInfo& info) {
    return info.logical_shape.empty()
        ? info.layout.logical_shape(info.storage_shape)
        : info.logical_shape;
}

int convolution_outputs(const nn::io::Source& source, const std::string& name) {
    const auto index = source.find(name);
    if (!index) return 0;
    const nn::Shape shape = logical_shape(source.info(*index));
    return shape.size() == 4 && shape[3] > 0 ? static_cast<int>(shape[3]) : 0;
}

int linear_outputs(const nn::io::Source& source, const std::string& name) {
    const auto index = source.find(name);
    if (!index) return 0;
    const nn::Shape shape = logical_shape(source.info(*index));
    return shape.size() == 2 && shape[1] > 0 ? static_cast<int>(shape[1]) : 0;
}

} // namespace

bool Config::valid() const {
    if (head_channels <= 0 || class_count <= 0) return false;
    for (int layer : {0, 1, 3, 5, 7}) {
        if (out_channels[layer] <= 0) return false;
    }
    for (int layer : {2, 4, 6, 8}) {
        if (out_channels[layer] <= 0 || hidden_channels[layer] <= 0 ||
            repeats[layer] <= 0) return false;
    }
    return true;
}

Config Config::from_source(const nn::io::Source& source,
                           int expected_classes, std::string& error) {
    Config config;
    error.clear();
    for (int layer : {0, 1, 3, 5, 7}) {
        config.out_channels[layer] = convolution_outputs(
            source, "model." + std::to_string(layer) + ".conv.weight");
    }
    for (int layer : {2, 4, 6, 8}) {
        const std::string base = "model." + std::to_string(layer);
        config.out_channels[layer] = convolution_outputs(source, base + ".cv2.conv.weight");
        const int cv1_outputs = convolution_outputs(source, base + ".cv1.conv.weight");
        config.hidden_channels[layer] = cv1_outputs / 2;
        while (source.find(base + ".m." + std::to_string(config.repeats[layer]) +
                           ".cv1.conv.weight")) {
            ++config.repeats[layer];
        }
    }
    config.head_channels = convolution_outputs(source, "model.9.conv.conv.weight");
    config.class_count = linear_outputs(source, "model.9.linear.weight");
    if (!config.valid()) {
        error = "YOLOv8 Classify tensor topology is incomplete or invalid";
    } else if (config.class_count != expected_classes) {
        std::ostringstream message;
        message << "YOLOv8 Classify metadata/topology mismatch: classes="
                << config.class_count << "/" << expected_classes;
        error = message.str();
    }
    return config;
}

Classify::Classify(const Config& config)
    : conv_(submodule<Conv>("conv", config.out_channels[8],
                            config.head_channels, 1)),
      linear_(submodule<nn::Linear>("linear")) {}

ggml_tensor* Classify::forward(nn::Context& context, ggml_tensor* input,
                               ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    input = conv_.forward(context, input, backend);
    input = ggml_pool_2d(ctx, input, GGML_OP_POOL_AVG,
                         static_cast<int>(input->ne[0]), static_cast<int>(input->ne[1]),
                         static_cast<int>(input->ne[0]), static_cast<int>(input->ne[1]),
                         0, 0);
    input = ggml_reshape_2d(ctx, input, input->ne[2], input->ne[3]);
    return linear_.forward(context, input);
}

Graph::Graph(const Config& config)
    : layer0_(submodule<Conv>("0", 3, config.out_channels[0], 3, 2)),
      layer1_(submodule<Conv>("1", config.out_channels[0],
                              config.out_channels[1], 3, 2)),
      layer2_(submodule<C2f>("2", config.out_channels[1],
                             config.out_channels[2], config.hidden_channels[2],
                             config.repeats[2], true)),
      layer3_(submodule<Conv>("3", config.out_channels[2],
                              config.out_channels[3], 3, 2)),
      layer4_(submodule<C2f>("4", config.out_channels[3],
                             config.out_channels[4], config.hidden_channels[4],
                             config.repeats[4], true)),
      layer5_(submodule<Conv>("5", config.out_channels[4],
                              config.out_channels[5], 3, 2)),
      layer6_(submodule<C2f>("6", config.out_channels[5],
                             config.out_channels[6], config.hidden_channels[6],
                             config.repeats[6], true)),
      layer7_(submodule<Conv>("7", config.out_channels[6],
                              config.out_channels[7], 3, 2)),
      layer8_(submodule<C2f>("8", config.out_channels[7],
                             config.out_channels[8], config.hidden_channels[8],
                             config.repeats[8], true)),
      layer9_(submodule<Classify>("9", config)) {}

ggml_tensor* Graph::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    input = layer0_.forward(context, input, backend);
    input = layer1_.forward(context, input, backend);
    input = layer2_.forward(context, input, backend);
    input = layer3_.forward(context, input, backend);
    input = layer4_.forward(context, input, backend);
    input = layer5_.forward(context, input, backend);
    input = layer6_.forward(context, input, backend);
    input = layer7_.forward(context, input, backend);
    input = layer8_.forward(context, input, backend);
    return layer9_.forward(context, input, backend);
}

Model::Model(Config config)
    : config_(std::move(config)), graph_(submodule<Graph>("model", config_)) {}

ggml_tensor* Model::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    return graph_.forward(context, input, backend);
}

} // namespace visual_perception::yolo::v8::classification
