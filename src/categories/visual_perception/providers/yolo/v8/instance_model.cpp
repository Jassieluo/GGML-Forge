#include "providers/yolo/v8/instance_model.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace visual_perception::yolo::v8 {
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

} // namespace

bool Config::valid() const {
    if (class_count <= 0 || reg_max <= 0 || detect_box_channels <= 0 ||
        detect_class_channels <= 0) return false;
    if (task == Task::instance_segmentation &&
        (mask_count <= 0 || mask_channels <= 0 || prototype_channels <= 0)) return false;
    if (task == Task::pose &&
        (keypoint_count <= 0 || (keypoint_dimensions != 2 && keypoint_dimensions != 3) ||
         keypoint_channels <= 0)) return false;
    if (task == Task::oriented_detection &&
        (angle_count != 1 || angle_channels <= 0)) return false;
    for (int layer : {0, 1, 3, 5, 7, 9, 16, 19}) {
        if (out_channels[layer] <= 0) return false;
    }
    for (int layer : {2, 4, 6, 8, 12, 15, 18, 21}) {
        if (out_channels[layer] <= 0 || hidden_channels[layer] <= 0 ||
            repeats[layer] <= 0) return false;
    }
    return true;
}

Config Config::from_source(const nn::io::Source& source, int expected_classes,
                           int expected_reg_max, Task expected_task,
                           int expected_mask_count, int expected_keypoint_count,
                           int expected_keypoint_dimensions, int expected_angle_count,
                           std::string& error) {
    Config config;
    error.clear();
    config.task = expected_task;
    for (int layer : {0, 1, 3, 5, 7, 16, 19}) {
        config.out_channels[layer] = convolution_outputs(
            source, "model." + std::to_string(layer) + ".conv.weight");
    }
    for (int layer : {2, 4, 6, 8, 12, 15, 18, 21}) {
        const std::string base = "model." + std::to_string(layer);
        config.out_channels[layer] = convolution_outputs(source, base + ".cv2.conv.weight");
        const int cv1_outputs = convolution_outputs(source, base + ".cv1.conv.weight");
        config.hidden_channels[layer] = cv1_outputs / 2;
        while (source.find(base + ".m." + std::to_string(config.repeats[layer]) +
                           ".cv1.conv.weight")) {
            ++config.repeats[layer];
        }
    }
    config.out_channels[9] = convolution_outputs(source, "model.9.cv2.conv.weight");
    config.detect_box_channels = convolution_outputs(
        source, "model.22.cv2.0.0.conv.weight");
    config.detect_class_channels = convolution_outputs(
        source, "model.22.cv3.0.0.conv.weight");
    const int box_outputs = convolution_outputs(source, "model.22.cv2.0.2.weight");
    config.class_count = convolution_outputs(source, "model.22.cv3.0.2.weight");
    config.reg_max = box_outputs > 0 && box_outputs % 4 == 0 ? box_outputs / 4 : 0;
    if (config.task == Task::instance_segmentation) {
        config.mask_channels = convolution_outputs(source, "model.22.cv4.0.0.conv.weight");
        config.mask_count = convolution_outputs(source, "model.22.cv4.0.2.weight");
        config.prototype_channels = convolution_outputs(source, "model.22.proto.cv1.conv.weight");
        const int prototype_masks = convolution_outputs(source, "model.22.proto.cv3.conv.weight");
        for (int scale = 0; scale < 3; ++scale) {
            const std::string base = "model.22.cv4." + std::to_string(scale);
            if (convolution_outputs(source, base + ".0.conv.weight") != config.mask_channels ||
                convolution_outputs(source, base + ".1.conv.weight") != config.mask_channels ||
                convolution_outputs(source, base + ".2.weight") != config.mask_count) {
                error = "YOLOv8 Segment mask branch topology is inconsistent";
                return config;
            }
        }
        if (prototype_masks != config.mask_count || config.mask_count != expected_mask_count) {
            error = "YOLOv8 Segment prototype/mask metadata mismatch";
            return config;
        }
    } else if (config.task == Task::pose) {
        config.keypoint_count = expected_keypoint_count;
        config.keypoint_dimensions = expected_keypoint_dimensions;
        config.keypoint_channels = convolution_outputs(
            source, "model.22.cv4.0.0.conv.weight");
        const int keypoint_outputs = config.keypoint_count * config.keypoint_dimensions;
        for (int scale = 0; scale < 3; ++scale) {
            const std::string base = "model.22.cv4." + std::to_string(scale);
            if (convolution_outputs(source, base + ".0.conv.weight") != config.keypoint_channels ||
                convolution_outputs(source, base + ".1.conv.weight") != config.keypoint_channels ||
                convolution_outputs(source, base + ".2.weight") != keypoint_outputs) {
                error = "YOLOv8 Pose keypoint branch topology is inconsistent";
                return config;
            }
        }
    } else if (config.task == Task::oriented_detection) {
        config.angle_count = expected_angle_count;
        config.angle_channels = convolution_outputs(
            source, "model.22.cv4.0.0.conv.weight");
        for (int scale = 0; scale < 3; ++scale) {
            const std::string base = "model.22.cv4." + std::to_string(scale);
            if (convolution_outputs(source, base + ".0.conv.weight") != config.angle_channels ||
                convolution_outputs(source, base + ".1.conv.weight") != config.angle_channels ||
                convolution_outputs(source, base + ".2.weight") != config.angle_count) {
                error = "YOLOv8 OBB angle branch topology is inconsistent";
                return config;
            }
        }
    }
    if (!config.valid()) {
        error = "YOLOv8 tensor topology is incomplete or invalid";
    } else if (config.class_count != expected_classes || config.reg_max != expected_reg_max) {
        std::ostringstream message;
        message << "YOLOv8 metadata/topology mismatch: classes=" << config.class_count
                << "/" << expected_classes << ", reg_max=" << config.reg_max
                << "/" << expected_reg_max;
        error = message.str();
    }
    return config;
}

Conv::Conv(int, int, int kernel, int stride)
    : conv_(submodule<nn::Conv2d>("conv", stride, kernel / 2)) {}

ggml_tensor* Conv::forward(nn::Context& context, ggml_tensor* input,
                           ggml_backend_t backend) {
    return ggml_silu(context.native_handle(), conv_.forward(context, input, backend));
}

Bottleneck::Bottleneck(int channels, bool shortcut)
    : shortcut_(shortcut),
      cv1_(submodule<Conv>("cv1", channels, channels, 3)),
      cv2_(submodule<Conv>("cv2", channels, channels, 3)) {}

ggml_tensor* Bottleneck::forward(nn::Context& context, ggml_tensor* input,
                                 ggml_backend_t backend) {
    ggml_tensor* output = cv2_.forward(context, cv1_.forward(context, input, backend), backend);
    return shortcut_ ? ggml_add(context.native_handle(), input, output) : output;
}

C2f::C2f(int input_channels, int output_channels, int hidden_channels,
         int repeats, bool shortcut)
    : hidden_channels_(hidden_channels),
      cv1_(submodule<Conv>("cv1", input_channels, hidden_channels * 2, 1)),
      cv2_(submodule<Conv>("cv2", hidden_channels * (2 + repeats), output_channels, 1)),
      blocks_(submodule<nn::ModuleList<Bottleneck>>("m")) {
    for (int index = 0; index < repeats; ++index) {
        blocks_.emplace_back(hidden_channels, shortcut);
    }
}

ggml_tensor* C2f::forward(nn::Context& context, ggml_tensor* input,
                          ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* split = cv1_.forward(context, input, backend);
    ggml_tensor* first = ggml_view_4d(
        ctx, split, split->ne[0], split->ne[1], hidden_channels_, split->ne[3],
        split->nb[1], split->nb[2], split->nb[3], 0);
    ggml_tensor* last = ggml_view_4d(
        ctx, split, split->ne[0], split->ne[1], hidden_channels_, split->ne[3],
        split->nb[1], split->nb[2], split->nb[3],
        static_cast<size_t>(hidden_channels_) * split->nb[2]);
    ggml_tensor* joined = ggml_concat(ctx, first, last, 2);
    for (size_t index = 0; index < blocks_.size(); ++index) {
        last = blocks_[index].forward(context, last, backend);
        joined = ggml_concat(ctx, joined, last, 2);
    }
    return cv2_.forward(context, joined, backend);
}

Sppf::Sppf(int input_channels, int output_channels)
    : cv1_(submodule<Conv>("cv1", input_channels, input_channels / 2, 1)),
      cv2_(submodule<Conv>("cv2", input_channels * 2, output_channels, 1)) {}

ggml_tensor* Sppf::forward(nn::Context& context, ggml_tensor* input,
                           ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* first = cv1_.forward(context, input, backend);
    ggml_tensor* second = ggml_pool_2d(ctx, first, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
    ggml_tensor* third = ggml_pool_2d(ctx, second, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
    ggml_tensor* fourth = ggml_pool_2d(ctx, third, GGML_OP_POOL_MAX, 5, 5, 1, 1, 2, 2);
    ggml_tensor* joined = ggml_concat(ctx, first, second, 2);
    joined = ggml_concat(ctx, joined, third, 2);
    joined = ggml_concat(ctx, joined, fourth, 2);
    return cv2_.forward(context, joined, backend);
}

HeadBranch::HeadBranch(int input_channels, int hidden_channels, int output_channels)
    : first_(submodule<Conv>("0", input_channels, hidden_channels, 3)),
      second_(submodule<Conv>("1", hidden_channels, hidden_channels, 3)),
      output_(submodule<nn::Conv2d>("2", 1, 0)) {
    (void) output_channels;
}

Proto::Proto(const Config& config)
    : cv1_(submodule<Conv>("cv1", config.out_channels[15],
                           config.prototype_channels, 3)),
      upsample_(submodule<nn::ConvTranspose2d>("upsample")),
      cv2_(submodule<Conv>("cv2", config.prototype_channels,
                           config.prototype_channels, 3)),
      cv3_(submodule<Conv>("cv3", config.prototype_channels,
                           config.mask_count, 1)) {
    upsample_.stride[0] = 2;
    upsample_.stride[1] = 2;
}

ggml_tensor* Proto::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    input = cv1_.forward(context, input, backend);
    input = upsample_.forward(context, input, backend);
    input = cv2_.forward(context, input, backend);
    return cv3_.forward(context, input, backend);
}

ggml_tensor* HeadBranch::forward(nn::Context& context, ggml_tensor* input,
                                 ggml_backend_t backend) {
    return output_.forward(
        context, second_.forward(context, first_.forward(context, input, backend), backend),
        backend);
}

Detect::Detect(const Config& config)
    : reg_max_(config.reg_max), class_count_(config.class_count),
      extra_count_(config.task == Task::instance_segmentation
                       ? config.mask_count
                       : (config.task == Task::pose
                              ? config.keypoint_count * config.keypoint_dimensions
                              : (config.task == Task::oriented_detection
                                     ? config.angle_count : 0))),
      boxes_(submodule<nn::ModuleList<HeadBranch>>("cv2")),
      classes_(submodule<nn::ModuleList<HeadBranch>>("cv3")) {
    const std::array<int, 3> inputs = {
        config.out_channels[15], config.out_channels[18], config.out_channels[21]};
    for (int input : inputs) {
        boxes_.emplace_back(input, config.detect_box_channels, config.reg_max * 4);
        classes_.emplace_back(input, config.detect_class_channels, config.class_count);
    }
    if (config.task == Task::instance_segmentation) {
        prototype_ = &submodule<Proto>("proto", config);
        extras_ = &submodule<nn::ModuleList<HeadBranch>>("cv4");
        for (int input : inputs) {
            extras_->emplace_back(input, config.mask_channels, config.mask_count);
        }
    } else if (config.task == Task::pose) {
        extras_ = &submodule<nn::ModuleList<HeadBranch>>("cv4");
        for (int input : inputs) {
            extras_->emplace_back(input, config.keypoint_channels, extra_count_);
        }
    } else if (config.task == Task::oriented_detection) {
        extras_ = &submodule<nn::ModuleList<HeadBranch>>("cv4");
        for (int input : inputs) {
            extras_->emplace_back(input, config.angle_channels, config.angle_count);
        }
    }
}

ggml_tensor* Detect::forward_scale(nn::Context& context, ggml_tensor* input,
                                   size_t scale, ggml_backend_t backend) {
    ggml_tensor* boxes = boxes_[scale].forward(context, input, backend);
    ggml_tensor* classes = classes_[scale].forward(context, input, backend);
    ggml_tensor* output = ggml_concat(context.native_handle(), boxes, classes, 2);
    if (extras_) {
        output = ggml_concat(
            context.native_handle(), output,
            (*extras_)[scale].forward(context, input, backend), 2);
    }
    return output;
}

ggml_tensor* Detect::forward_prototypes(nn::Context& context, ggml_tensor* input,
                                        ggml_backend_t backend) {
    return prototype_ ? prototype_->forward(context, input, backend) : nullptr;
}

Graph::Graph(const Config& config)
    : layer0_(submodule<Conv>("0", 3, config.out_channels[0], 3, 2)),
      layer1_(submodule<Conv>("1", config.out_channels[0], config.out_channels[1], 3, 2)),
      layer2_(submodule<C2f>("2", config.out_channels[1], config.out_channels[2],
                             config.hidden_channels[2], config.repeats[2], true)),
      layer3_(submodule<Conv>("3", config.out_channels[2], config.out_channels[3], 3, 2)),
      layer4_(submodule<C2f>("4", config.out_channels[3], config.out_channels[4],
                             config.hidden_channels[4], config.repeats[4], true)),
      layer5_(submodule<Conv>("5", config.out_channels[4], config.out_channels[5], 3, 2)),
      layer6_(submodule<C2f>("6", config.out_channels[5], config.out_channels[6],
                             config.hidden_channels[6], config.repeats[6], true)),
      layer7_(submodule<Conv>("7", config.out_channels[6], config.out_channels[7], 3, 2)),
      layer8_(submodule<C2f>("8", config.out_channels[7], config.out_channels[8],
                             config.hidden_channels[8], config.repeats[8], true)),
      layer9_(submodule<Sppf>("9", config.out_channels[8], config.out_channels[9])),
      layer12_(submodule<C2f>("12", config.out_channels[9] + config.out_channels[6],
                              config.out_channels[12], config.hidden_channels[12],
                              config.repeats[12], false)),
      layer15_(submodule<C2f>("15", config.out_channels[12] + config.out_channels[4],
                              config.out_channels[15], config.hidden_channels[15],
                              config.repeats[15], false)),
      layer16_(submodule<Conv>("16", config.out_channels[15], config.out_channels[16], 3, 2)),
      layer18_(submodule<C2f>("18", config.out_channels[16] + config.out_channels[12],
                              config.out_channels[18], config.hidden_channels[18],
                              config.repeats[18], false)),
      layer19_(submodule<Conv>("19", config.out_channels[18], config.out_channels[19], 3, 2)),
      layer21_(submodule<C2f>("21", config.out_channels[19] + config.out_channels[9],
                              config.out_channels[21], config.hidden_channels[21],
                              config.repeats[21], false)),
      layer22_(submodule<Detect>("22", config)) {}

ggml_tensor* Graph::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    return forward_outputs(context, input, backend).predictions;
}

Outputs Graph::forward_outputs(nn::Context& context, ggml_tensor* input,
                               ggml_backend_t backend) {
    ggml_context* ctx = context.native_handle();
    ggml_tensor* x0 = layer0_.forward(context, input, backend);
    ggml_tensor* x1 = layer1_.forward(context, x0, backend);
    ggml_tensor* x2 = layer2_.forward(context, x1, backend);
    ggml_tensor* x3 = layer3_.forward(context, x2, backend);
    ggml_tensor* x4 = layer4_.forward(context, x3, backend);
    ggml_tensor* x5 = layer5_.forward(context, x4, backend);
    ggml_tensor* x6 = layer6_.forward(context, x5, backend);
    ggml_tensor* x7 = layer7_.forward(context, x6, backend);
    ggml_tensor* x8 = layer8_.forward(context, x7, backend);
    ggml_tensor* x9 = layer9_.forward(context, x8, backend);
    ggml_tensor* x12 = layer12_.forward(
        context, ggml_concat(ctx, ggml_upscale(ctx, x9, 2, GGML_SCALE_MODE_NEAREST), x6, 2),
        backend);
    ggml_tensor* x15 = layer15_.forward(
        context, ggml_concat(ctx, ggml_upscale(ctx, x12, 2, GGML_SCALE_MODE_NEAREST), x4, 2),
        backend);
    ggml_tensor* x16 = layer16_.forward(context, x15, backend);
    ggml_tensor* x18 = layer18_.forward(context, ggml_concat(ctx, x16, x12, 2), backend);
    ggml_tensor* x19 = layer19_.forward(context, x18, backend);
    ggml_tensor* x21 = layer21_.forward(context, ggml_concat(ctx, x19, x9, 2), backend);

    auto flatten = [&](ggml_tensor* value, size_t scale) {
        value = layer22_.forward_scale(context, value, scale, backend);
        value = ggml_cont(ctx, value);
        return ggml_reshape_2d(ctx, value, value->ne[0] * value->ne[1],
                               layer22_.prediction_channels());
    };
    ggml_tensor* p3 = flatten(x15, 0);
    ggml_tensor* p4 = flatten(x18, 1);
    ggml_tensor* p5 = flatten(x21, 2);
    Outputs outputs;
    outputs.predictions = ggml_concat(ctx, ggml_concat(ctx, p3, p4, 0), p5, 0);
    outputs.prototypes = layer22_.forward_prototypes(context, x15, backend);
    return outputs;
}

Model::Model(Config config)
    : config_(std::move(config)), graph_(submodule<Graph>("model", config_)) {}

ggml_tensor* Model::forward(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend) {
    return graph_.forward(context, input, backend);
}

Outputs Model::forward_outputs(nn::Context& context, ggml_tensor* input,
                               ggml_backend_t backend) {
    return graph_.forward_outputs(context, input, backend);
}

} // namespace visual_perception::yolo::v8
