#pragma once

#include "nn/nn.h"
#include "nn/io/source.h"

#include <array>
#include <memory>
#include <string>

namespace visual_perception::instance::yolo::v8 {

enum class Task {
    detection,
    instance_segmentation,
    pose,
    oriented_detection,
};

struct Outputs {
    ggml_tensor* predictions = nullptr;
    ggml_tensor* prototypes = nullptr;
};

struct Config {
    std::array<int, 23> out_channels{};
    std::array<int, 23> hidden_channels{};
    std::array<int, 23> repeats{};
    int detect_box_channels = 0;
    int detect_class_channels = 0;
    int reg_max = 0;
    int class_count = 0;
    Task task = Task::detection;
    int mask_count = 0;
    int mask_channels = 0;
    int prototype_channels = 0;
    int keypoint_count = 0;
    int keypoint_dimensions = 0;
    int keypoint_channels = 0;
    int angle_count = 0;
    int angle_channels = 0;

    bool valid() const;
    static Config from_source(const nn::io::Source& source, int expected_classes,
                              int expected_reg_max, Task expected_task,
                              int expected_mask_count, int expected_keypoint_count,
                              int expected_keypoint_dimensions, int expected_angle_count,
                              std::string& error);
};

class Conv final : public nn::Module<Conv> {
public:
    Conv(int input_channels, int output_channels, int kernel, int stride = 1);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    nn::Conv2d& conv_;
};

class Bottleneck final : public nn::Module<Bottleneck> {
public:
    Bottleneck(int channels, bool shortcut);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    bool shortcut_ = false;
    Conv& cv1_;
    Conv& cv2_;
};

class C2f final : public nn::Module<C2f> {
public:
    C2f(int input_channels, int output_channels, int hidden_channels,
        int repeats, bool shortcut);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    int hidden_channels_ = 0;
    Conv& cv1_;
    Conv& cv2_;
    nn::ModuleList<Bottleneck>& blocks_;
};

class Sppf final : public nn::Module<Sppf> {
public:
    Sppf(int input_channels, int output_channels);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    Conv& cv1_;
    Conv& cv2_;
};

class HeadBranch final : public nn::Module<HeadBranch> {
public:
    HeadBranch(int input_channels, int hidden_channels, int output_channels);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    Conv& first_;
    Conv& second_;
    nn::Conv2d& output_;
};

class Proto final : public nn::Module<Proto> {
public:
    explicit Proto(const Config& config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    Conv& cv1_;
    nn::ConvTranspose2d& upsample_;
    Conv& cv2_;
    Conv& cv3_;
};

class Detect final : public nn::Module<Detect> {
public:
    explicit Detect(const Config& config);
    ggml_tensor* forward_scale(nn::Context& context, ggml_tensor* input,
                               size_t scale, ggml_backend_t backend);
    ggml_tensor* forward_prototypes(nn::Context& context, ggml_tensor* input,
                                    ggml_backend_t backend);
    int output_channels() const { return reg_max_ * 4 + class_count_; }
    int prediction_channels() const { return output_channels() + extra_count_; }
    bool has_masks() const { return prototype_ != nullptr; }

private:
    int reg_max_ = 0;
    int class_count_ = 0;
    int extra_count_ = 0;
    nn::ModuleList<HeadBranch>& boxes_;
    nn::ModuleList<HeadBranch>& classes_;
    Proto* prototype_ = nullptr;
    nn::ModuleList<HeadBranch>* extras_ = nullptr;
};

class Graph final : public nn::Module<Graph> {
public:
    explicit Graph(const Config& config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
    Outputs forward_outputs(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend);

private:
    Conv& layer0_;
    Conv& layer1_;
    C2f& layer2_;
    Conv& layer3_;
    C2f& layer4_;
    Conv& layer5_;
    C2f& layer6_;
    Conv& layer7_;
    C2f& layer8_;
    Sppf& layer9_;
    C2f& layer12_;
    C2f& layer15_;
    Conv& layer16_;
    C2f& layer18_;
    Conv& layer19_;
    C2f& layer21_;
    Detect& layer22_;
};

class Model final : public nn::Module<Model> {
public:
    explicit Model(Config config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
    Outputs forward_outputs(nn::Context& context, ggml_tensor* input,
                            ggml_backend_t backend);
    const Config& config() const { return config_; }

private:
    Config config_;
    Graph& graph_;
};

} // namespace visual_perception::instance::yolo::v8
