#pragma once

#include "providers/yolo/v8/instance_model.h"

#include <array>
#include <string>

namespace visual_perception::yolo::v8::classification {

struct Config {
    std::array<int, 10> out_channels{};
    std::array<int, 10> hidden_channels{};
    std::array<int, 10> repeats{};
    int head_channels = 0;
    int class_count = 0;

    bool valid() const;
    static Config from_source(const nn::io::Source& source,
                              int expected_classes, std::string& error);
};

class Classify final : public nn::Module<Classify> {
public:
    explicit Classify(const Config& config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);

private:
    Conv& conv_;
    nn::Linear& linear_;
};

class Graph final : public nn::Module<Graph> {
public:
    explicit Graph(const Config& config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
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
    Classify& layer9_;
};

class Model final : public nn::Module<Model> {
public:
    explicit Model(Config config);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input,
                         ggml_backend_t backend);
    const Config& config() const { return config_; }

private:
    Config config_;
    Graph& graph_;
};

} // namespace visual_perception::yolo::v8::classification
