#pragma once

#include "nn/nn.h"

namespace visual_perception::depth::stereonet {

class ResidualBlock final : public nn::Module<ResidualBlock> {
public:
    explicit ResidualBlock(int dilation = 1);
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input, ggml_backend_t backend);
private:
    nn::Conv2d& first_;
    nn::Conv2d& second_;
};

class FeatureExtractor final : public nn::Module<FeatureExtractor> {
public:
    FeatureExtractor();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* input, ggml_backend_t backend);
private:
    nn::ModuleList<nn::Conv2d>& downsample_;
    nn::ModuleList<ResidualBlock>& residual_;
    nn::Conv2d& output_;
};

class CostVolume final : public nn::Module<CostVolume> {
public:
    CostVolume();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* reference,
                         ggml_tensor* target, ggml_backend_t backend);
private:
    nn::ModuleList<nn::Conv3d>& layers_;
    nn::Conv3d& output_;
};

class Refiner final : public nn::Module<Refiner> {
public:
    Refiner();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* reference,
                         ggml_tensor* disparity, int width, int height,
                         ggml_backend_t backend);
private:
    nn::Conv2d& input_;
    nn::ModuleList<ResidualBlock>& residual_;
    nn::Conv2d& output_;
};

class Model final : public nn::Module<Model> {
public:
    Model();
    ggml_tensor* forward(nn::Context& context, ggml_tensor* left,
                         ggml_tensor* right, ggml_backend_t backend);
private:
    FeatureExtractor& features_;
    CostVolume& cost_;
    nn::ModuleList<Refiner>& refiners_;
};

} // namespace visual_perception::depth::stereonet
