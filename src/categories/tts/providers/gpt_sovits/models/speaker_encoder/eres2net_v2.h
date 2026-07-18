#pragma once

#include "nn/nn.h"

#include <string>
#include <vector>

namespace gpt_sovits {

struct SpeakerAFF : public nn::Module<SpeakerAFF> {
    nn::Conv2d& first = submodule<nn::Conv2d>("first");
    nn::Conv2d& second = submodule<nn::Conv2d>("second");

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* x, ggml_tensor* residual);
};

struct ERes2NetV2Block : public nn::Module<ERes2NetV2Block> {
    nn::Conv2d& first = submodule<nn::Conv2d>("first");
    nn::ModuleList<nn::Conv2d>& branches = submodule<nn::ModuleList<nn::Conv2d>>("branches");
    nn::Conv2d& output = submodule<nn::Conv2d>("output");
    nn::ModuleList<SpeakerAFF>& fusions = submodule<nn::ModuleList<SpeakerAFF>>("fusions");

    ERes2NetV2Block(
        int input_channels,
        int planes,
        int stride,
        bool use_aff,
        int base_width = 24,
        int scale = 4,
        int expansion = 4);

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input);

private:
    int width_ = 0;
    int scale_ = 0;
    bool use_aff_ = false;
    nn::Conv2d* shortcut_ = nullptr;
};

struct ERes2NetV2Stage : public nn::Module<ERes2NetV2Stage> {
    nn::ModuleList<ERes2NetV2Block>& blocks =
        submodule<nn::ModuleList<ERes2NetV2Block>>("blocks");

    ERes2NetV2Stage(
        int input_channels,
        int planes,
        int block_count,
        int stride,
        bool use_aff);

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input);
};

class ERes2NetV2 : public nn::Module<ERes2NetV2> {
public:
    nn::Conv2d& input = submodule<nn::Conv2d>("input");
    ERes2NetV2Stage& stage1 = submodule<ERes2NetV2Stage>("stage1", 64, 64, 3, 1, false);
    ERes2NetV2Stage& stage2 = submodule<ERes2NetV2Stage>("stage2", 256, 128, 4, 2, false);
    ERes2NetV2Stage& stage3 = submodule<ERes2NetV2Stage>("stage3", 512, 256, 6, 2, true);
    ERes2NetV2Stage& stage4 = submodule<ERes2NetV2Stage>("stage4", 1024, 512, 3, 2, true);
    nn::Conv2d& stage3_downsample = submodule<nn::Conv2d>("stage3_downsample", 2, 1);
    SpeakerAFF& output_fusion = submodule<SpeakerAFF>("output_fusion");

    ERes2NetV2();

    bool load(const std::string& path, ggml_backend_t backend);
    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* fbank);
};

class ERes2NetV2Runner {
public:
    ERes2NetV2Runner(ERes2NetV2& model, ggml_backend_t backend)
        : model_(model), backend_(backend) {}

    bool encode(const std::vector<float>& fbank, int frame_count, std::vector<float>& embedding);

private:
    ERes2NetV2& model_;
    ggml_backend_t backend_ = nullptr;
};

} // namespace gpt_sovits
