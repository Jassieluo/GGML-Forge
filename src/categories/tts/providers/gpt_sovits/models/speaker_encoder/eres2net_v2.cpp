#include "providers/gpt_sovits/models/speaker_encoder/eres2net_v2.h"

#include "gguf.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>

namespace gpt_sovits {
namespace {

ggml_tensor* hard_relu(ggml_context* ctx, ggml_tensor* input) {
    return ggml_clamp(ctx, input, 0.0f, 20.0f);
}

ggml_tensor* channel_slice(
    ggml_context* ctx, ggml_tensor* input, int start, int channels
) {
    ggml_tensor* view = ggml_view_4d(
        ctx, input,
        input->ne[0], input->ne[1], channels, input->ne[3],
        input->nb[1], input->nb[2], input->nb[3],
        static_cast<size_t>(start) * input->nb[2]);
    return ggml_cont(ctx, view);
}

} // namespace

ggml_tensor* SpeakerAFF::forward(
    ggml_context* ctx, ggml_tensor* x, ggml_tensor* residual
) {
    ggml_tensor* joined = ggml_concat(ctx, x, residual, 2);
    ggml_tensor* attention = first.forward(ctx, joined);
    attention = ggml_silu(ctx, attention);
    attention = second.forward(ctx, attention);
    attention = nn::F::add_scalar(ctx, ggml_tanh(ctx, attention), 1.0f);
    ggml_tensor* inverse = nn::F::add_scalar(
        ctx, ggml_scale(ctx, attention, -1.0f), 2.0f);
    return ggml_add(
        ctx,
        ggml_mul(ctx, x, attention),
        ggml_mul(ctx, residual, inverse));
}

ERes2NetV2Block::ERes2NetV2Block(
    int input_channels,
    int planes,
    int stride,
    bool use_aff,
    int base_width,
    int scale,
    int expansion
) : width_(static_cast<int>(std::floor(planes * (base_width / 64.0)))),
    scale_(scale), use_aff_(use_aff) {
    first.stride_width = stride;
    first.stride_height = stride;
    for (int i = 0; i < scale_; ++i) {
        nn::Conv2d& branch = branches.emplace_back();
        branch.padding_width = 1;
        branch.padding_height = 1;
        if (i > 0 && use_aff_) fusions.emplace_back();
    }
    if (stride != 1 || input_channels != expansion * planes) {
        shortcut_ = &submodule<nn::Conv2d>("shortcut", stride, 0);
    }
}

ggml_tensor* ERes2NetV2Block::forward(ggml_context* ctx, ggml_tensor* input) {
    ggml_tensor* residual = shortcut_ ? shortcut_->forward(ctx, input) : input;
    ggml_tensor* expanded = hard_relu(ctx, first.forward(ctx, input));
    ggml_tensor* previous = nullptr;
    ggml_tensor* joined = nullptr;
    for (int i = 0; i < scale_; ++i) {
        ggml_tensor* current = channel_slice(ctx, expanded, i * width_, width_);
        if (i > 0) {
            current = use_aff_
                ? fusions[static_cast<size_t>(i - 1)].forward(ctx, previous, current)
                : ggml_add(ctx, previous, current);
        }
        current = hard_relu(ctx, branches[static_cast<size_t>(i)].forward(ctx, current));
        previous = current;
        joined = joined ? ggml_concat(ctx, joined, current, 2) : current;
    }
    ggml_tensor* projected = output.forward(ctx, joined);
    return hard_relu(ctx, ggml_add(ctx, projected, residual));
}

ERes2NetV2Stage::ERes2NetV2Stage(
    int input_channels,
    int planes,
    int block_count,
    int stride,
    bool use_aff
) {
    blocks.emplace_back(input_channels, planes, stride, use_aff);
    for (int i = 1; i < block_count; ++i) {
        blocks.emplace_back(planes * 4, planes, 1, use_aff);
    }
}

ggml_tensor* ERes2NetV2Stage::forward(ggml_context* ctx, ggml_tensor* input) {
    for (size_t i = 0; i < blocks.size(); ++i) input = blocks[i].forward(ctx, input);
    return input;
}

ERes2NetV2::ERes2NetV2() {
    input.padding_width = 1;
    input.padding_height = 1;
}

bool ERes2NetV2::load(const std::string& path, ggml_backend_t selected_backend) {
    if (!selected_backend) return false;
    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[ERes2NetV2] " << error.what() << '\n';
        return false;
    }
    const gguf_context* metadata = source->metadata_context();
    const int architecture_key = gguf_find_key(metadata, "general.architecture");
    if (architecture_key < 0 || gguf_get_kv_type(metadata, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata, architecture_key)) != "gpt_sovits_eres2net_v2") {
        std::cerr << "[ERes2NetV2] Invalid architecture metadata.\n";
        return false;
    }
    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, selected_backend);
    if (!loaded) {
        std::cerr << "[ERes2NetV2] " << loaded.error << '\n';
        return false;
    }
    to(selected_backend);
    return true;
}

ggml_tensor* ERes2NetV2::forward(ggml_context* ctx, ggml_tensor* fbank) {
    ggml_tensor* stem = ggml_relu(ctx, input.forward(ctx, fbank));
    ggml_tensor* first_stage = stage1.forward(ctx, stem);
    ggml_tensor* second_stage = stage2.forward(ctx, first_stage);
    ggml_tensor* third_stage = stage3.forward(ctx, second_stage);
    ggml_tensor* fourth_stage = stage4.forward(ctx, third_stage);
    ggml_tensor* downsampled = stage3_downsample.forward(ctx, third_stage);
    ggml_tensor* fused = output_fusion.forward(ctx, fourth_stage, downsampled);
    ggml_tensor* temporal_mean = ggml_mean(ctx, fused);
    return ggml_reshape_1d(ctx, temporal_mean, ggml_nelements(temporal_mean));
}

bool ERes2NetV2Runner::encode(
    const std::vector<float>& fbank,
    int frame_count,
    std::vector<float>& embedding
) {
    if (!backend_ || frame_count <= 0 ||
        fbank.size() != static_cast<size_t>(frame_count) * 80) return false;
    nn::Context context(1024ull * 1024 * 1024, true);
    ggml_tensor* input = context.input<float>(
        "speaker_encoder.fbank", {frame_count, 80, 1, 1}, nn::data::borrow(fbank));
    ggml_tensor* output = model_.forward(context.native_handle(), input);
    if (!output || ggml_nelements(output) != 20480) return false;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(
        context.native_handle(), backend_);
    if (!buffer) return false;
    context.materialize();
    ggml_cgraph* graph = context.build(output, 65536);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend_, graph);
    if (status == GGML_STATUS_SUCCESS) {
        embedding.resize(20480);
        ggml_backend_tensor_get(output, embedding.data(), 0, embedding.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    return status == GGML_STATUS_SUCCESS;
}

} // namespace gpt_sovits
