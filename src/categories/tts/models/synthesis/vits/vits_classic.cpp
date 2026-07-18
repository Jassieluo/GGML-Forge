#include "vits.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <random>
#include <array>
#include <vector>

namespace gpt_sovits {

struct ggml_tensor* VITSModelClassic::forward_from_latent(
    nn::Context& context,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSRunState& state,
    ggml_backend_t backend
) {
    (void)state;
    struct ggml_context* ctx_graph = context.native_handle();
    struct ggml_tensor* ge = speaker_embedding;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
    }
    return generator->forward(context, latent, ge, backend);
}

struct ggml_tensor* VITSModelClassic::forward(
    nn::Context& context,
    struct ggml_tensor* phone_ids,
    struct ggml_tensor* phone_lengths,
    struct ggml_tensor* word2ph,
    struct ggml_tensor* bert_features,
    struct ggml_tensor* prompt_semantics,
    struct ggml_tensor* refer_audio,
    VITSRunState& state,
    float speed,
    ggml_backend_t backend
) {
    struct ggml_context* ctx_graph = context.native_handle();
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Entering forward..." << std::endl;
    EncodeResult res = encode_semantic_base(context, phone_ids, prompt_semantics, refer_audio, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] encode_semantic_base completed successfully." << std::endl;
    struct ggml_tensor* y2 = res.y2;
    struct ggml_tensor* ge = res.ge;
    int T_y = res.T_y;

    if (!y2) {
        return nullptr;
    }

    // Step 9: Speed scaling
    if (speed != 1.0f && speed > 0.0f) {
        int target_frames = static_cast<int>(T_y / speed) + 1;
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Speed scaling: " << T_y << " -> " << target_frames << " frames" << std::endl;
        y2 = nn::F::interpolate_linear(context, y2, target_frames);
        if (!y2) {
            return nullptr;
        }
        T_y = target_frames;
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Step 9 speed scaling done." << std::endl;

    // Step 10: proj - Conv1d(192, 384, 1) -> split to m_p (192) and logs (192)
    struct ggml_tensor* stats = semantic.output_projection.forward(ctx_graph, y2, backend);
    struct ggml_tensor* m_p = ggml_view_2d(ctx_graph, stats, 192, stats->ne[1], stats->nb[1], 0);
    m_p = ggml_cont(ctx_graph, m_p);
    struct ggml_tensor* logs_p = ggml_view_2d(
        ctx_graph, stats, 192, stats->ne[1], stats->nb[1], 192 * sizeof(float));
    logs_p = ggml_cont(ctx_graph, logs_p);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Step 10 proj done." << std::endl;
    // Step 11: Flow reverse (ResidualCouplingBlock)
    static const std::array<float, 192 * 192> flip_data = [] {
        std::array<float, 192 * 192> values{};
        for (int row = 0; row < 192; ++row) values[row * 192 + (191 - row)] = 1.0f;
        return values;
    }();

    auto flip_ch = [&](struct ggml_tensor* t) -> struct ggml_tensor* {
        struct ggml_tensor* P = context.constant<float>(
            "vits.flow.flip", {192, 192}, nn::data::copy(flip_data));
        return ggml_cont(ctx_graph, ggml_mul_mat(ctx_graph, P, t));
    };

    float noise_scale = 0.5f;
    if (const char* noise_scale_env = std::getenv("VITS_NOISE_SCALE")) {
        noise_scale = std::strtof(noise_scale_env, nullptr);
    }
    std::random_device rd;
    uint32_t seed = rd();
    if (const char* seed_env = std::getenv("VITS_RANDOM_SEED")) {
        seed = static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 10));
    }
    std::mt19937 noise_generator(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> noise_host(192 * T_y);
    for (float& value : noise_host) {
        value = normal(noise_generator) * noise_scale;
    }
    struct ggml_tensor* noise = context.constant<float>(
        "vits.flow.noise", {192, T_y}, nn::data::copy(noise_host));

    struct ggml_tensor* z = ggml_add(
        ctx_graph,
        m_p,
        ggml_mul(ctx_graph, noise, ggml_exp(ctx_graph, logs_p)));
    for (int fi : {6, 4, 2, 0}) {
        z = flip_ch(z);
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Flow " << fi << " starting..." << std::endl;
        z = flow.flows.at(std::to_string(fi)).forward(ctx_graph, z, nullptr, ge, backend);
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Flow " << fi << " done." << std::endl;
    }
    z = ggml_cont(ctx_graph, z);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Flow done. z: [" << z->ne[0] << ", " << z->ne[1] << "]" << std::endl;

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Running generator..." << std::endl;
    struct ggml_tensor* synth = generator->forward(context, z, ge, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Generator completed." << std::endl;
    return synth;
}

} // namespace gpt_sovits
