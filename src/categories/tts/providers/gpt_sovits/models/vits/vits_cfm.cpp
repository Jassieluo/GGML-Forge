#include "providers/gpt_sovits/models/vits/vits.h"
#include <iostream>
#include <random>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdlib>

namespace gpt_sovits {

struct ggml_tensor* VITSModelCFM::forward_from_latent(
    nn::Context& context,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    VITSRunState& state,
    ggml_backend_t backend
) {
    (void)state;
    return generator->forward(context, latent, speaker_embedding, backend);
}

struct ggml_tensor* VITSModelCFM::condition_features(
    nn::Context& context,
    ggml_tensor* semantic_features,
    ggml_tensor* speaker_embedding,
    ggml_backend_t backend
) {
    if (!semantic_features) return nullptr;
    ggml_context* ctx = context.native_handle();
    ggml_tensor* features = bridge_projection.forward(context, semantic_features, backend);
    features = ggml_leaky_relu(ctx, features, 0.01f, false);
    const double scale = profile.feature_rate_scale;
    const int64_t target_length = static_cast<int64_t>(semantic_features->ne[1] * scale);
    features = nn::F::interpolate_nearest(context, features, target_length, scale);
    return wns1.forward(context, features, speaker_embedding, backend);
}

FlowMatchingInputs VITSModelCFM::prepare_inputs(
    nn::Context& context,
    int T_mel,
    int prompt_len,
    int prompt_start,
    VITSRunState& state
) {
    // Generate initial noise x_init on host
    std::vector<float> x_host(100 * T_mel);
    std::random_device rd;
    uint32_t seed = rd();
    if (const char* seed_env = std::getenv("CFM_RANDOM_SEED")) {
        seed = static_cast<uint32_t>(std::strtoul(seed_env, nullptr, 10));
    }
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float temperature = 1.0f;
    for (int i = 0; i < 100 * T_mel; ++i) {
        x_host[i] = dist(gen) * temperature;
    }
    for (int t = 0; t < prompt_len; ++t) {
        for (int c = 0; c < 100; ++c) {
            x_host[t * 100 + c] = 0.0f;
        }
    }
    struct ggml_tensor* x = context.constant<float>(
        "vits.cfm.noise", {100, T_mel}, nn::data::copy(x_host));

    // Construct prompt_x
    std::vector<float> prompt_x_host(100 * T_mel, 0.0f);
    if (prompt_len > 0 && !state.prompt_mel.empty()) {
        const float* sliced_mel_ptr = state.prompt_mel.data() + prompt_start * 100;
        int copy_len = prompt_len * 100;
        std::memcpy(prompt_x_host.data(), sliced_mel_ptr, copy_len * sizeof(float));
    }
    struct ggml_tensor* prompt_x = context.constant<float>(
        "vits.cfm.prompt", {100, T_mel}, nn::data::copy(prompt_x_host));

    // Construct prompt_mask
    std::vector<float> mask_host(T_mel, 1.0f);
    for (int t = 0; t < prompt_len; ++t) {
        mask_host[t] = 0.0f;
    }
    struct ggml_tensor* prompt_mask = context.constant<float>(
        "vits.cfm.prompt_mask", {1, T_mel}, nn::data::copy(mask_host));

    // Construct pos_tensor for RoPE
    std::vector<int32_t> pos_host(T_mel);
    for (int i = 0; i < T_mel; ++i) pos_host[i] = i;
    struct ggml_tensor* pos_tensor = context.constant<int32_t>(
        "vits.cfm.position", {T_mel}, nn::data::copy(pos_host));

    FlowMatchingInputs inputs;
    inputs.x = x;
    inputs.prompt_x = prompt_x;
    inputs.prompt_mask = prompt_mask;
    inputs.pos_tensor = pos_tensor;
    return inputs;
}

struct ggml_tensor* VITSModelCFM::integrate(
    nn::Context& context,
    const FlowMatchingInputs& inputs,
    struct ggml_tensor* text_embed,
    ggml_backend_t backend
) {
    struct ggml_context* ctx = context.native_handle();
    int n_timesteps = this->cfm_steps;
    if (n_timesteps < 1) {
        n_timesteps = 8;
    }
    float dt = 1.0f / n_timesteps;
    struct ggml_tensor* x = inputs.x;

    // Precompute static timestep embeddings (TS optimization)
    struct ggml_tensor* d_emb = estimator.embed_delta(context, dt, backend);
    std::vector<struct ggml_tensor*> t_cond_list(n_timesteps);
    for (int j = 0; j < n_timesteps; ++j) {
        float t_val = std::max(0.0f, std::min(1.0f, (float)j * dt));
        struct ggml_tensor* t_emb = estimator.embed_time(context, t_val, backend);
        t_cond_list[j] = ggml_add(ctx, t_emb, d_emb);
    }

    for (int j = 0; j < n_timesteps; ++j) {
        struct ggml_tensor* v_pred = estimator.velocity(
            context, x, inputs.prompt_x, t_cond_list[j], text_embed,
            inputs.pos_tensor, backend);
        x = ggml_add(ctx, x, ggml_scale(ctx, v_pred, dt));
        x = ggml_mul(ctx, x, ggml_repeat(ctx, inputs.prompt_mask, x));
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Completed ODE Euler loop" << std::endl;
    return x;
}

struct ggml_tensor* VITSModelCFM::forward(
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
    EncodeResult res = encode_semantic_base(context, phone_ids, prompt_semantics, refer_audio, backend);
    struct ggml_tensor* y2 = res.y2;
    struct ggml_tensor* ge = res.ge;
    int T_y = res.T_y;

    if (!y2) {
        return nullptr;
    }

    // Bridge projection: 192 -> 512
    if (speed > 0.0f && speed != 1.0f) {
        const int64_t speed_adjusted_len = static_cast<int64_t>(T_y / speed) + 1;
        y2 = nn::F::interpolate_linear(context, y2, speed_adjusted_len);
        if (!y2) {
            return nullptr;
        }
        T_y = static_cast<int>(speed_adjusted_len);
    }

    struct ggml_tensor* cond_text = condition_features(context, y2, ge, backend);
    if (!cond_text) return nullptr;
    const int64_t target_len = cond_text->ne[1];

    int prompt_len = 0;
    int prompt_start = 0;
    int mel_frames = !state.prompt_mel.empty()
        ? static_cast<int>(state.prompt_mel.size() / profile.prompt_mel_channels) : 0;
    int fea_ref_frames = !state.prompt_features.empty() ? (int)(state.prompt_features.size() / 512) : 0;
    
    int T_min = 0;
    if (mel_frames > 0 && fea_ref_frames > 0) {
        T_min = std::min(mel_frames, fea_ref_frames);
        int Tref = profile.max_prompt_frames;
        if (T_min > Tref) {
            prompt_start = T_min - Tref;
            T_min = Tref;
        }
        prompt_len = T_min;
    }
    int T_mel = prompt_len + (int)target_len;

    struct ggml_tensor* cond_text_padded = cond_text;
    if (prompt_len > 0) {
        const float* sliced_fea_ptr = state.prompt_features.data() + prompt_start * 512;
        struct ggml_tensor* prompt_fea_ref_tensor = context.constant<float>(
            "vits.cfm.prompt_features", {512, prompt_len},
            nn::data::copy(sliced_fea_ptr, static_cast<size_t>(512 * prompt_len)));

        cond_text_padded = ggml_concat(ctx_graph, prompt_fea_ref_tensor, cond_text, 1);
    }

    // Compute static text embeddings + ConvNeXtV2 blocks once before the loop
    struct ggml_tensor* text_embed = estimator.encode_text(
        context, cond_text_padded, T_mel, backend);

    // Build flow matching inputs
    FlowMatchingInputs inputs = prepare_inputs(context, T_mel, prompt_len, prompt_start, state);

    // Run ODE Euler loop
    struct ggml_tensor* x = integrate(context, inputs, text_embed, backend);
    // Denormalize Mel spectrogram back to linear range: (x + 1)/2 * 14 - 12
    struct ggml_tensor* cfm_res_denorm = nn::F::add_scalar(
        ctx_graph, ggml_scale(ctx_graph, nn::F::add_scalar(ctx_graph, x, 1.0f), 7.0f), -12.0f);

    struct ggml_tensor* target_mel = cfm_res_denorm;
    if (prompt_len > 0) {
        target_mel = ggml_view_2d(ctx_graph, cfm_res_denorm, 100, (int)target_len, cfm_res_denorm->nb[1], prompt_len * cfm_res_denorm->nb[1]);
    }

    // Feed to final BigVGAN / HiFi-GAN generator vocoder
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-CFM Debug] Running Final Generator Vocoder" << std::endl;
    return generator->forward(context, target_mel, ge, backend);
}

} // namespace gpt_sovits
