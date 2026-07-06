#pragma once

#include "models/gguf_model.h"
#include "nn/nn.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace gpt_sovits {

// SoVITS VITS Generator Graph Builder Base Class
struct VITSModel : public GGUFModel {
    // Pre-allocated static input placeholders
    nn::Buffer phone_ids;
    nn::Buffer phone_lengths;
    nn::Buffer word2ph;
    nn::Buffer bert_features;
    nn::Buffer prompt_semantics;
    nn::Buffer refer_audio;
    nn::Buffer prompt_mel;
    std::vector<float> prompt_mel_host;

    struct ggml_tensor* debug_conv_pre = nullptr;
    struct ggml_tensor* debug_cond = nullptr;
    struct ggml_tensor* debug_ups[5] = {nullptr};
    struct ggml_tensor* debug_resblocks[15] = {nullptr};
    struct ggml_tensor* debug_res0_convs1[3] = {nullptr};
    struct ggml_tensor* debug_res0_convs2[3] = {nullptr};
    struct ggml_tensor* debug_conv_post = nullptr;
    struct ggml_tensor* debug_ref_enc_spectral_0 = nullptr;
    struct ggml_tensor* debug_ref_enc_spectral_3 = nullptr;
    struct ggml_tensor* debug_ref_enc_temporal_0 = nullptr;
    struct ggml_tensor* debug_ref_enc_temporal_1 = nullptr;
    struct ggml_tensor* debug_ref_enc_pre_attn = nullptr;
    struct ggml_tensor* debug_ref_enc_post_attn = nullptr;
    struct ggml_tensor* debug_ref_enc_post_fc = nullptr;
    struct ggml_tensor* debug_ref_enc_pre_pool = nullptr;
    struct ggml_tensor* debug_enc_ssl_out = nullptr;
    struct ggml_tensor* debug_enc_text_out = nullptr;
    struct ggml_tensor* debug_enc_mrte_out = nullptr;
    struct ggml_tensor* debug_enc_enc2_out = nullptr;
    struct ggml_tensor* debug_enc_m_p = nullptr;
    struct ggml_tensor* debug_enc_z = nullptr;
    struct ggml_tensor* debug_ssl_proj = nullptr;
    struct ggml_tensor* debug_decoded = nullptr;
    struct ggml_tensor* debug_interp = nullptr;
    struct ggml_tensor* debug_enc_q = nullptr;
    struct ggml_tensor* debug_enc_attn = nullptr;
    struct ggml_tensor* debug_enc_q_cont = nullptr;
    struct ggml_tensor* debug_enc_fa_raw = nullptr;
    struct ggml_tensor* debug_enc_scores = nullptr;
    struct ggml_tensor* debug_enc_attn_w = nullptr;
    struct ggml_tensor* debug_enc_vt = nullptr;
    struct ggml_tensor* debug_enc_out_raw = nullptr;

    // Tensors created during graph construction that need data upload after backend alloc.
    struct UploadEntry {
        struct ggml_tensor* tensor;
        std::vector<uint8_t> data;  // raw bytes to upload
    };
    std::vector<UploadEntry> upload_entries;
    static float flip_data[192 * 192];
    static bool flip_data_ready;
    void upload_pending_data(ggml_backend_t backend);

    std::unordered_map<std::string, TensorShape> original_shapes;
    void on_prepare_tensor(struct ggml_tensor* tensor, const std::string& name) override;
    bool on_upload_tensor(
        struct ggml_tensor* t_backend,
        const void* raw_data,
        size_t size,
        enum ggml_type type,
        const std::string& name
    ) override;

    bool load(const std::string& path, ggml_backend_t backend);

    static std::unique_ptr<VITSModel> create(const std::string& path);

    virtual struct ggml_tensor* forward_from_latent(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        ggml_backend_t backend
    ) = 0;

    virtual struct ggml_tensor* forward(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        float speed,
        ggml_backend_t backend
    ) = 0;

    struct ggml_tensor* compute_speaker_embedding(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* mel_spec,
        struct ggml_tensor* sv_emb,
        ggml_backend_t backend
    );

    virtual ~VITSModel() = default;
};

// VITS Classic Generator (V1 / V2 / V2Pro)
struct VITSModelClassic : public VITSModel {
    struct ggml_tensor* forward_from_latent(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        ggml_backend_t backend
    ) override;

    struct ggml_tensor* forward(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        float speed,
        ggml_backend_t backend
    ) override;
};

// VITS Flow Matching / DiT Generator (V3 / V4)
struct VITSModelCFM : public VITSModel {
    struct ggml_tensor* forward_from_latent(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        ggml_backend_t backend
    ) override;

    struct ggml_tensor* forward(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* phone_ids,
        struct ggml_tensor* phone_lengths,
        struct ggml_tensor* word2ph,
        struct ggml_tensor* bert_features,
        struct ggml_tensor* prompt_semantics,
        struct ggml_tensor* refer_audio,
        float speed,
        ggml_backend_t backend
    ) override;
};

// Shared Helper functions declared for use by classic and CFM subclasses
void clear_conv_1d_params_pool();

struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w);
struct ggml_tensor* ggml_linear(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b);
struct ggml_tensor* ggml_conv_1d_vits(struct ggml_context* ctx, struct ggml_tensor* w, struct ggml_tensor* x, int stride, int padding, int dilation, ggml_backend_t backend);
struct ggml_tensor* ggml_conv_1d_with_bias(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int stride, int dilation, int padding, ggml_backend_t backend);
struct ggml_tensor* ggml_conv_transpose_1d_with_bias(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, ggml_backend_t backend);
struct ggml_tensor* ggml_conv_1d_with_bias_no_transpose(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int stride, int dilation, int padding, ggml_backend_t backend);
struct ggml_tensor* ggml_conv_transpose_1d_with_bias_no_transpose(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, ggml_backend_t backend);
struct ggml_tensor* ggml_layer_norm(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps, ggml_backend_t backend);
struct ggml_tensor* vq_decode(struct ggml_context* ctx, struct ggml_tensor* token_ids, VITSModel& model);
struct ggml_tensor* interp_nearest_2x(struct ggml_context* ctx, struct ggml_tensor* x, VITSModel& model);
struct ggml_tensor* build_encoder(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, VITSModel& model, const std::string& base_prefix, int n_layers, int n_head, int d_k, int T, ggml_backend_t backend);
struct ggml_tensor* build_mrte(struct ggml_context* ctx, struct ggml_tensor* y, struct ggml_tensor* y_mask, struct ggml_tensor* text, struct ggml_tensor* text_mask, struct ggml_tensor* ge, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* build_vits_generator(struct ggml_context* ctx_graph, struct ggml_tensor* latent, struct ggml_tensor* speaker_embedding, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* build_vits_generator_cfm(struct ggml_context* ctx_graph, struct ggml_tensor* latent, struct ggml_tensor* speaker_embedding, VITSModel& model, ggml_backend_t backend);
struct ggml_tensor* build_wn(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, VITSModel& model, const std::string& prefix, int hidden_channels, int kernel_size, int dilation_rate, int n_layers, ggml_backend_t backend);
struct ggml_tensor* build_coupling_layer(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* x_mask, struct ggml_tensor* g, VITSModel& model, const std::string& prefix, int channels, int hidden_channels, int kernel_size, int dilation_rate, int n_layers, bool reverse, ggml_backend_t backend);

} // namespace gpt_sovits
