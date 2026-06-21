#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include "ggml.h"
#include "ggml-backend.h"

#ifndef GPT_SOVITS_DEBUG_ENABLED
#define GPT_SOVITS_DEBUG_ENABLED() (std::getenv("GPT_SOVITS_DEBUG") != nullptr)
#endif

namespace gpt_sovits {

// Base Model Weights Struct
struct GGUFModel {
    struct ggml_context* ctx = nullptr;
    ggml_backend_buffer_t backend_buffer = nullptr;
    std::vector<uint8_t> weight_data; // holds the tensor binary data if loaded in memory
    std::unordered_map<std::string, struct ggml_tensor*> tensors;
    
    int n_heads = 8;
    int head_dim = 64;
    
    ~GGUFModel() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (backend_buffer) {
            ggml_backend_buffer_free(backend_buffer);
        }
    }
    
    struct ggml_tensor* get_tensor(const std::string& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) {
            return it->second;
        }
        return nullptr;
    }
};

// CNHuBERT Graph Builder
struct HubertModel : public GGUFModel {
    struct ggml_tensor* pos_conv_w = nullptr;
    std::vector<uint8_t> pos_conv_w_data;
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;
    
    ~HubertModel() {
        if (custom_ctx) {
            ggml_free(custom_ctx);
        }
        if (custom_buffer) {
            ggml_backend_buffer_free(custom_buffer);
        }
    }

    bool load(const std::string& path, ggml_backend_t backend);
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, struct ggml_tensor* input_audio, ggml_backend_t backend);
};

// RoBERTa BERT Graph Builder
struct BertModel : public GGUFModel {
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;

    ~BertModel() {
        if (custom_ctx) {
            ggml_free(custom_ctx);
        }
        if (custom_buffer) {
            ggml_backend_buffer_free(custom_buffer);
        }
    }

    bool load(const std::string& path, ggml_backend_t backend);
    struct ggml_tensor* forward(struct ggml_context* ctx_graph, const std::vector<int32_t>& input_ids, ggml_backend_t backend);
};

// T2S Autoregressive GPT Graph Builder
struct T2SModel : public GGUFModel {
    struct ggml_context* kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buffer = nullptr;
    struct ggml_tensor* kv_k = nullptr;
    struct ggml_tensor* kv_v = nullptr;
    
    struct ggml_context* custom_ctx = nullptr;
    ggml_backend_buffer_t custom_buffer = nullptr;

    ~T2SModel() {
        if (kv_buffer) ggml_backend_buffer_free(kv_buffer);
        if (kv_ctx) ggml_free(kv_ctx);
        if (custom_buffer) ggml_backend_buffer_free(custom_buffer);
        if (custom_ctx) ggml_free(custom_ctx);
    }

    bool load(const std::string& path, ggml_backend_t backend);
    
    // Autoregressive token-by-token decoding with KV Cache
    std::vector<int32_t> forward(
        struct ggml_context* ctx_graph, 
        const std::vector<int32_t>& prompt_phones,
        const std::vector<int32_t>& target_phones,
        const std::vector<int32_t>& prompt_semantics,
        struct ggml_tensor* bert_features,
        const std::vector<int32_t>& target_word2ph,
        int max_len,
        ggml_backend_t backend
    );
};

// SoVITS VITS Generator Graph Builder
struct VITSModel : public GGUFModel {
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
    // (e.g. flip permutation matrices, interp indices, etc.)
    struct UploadEntry {
        struct ggml_tensor* tensor;
        std::vector<uint8_t> data;  // raw bytes to upload
    };
    std::vector<UploadEntry> upload_entries;
    static float flip_data[192 * 192];
    static bool flip_data_ready;
    void upload_pending_data(ggml_backend_t backend);

    bool load(const std::string& path, ggml_backend_t backend);

    struct ggml_tensor* forward_from_latent(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* latent,
        struct ggml_tensor* speaker_embedding,
        ggml_backend_t backend
    );

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
    );

    struct ggml_tensor* compute_speaker_embedding(
        struct ggml_context* ctx_graph,
        struct ggml_tensor* mel_spec,
        ggml_backend_t backend
    );
};

// Helper to load GGUF files and extract weights into ggml tensors
bool load_gguf_model(const std::string& path, GGUFModel& model, ggml_backend_t backend);

} // namespace gpt_sovits
