#include "gpt_t2s.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <map>

namespace gpt_sovits {

namespace {

static void print_tensor_stats(const std::string& name, struct ggml_tensor* t) {
    if (!t) return;
    int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> data(n);
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(int32_t));
        std::cout << "[T2S Debug Tensor] " << name << " (I32) - values: ";
        for (int i = 0; i < std::min((int64_t)40, n); ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> data(n);
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(ggml_fp16_t));
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        bool has_nan = false;
        for (ggml_fp16_t h : data) {
            float v = ggml_fp16_to_fp32(h);
            if (std::isnan(v)) {
                has_nan = true;
            } else {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
        std::cout << "[T2S Debug Tensor] " << name << " (F16) - nelements: " << n 
                  << ", min: " << min_val << ", max: " << max_val 
                  << (has_nan ? " [HAS NAN]" : "") << "\n";
    } else if (t->type == GGML_TYPE_F32) {
        std::vector<float> data(n);
        ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
        float min_val = std::numeric_limits<float>::max();
        float max_val = -std::numeric_limits<float>::max();
        bool has_nan = false;
        for (float v : data) {
            if (std::isnan(v)) {
                has_nan = true;
            } else {
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
            }
        }
        std::cout << "[T2S Debug Tensor] " << name << " (F32) - nelements: " << n 
                  << ", min: " << min_val << ", max: " << max_val 
                  << (has_nan ? " [HAS NAN]" : "") << "\n";
    } else {
        std::cout << "[T2S Debug Tensor] " << name << " - unsupported type: " << t->type << "\n";
    }
    std::fflush(stdout);
}

static std::vector<float> compute_positional_embeddings(int seq_len, int hidden_dim, float alpha, int pos_offset = 0) {
    std::vector<float> data(seq_len * hidden_dim, 0.0f);
    for (int i_pos = 0; i_pos < seq_len; ++i_pos) {
        int pos = i_pos + pos_offset;
        for (int i = 0; i < hidden_dim / 2; ++i) {
            float denom = std::pow(10000.0f, (2.0f * i) / (float)hidden_dim);
            float val = (float)pos / denom;
            data[i_pos * hidden_dim + 2 * i] = alpha * std::sin(val);
            data[i_pos * hidden_dim + 2 * i + 1] = alpha * std::cos(val);
        }
    }
    return data;
}

static std::vector<float> compute_prefix_causal_mask(int seq_len, int text_len, int n_heads) {
    std::vector<float> data(seq_len * seq_len * n_heads, 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        for (int r = 0; r < seq_len; ++r) {
            for (int c = 0; c < seq_len; ++c) {
                int64_t idx = h * (seq_len * seq_len) + r * seq_len + c;
                if (r < text_len) {
                    data[idx] = (c < text_len) ? 0.0f : -1e4f;
                } else {
                    data[idx] = (c <= r) ? 0.0f : -1e4f;
                }
            }
        }
    }
    return data;
}

// Low-level sampler for top-k/top-p logits sampling
static int32_t sample_logits(
    std::vector<float>& logits, 
    const std::vector<int32_t>& current_audio_ids,
    float temp, int top_k, float top_p, float rep_penalty, bool is_greedy
) {
    if (rep_penalty != 1.0f) {
        std::vector<bool> seen(1025, false);
        for (int id : current_audio_ids) {
            if (id >= 0 && id < 1025) seen[id] = true;
        }
        for (int i = 0; i < (int)logits.size(); ++i) {
            if (seen[i]) {
                if (logits[i] < 0.0f) {
                    logits[i] *= rep_penalty;
                } else {
                    logits[i] /= rep_penalty;
                }
            }
        }
    }

    if (is_greedy) {
        return (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<std::pair<float, int>> indexed_logits(logits.size());
    for (int i = 0; i < (int)logits.size(); ++i) {
        indexed_logits[i] = {logits[i], i};
    }
    std::sort(indexed_logits.begin(), indexed_logits.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    if (top_k > 0 && top_k < (int)logits.size()) {
        indexed_logits.resize(top_k);
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(indexed_logits.size());
    float max_l = indexed_logits[0].first;
    for (int i = 0; i < (int)indexed_logits.size(); ++i) {
        probs[i] = std::exp((indexed_logits[i].first - max_l) / temp);
        sum_exp += probs[i];
    }
    for (int i = 0; i < (int)indexed_logits.size(); ++i) {
        probs[i] /= sum_exp;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        float cumsum = 0.0f;
        int cutoff = 0;
        for (int i = 0; i < (int)probs.size(); ++i) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                cutoff = i;
                break;
            }
        }
        probs.resize(cutoff + 1);
        indexed_logits.resize(cutoff + 1);
        float p_sum = 0.0f;
        for (float p : probs) p_sum += p;
        for (float& p : probs) p /= p_sum;
    }

    static constexpr uint32_t default_seed = 42u;
    static std::mt19937 gen(default_seed);
    static thread_local std::string last_seed_env;
    const char * env_seed = std::getenv("T2S_RANDOM_SEED");
    const std::string seed_env = env_seed ? env_seed : "";
    if (seed_env != last_seed_env) {
        if (!seed_env.empty()) {
            gen.seed((uint32_t) std::strtoul(seed_env.c_str(), nullptr, 10));
        } else {
            gen.seed(default_seed);
        }
        last_seed_env = seed_env;
    }
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int sampled_idx = dist(gen);
    return indexed_logits[sampled_idx].second;
}

} // namespace

void T2SModel::on_read_metadata(struct gguf_context* ctx_gguf) {
    int kid_layers = gguf_find_key(ctx_gguf, "gpt_sovits.t2s.n_layers");
    if (kid_layers >= 0) {
        n_layers = (int)gguf_get_val_u32(ctx_gguf, kid_layers);
    } else {
        n_layers = 24;
    }
    int kid_heads = gguf_find_key(ctx_gguf, "attention.head_count");
    if (kid_heads >= 0) {
        n_heads = (int)gguf_get_val_u32(ctx_gguf, kid_heads);
    }
}

bool T2SModel::load(const std::string& path, ggml_backend_t backend) {
    if (!load_gguf_model(path, *this, backend)) {
        return false;
    }

    // Pre-convert FP16 weights to FP32 for non-CUDA (CPU/SYCL) backends
    bool is_cuda = false;
    if (backend) {
        const char * bname = ggml_backend_name(backend);
        if (bname && strncmp(bname, "CUDA", 4) == 0) {
            is_cuda = true;
        }
    }

    if (!is_cuda) {
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S load] Non-CUDA backend detected. Pre-converting loaded FP16 weights to FP32..." << std::endl;
        struct ggml_init_params custom_params = {
            /* .mem_size   = */ 16 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true
        };
        custom_ctx = ggml_init(custom_params);
        if (!custom_ctx) {
            std::cerr << "[T2S load] Error: Failed to initialize custom_ctx for FP32 weights!" << std::endl;
            return false;
        }

        struct UploadF32Entry {
            std::string name;
            std::vector<float> data;
        };
        std::vector<UploadF32Entry> fp32_upload_list;
        std::vector<struct ggml_tensor*> fp32_tensors_list;

        for (const auto& pair : tensors) {
            struct ggml_tensor* old_w = pair.second;
            if (!old_w || old_w->type != GGML_TYPE_F16) continue;

            if (pair.first.find("embedding") != std::string::npos) {
                continue; // Skip embedding weights
            }

            int64_t w_elems = ggml_nelements(old_w);
            std::vector<uint8_t> w_bytes(ggml_nbytes(old_w));
            ggml_backend_tensor_get(old_w, w_bytes.data(), 0, w_bytes.size());

            std::vector<float> w_f32_data(w_elems);
            const ggml_fp16_t* ptr = (const ggml_fp16_t*)w_bytes.data();
            for (int64_t i = 0; i < w_elems; ++i) {
                w_f32_data[i] = ggml_fp16_to_fp32(ptr[i]);
            }

            struct ggml_tensor* new_w = ggml_new_tensor(custom_ctx, GGML_TYPE_F32, ggml_n_dims(old_w), old_w->ne);
            ggml_set_name(new_w, old_w->name);

            fp32_tensors_list.push_back(new_w);
            fp32_upload_list.push_back({pair.first, w_f32_data});
        }

        custom_buffer = ggml_backend_alloc_ctx_tensors(custom_ctx, backend);
        if (!custom_buffer) {
            std::cerr << "[T2S load] Error: Failed to allocate custom_buffer for weights!" << std::endl;
            return false;
        }

        for (size_t i = 0; i < fp32_tensors_list.size(); ++i) {
            struct ggml_tensor* nt = fp32_tensors_list[i];
            const auto& upload_entry = fp32_upload_list[i];
            ggml_backend_tensor_set(nt, upload_entry.data.data(), 0, upload_entry.data.size() * sizeof(float));
            tensors[upload_entry.name] = nt;
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S load] Pre-converted and uploaded " << fp32_tensors_list.size() << " weights to FP32 successfully." << std::endl;
    }

    // Dynamically retrieve head dim
    struct ggml_tensor* qw = get_tensor("h.layers.0.self_attn.q.weight");
    if (qw) {
        int hidden_dim = (int)qw->ne[0];
        head_dim = hidden_dim / n_heads;
    }

    // Allocate GPU resident Keys and Values KV Cache (Native [head_dim, 512, n_heads, 24] shapes)
    // and pre-allocated static input placeholders.
    struct ggml_init_params kv_params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };
    kv_ctx = ggml_init(kv_params);
    kv_k = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, 24);
    kv_v = ggml_new_tensor_4d(kv_ctx, GGML_TYPE_F32, head_dim, 512, n_heads, 24);

    text_ids.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(text_ids.tensor, "input_text_ids");
    audio_ids.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 512);
    ggml_set_name(audio_ids.tensor, "input_audio_ids");
    token.tensor = ggml_new_tensor_1d(kv_ctx, GGML_TYPE_I32, 1);
    ggml_set_name(token.tensor, "input_token");
    bert_features.tensor = ggml_new_tensor_2d(kv_ctx, GGML_TYPE_F32, 1024, 512);
    ggml_set_name(bert_features.tensor, "input_bert_features");

    kv_buffer = ggml_backend_alloc_ctx_tensors(kv_ctx, backend);
    if (kv_buffer) {
        size_t total_elements = (size_t)head_dim * 512 * n_heads * 24;
        std::vector<float> zero_buf(total_elements, 0.0f);
        ggml_backend_tensor_set(kv_k, zero_buf.data(), 0, total_elements * sizeof(float));
        ggml_backend_tensor_set(kv_v, zero_buf.data(), 0, total_elements * sizeof(float));
    } else {
        std::cerr << "[T2S] Failed to allocate GPU resident KV Cache!\n";
        return false;
    }

    return true;
}

std::vector<int32_t> T2SModel::forward(
    struct ggml_context* ctx_graph, 
    const std::vector<int32_t>& prompt_phones,
    const std::vector<int32_t>& target_phones,
    const std::vector<int32_t>& prompt_semantics,
    struct ggml_tensor* bert_features,
    const std::vector<int32_t>& target_word2ph,
    int max_len,
    ggml_backend_t backend
) {
    bool align_mode = (std::getenv("T2S_ALIGNMENT") != nullptr);
    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "[GPT-SoVITS Debug] Running T2S forward...\n";
        std::cout << "[T2S Debug] prompt_phones (size=" << prompt_phones.size() << "): ";
        for (auto p : prompt_phones) std::cout << p << " ";
        std::cout << "\n[T2S Debug] target_phones (size=" << target_phones.size() << "): ";
        for (auto p : target_phones) std::cout << p << " ";
        std::cout << "\n[T2S Debug] prompt_semantics (size=" << prompt_semantics.size() << "): ";
        for (auto p : prompt_semantics) std::cout << p << " ";
        std::cout << "\n[T2S Debug] bert_features dimensions: " << bert_features->ne[0] << "x" << bert_features->ne[1] << "\n";
        std::fflush(stdout);
    }

    // Setup models weights
    struct ggml_tensor* text_embed = get_tensor("ar_text_embedding.word_embeddings.weight");
    struct ggml_tensor* audio_embed = get_tensor("ar_audio_embedding.word_embeddings.weight");
    struct ggml_tensor* bert_proj_w = get_tensor("bert_proj.weight");
    struct ggml_tensor* bert_proj_b = get_tensor("bert_proj.bias");
    struct ggml_tensor* ar_text_position_alpha = get_tensor("ar_text_position.alpha");
    struct ggml_tensor* ar_audio_position_alpha = get_tensor("ar_audio_position.alpha");
    struct ggml_tensor* predict_w = get_tensor("ar_predict_layer.weight");
    
    if (!text_embed || !audio_embed || !bert_proj_w || !bert_proj_b || !ar_text_position_alpha || !ar_audio_position_alpha || !predict_w) {
        std::cerr << "[T2S] Error: Missing model weights in GGUF weight mapping!\n";
        return {};
    }

    float text_alpha = 1.0f;
    float audio_alpha = 1.0f;
    ggml_backend_tensor_get(ar_text_position_alpha, &text_alpha, 0, sizeof(float));
    ggml_backend_tensor_get(ar_audio_position_alpha, &audio_alpha, 0, sizeof(float));

    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "\n=== Weight Tensor Stats ===\n";
        print_tensor_stats("text_embed", text_embed);
        print_tensor_stats("audio_embed", audio_embed);
        print_tensor_stats("bert_proj_w", bert_proj_w);
        print_tensor_stats("bert_proj_b", bert_proj_b);
        std::cout << "text_alpha: " << text_alpha << ", audio_alpha: " << audio_alpha << "\n";
        std::cout << "===========================\n\n";
        std::fflush(stdout);
    }

    int text_len = (int)(prompt_phones.size() + target_phones.size());
    std::vector<int32_t> text_ids_vec;
    text_ids_vec.reserve(text_len);
    text_ids_vec.insert(text_ids_vec.end(), prompt_phones.begin(), prompt_phones.end());
    text_ids_vec.insert(text_ids_vec.end(), target_phones.begin(), target_phones.end());

    std::vector<int32_t> current_audio_ids = prompt_semantics;
    std::vector<int32_t> generated_semantics;
    int total_decoded = 0;

    // Wrap embedding & linear layers
    nn::Linear bert_proj(bert_proj_w, bert_proj_b);
    nn::Linear predict(predict_w, nullptr);

    // Setup persistent graph allocator (gallocr) for decoder steps
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc) {
        std::cerr << "[T2S] Error: Failed to create graph allocator (gallocr)!\n";
        return {};
    }

    // Context size 4MB, allocated once and reset per step to stabilize graph keys
    struct ggml_init_params init_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_step = ggml_init(init_params);

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[T2S Debug] Entering autoregressive loop..." << std::endl; std::fflush(stdout);

    while (total_decoded < max_len) {
        int audio_len = (int)current_audio_ids.size();
        int total_len = text_len + audio_len;
        if (total_len >= 512) break;

        std::cout << "[T2S Trace] Step " << total_decoded << " starting...\n"; std::fflush(stdout);
        ggml_reset(ctx_step);
        struct ggml_cgraph* cgraph = ggml_new_graph(ctx_step);

        struct ggml_tensor* x = nullptr;
        struct ggml_tensor* bert_features_local = nullptr;
        struct ggml_tensor* text_ids_tensor_view = nullptr;
        struct ggml_tensor* audio_ids_tensor_view = nullptr;
        struct ggml_tensor* token_tensor_view = nullptr;
        struct ggml_tensor* text_pe = nullptr;
        struct ggml_tensor* audio_pe = nullptr;
        struct ggml_tensor* mask = nullptr;

        std::vector<float> text_pe_data;
        std::vector<float> audio_pe_data;
        std::vector<float> mask_data;

        struct ggml_tensor* t_emb = nullptr;
        struct ggml_tensor* text_fused = nullptr;
        struct ggml_tensor* text_rep = nullptr;
        struct ggml_tensor* audio_rep = nullptr;

        struct ggml_tensor* dbg_text_ids = nullptr;
        struct ggml_tensor* dbg_bert_features = nullptr;
        struct ggml_tensor* dbg_bert_proj_aligned = nullptr;
        struct ggml_tensor* dbg_t_emb = nullptr;
        struct ggml_tensor* dbg_text_pe = nullptr;
        struct ggml_tensor* dbg_text_rep = nullptr;
        struct ggml_tensor* dbg_audio_ids = nullptr;
        struct ggml_tensor* dbg_a_emb = nullptr;
        struct ggml_tensor* dbg_audio_pe = nullptr;
        struct ggml_tensor* dbg_audio_rep = nullptr;

        std::vector<float> temp_bert(1024 * text_len);
        std::vector<nn::Input> step_inputs;
        int32_t last_token = 0;

        if (total_decoded == 0) {
            if (bert_features->buffer) {
                ggml_backend_tensor_get(bert_features, temp_bert.data(), 0, temp_bert.size() * sizeof(float));
            } else if (bert_features->data) {
                std::memcpy(temp_bert.data(), bert_features->data, temp_bert.size() * sizeof(float));
            }

            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "[T2S Debug] bert_features first 10 values: ";
                for (int i = 0; i < 10; ++i) {
                    std::cout << temp_bert[i] << " ";
                }
                std::cout << "\n";
                std::fflush(stdout);
            }

            // First step: Process prompt phones and prompt semantics entirely
            nn::Input bert_in;
            bert_in.tensor = this->bert_features.tensor;
            bert_in.data_ptr = temp_bert.data();
            bert_in.size_bytes = temp_bert.size() * sizeof(float);
            step_inputs.push_back(bert_in);

            bert_features_local = ggml_view_2d(ctx_step, this->bert_features.tensor, 1024, text_len, this->bert_features.tensor->nb[1], 0);

            struct ggml_tensor* bert_proj_aligned = ggml_add(ctx_step,
                ggml_mul_mat(ctx_step, bert_proj_w, bert_features_local),
                ggml_reshape_2d(ctx_step, bert_proj_b, ggml_nelements(bert_proj_b), 1)
            );

            // Text embeddings
            nn::Input text_in;
            text_in.tensor = this->text_ids.tensor;
            text_in.data_ptr = text_ids_vec.data();
            text_in.size_bytes = text_len * sizeof(int32_t);
            step_inputs.push_back(text_in);

            text_ids_tensor_view = ggml_view_1d(ctx_step, this->text_ids.tensor, text_len, 0);

            t_emb = ggml_get_rows(ctx_step, text_embed, text_ids_tensor_view);
            text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);

            text_pe_data = compute_positional_embeddings(text_len, 512, text_alpha);
            auto text_pe_in = nn::Input::tensor_2d(ctx_step, GGML_TYPE_F32, 512, text_len, text_pe_data.data(), text_pe_data.size() * sizeof(float));
            text_pe = text_pe_in.tensor;
            step_inputs.push_back(text_pe_in);

            text_rep = ggml_add(ctx_step, text_fused, text_pe);

            // Audio embeddings
            nn::Input audio_in;
            audio_in.tensor = this->audio_ids.tensor;
            audio_in.data_ptr = current_audio_ids.data();
            audio_in.size_bytes = audio_len * sizeof(int32_t);
            step_inputs.push_back(audio_in);

            audio_ids_tensor_view = ggml_view_1d(ctx_step, this->audio_ids.tensor, audio_len, 0);

            struct ggml_tensor* a_emb = ggml_get_rows(ctx_step, audio_embed, audio_ids_tensor_view);

            audio_pe_data = compute_positional_embeddings(audio_len, 512, audio_alpha);
            auto audio_pe_in = nn::Input::tensor_2d(ctx_step, GGML_TYPE_F32, 512, audio_len, audio_pe_data.data(), audio_pe_data.size() * sizeof(float));
            audio_pe = audio_pe_in.tensor;
            step_inputs.push_back(audio_pe_in);

            audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

            x = ggml_concat(ctx_step, text_rep, audio_rep, 1);

            dbg_text_ids = text_ids_tensor_view;
            dbg_bert_features = bert_features_local;
            dbg_bert_proj_aligned = bert_proj_aligned;
            dbg_t_emb = t_emb;
            dbg_text_pe = text_pe;
            dbg_text_rep = text_rep;
            dbg_audio_ids = audio_ids_tensor_view;
            dbg_a_emb = a_emb;
            dbg_audio_pe = audio_pe;
            dbg_audio_rep = audio_rep;
        } else {
            // Self-regressive: Feed only the latest token
            last_token = current_audio_ids.back();
            nn::Input token_in;
            token_in.tensor = this->token.tensor;
            token_in.data_ptr = &last_token;
            token_in.size_bytes = sizeof(int32_t);
            step_inputs.push_back(token_in);

            token_tensor_view = ggml_view_1d(ctx_step, this->token.tensor, 1, 0);

            x = ggml_get_rows(ctx_step, audio_embed, token_tensor_view);

            int pos_idx = audio_len - 1;
            audio_pe_data = compute_positional_embeddings(1, 512, audio_alpha, pos_idx);
            auto audio_pe_in = nn::Input::tensor_1d(ctx_step, GGML_TYPE_F32, 512, audio_pe_data.data(), audio_pe_data.size() * sizeof(float));
            audio_pe = audio_pe_in.tensor;
            step_inputs.push_back(audio_pe_in);

            x = ggml_add(ctx_step, x, audio_pe);
        }

        // Generate mask
        if (total_decoded == 0) {
            mask_data = compute_prefix_causal_mask(total_len, text_len, n_heads);
            auto mask_in = nn::Input::tensor_3d(ctx_step, GGML_TYPE_F32, total_len, total_len, n_heads, mask_data.data(), mask_data.size() * sizeof(float));
            mask = mask_in.tensor;
            step_inputs.push_back(mask_in);
        }

        int q_len = (total_decoded == 0) ? total_len : 1;
        const int hidden_dim = n_heads * head_dim;

        struct ggml_tensor* dbg_x_in = x;
        struct ggml_tensor* dbg_attn_out = nullptr;
        struct ggml_tensor* dbg_x_attn = nullptr;
        struct ggml_tensor* dbg_mlp_out = nullptr;
        struct ggml_tensor* dbg_x_out = nullptr;

        // Execute attention layers
        for (int layer = 0; layer < 24; ++layer) {
            std::string prefix = "h.layers." + std::to_string(layer) + ".";
            struct ggml_tensor* qw = get_tensor(prefix + "self_attn.q.weight");
            struct ggml_tensor* qb = get_tensor(prefix + "self_attn.q.bias");
            struct ggml_tensor* kw = get_tensor(prefix + "self_attn.k.weight");
            struct ggml_tensor* kb = get_tensor(prefix + "self_attn.k.bias");
            struct ggml_tensor* vw = get_tensor(prefix + "self_attn.v.weight");
            struct ggml_tensor* vb = get_tensor(prefix + "self_attn.v.bias");
            struct ggml_tensor* out_w = get_tensor(prefix + "self_attn.out_proj.weight");
            struct ggml_tensor* out_b = get_tensor(prefix + "self_attn.out_proj.bias");

            struct ggml_tensor* ln1_w = get_tensor(prefix + "norm1.weight");
            struct ggml_tensor* ln1_b = get_tensor(prefix + "norm1.bias");
            struct ggml_tensor* ln2_w = get_tensor(prefix + "norm2.weight");
            struct ggml_tensor* ln2_b = get_tensor(prefix + "norm2.bias");

            struct ggml_tensor* ffn_w1 = get_tensor(prefix + "linear1.weight");
            struct ggml_tensor* ffn_b1 = get_tensor(prefix + "linear1.bias");
            struct ggml_tensor* ffn_w2 = get_tensor(prefix + "linear2.weight");
            struct ggml_tensor* ffn_b2 = get_tensor(prefix + "linear2.bias");

            if (!qw || !qb || !kw || !kb || !vw || !vb || !out_w || !out_b || !ln1_w || !ln1_b || !ln2_w || !ln2_b || !ffn_w1 || !ffn_b1 || !ffn_w2 || !ffn_b2) {
                std::cerr << "[T2S] Error: Missing layer " << layer << " weights in GGUF weight mapping!\n";
                ggml_free(ctx_step);
                ggml_gallocr_free(galloc);
                return {};
            }

            // Wrap modules on the stack
            nn::KVHeadAttention self_attn(qw, qb, kw, kb, vw, vb, out_w, out_b, n_heads, head_dim, layer);
            nn::LayerNorm ln1(ln1_w, ln1_b, 1e-5f);
            nn::LayerNorm ln2(ln2_w, ln2_b, 1e-5f);
            nn::FeedForward ffn(ffn_w1, ffn_b1, ffn_w2, ffn_b2, nn::ActivationType::DOUBLE_SWISH);

            struct ggml_tensor* attn_out = self_attn.forward(ctx_step, x, kv_k, kv_v, q_len, total_len, mask, cgraph, backend);

            // Residual + LN1
            struct ggml_tensor* x_attn = ggml_add(ctx_step, x, attn_out);
            x_attn = ln1.forward(ctx_step, x_attn, backend);

            // MLP using FeedForward module
            struct ggml_tensor* mlp_out = ffn.forward(ctx_step, x_attn, backend);

            // Residual + LN2
            x = ggml_add(ctx_step, x_attn, mlp_out);
            x = ln2.forward(ctx_step, x, backend);

            if (layer == 0) {
                dbg_attn_out = attn_out;
                dbg_x_attn = x_attn;
                dbg_mlp_out = mlp_out;
                dbg_x_out = x;
            }
        }

        // No final LayerNorm in GPT-SoVITS T2S model, proceed directly to prediction

        // Predict logits
        struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, hidden_dim, 1, x->nb[1], (q_len - 1) * x->nb[1]);
        struct ggml_tensor* logits_tensor = predict.forward(ctx_step, last_token_rep);
        ggml_build_forward_expand(cgraph, logits_tensor);

        std::cout << "[T2S Trace] Step " << total_decoded << " graph expansion done.\n"; std::fflush(stdout);

        // Allocate step buffers using persistent galloc
        if (!ggml_gallocr_alloc_graph(galloc, cgraph)) {
            std::cerr << "[T2S] Error: Failed to allocate graph using gallocr!\n";
            ggml_free(ctx_step);
            ggml_gallocr_free(galloc);
            return {};
        }

        std::cout << "[T2S Trace] Step " << total_decoded << " graph allocation done.\n"; std::fflush(stdout);

        // Upload input data using PyTorch-style Input wrappers
        for (const auto& input : step_inputs) {
            input.upload();
        }



        std::cout << "[T2S Trace] Step " << total_decoded << " input upload done. Executing compute...\n"; std::fflush(stdout);

        ggml_backend_graph_compute(backend, cgraph);

        std::cout << "[T2S Trace] Step " << total_decoded << " graph compute done.\n"; std::fflush(stdout);

        if (GPT_SOVITS_DEBUG_ENABLED() && total_decoded == 0) {
            std::cout << "\n=== Step 0 Input Tensor Stats ===\n";
            print_tensor_stats("dbg_text_ids", dbg_text_ids);
            print_tensor_stats("dbg_bert_features", dbg_bert_features);
            print_tensor_stats("dbg_bert_proj_aligned", dbg_bert_proj_aligned);
            print_tensor_stats("dbg_t_emb", dbg_t_emb);
            print_tensor_stats("dbg_text_pe", dbg_text_pe);
            print_tensor_stats("dbg_text_rep", dbg_text_rep);
            print_tensor_stats("dbg_audio_ids", dbg_audio_ids);
            print_tensor_stats("dbg_a_emb", dbg_a_emb);
            print_tensor_stats("dbg_audio_pe", dbg_audio_pe);
            print_tensor_stats("dbg_audio_rep", dbg_audio_rep);
            print_tensor_stats("dbg_x_in", dbg_x_in);
            print_tensor_stats("dbg_attn_out", dbg_attn_out);
            print_tensor_stats("dbg_x_attn", dbg_x_attn);
            print_tensor_stats("dbg_mlp_out", dbg_mlp_out);
            print_tensor_stats("dbg_x_out", dbg_x_out);
            std::cout << "=================================\n\n";
            std::fflush(stdout);
        }

        // Get logits back to CPU
        std::vector<float> host_logits(1025);
        ggml_backend_tensor_get(logits_tensor, host_logits.data(), 0, 1025 * sizeof(float));

        if (GPT_SOVITS_DEBUG_ENABLED()) {
            float max_val = *std::max_element(host_logits.begin(), host_logits.end());
            float min_val = *std::min_element(host_logits.begin(), host_logits.end());
            std::cout << "[T2S Debug] Step " << total_decoded << " logits - min: " << min_val << ", max: " << max_val << "\n";
            std::cout << "[T2S Debug] First 10 logits: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << host_logits[i] << " ";
            }
            std::cout << "\n";
            std::fflush(stdout);
        }

        // Config sampling
        float temp = 0.6f;
        int top_k = 20;
        float top_p = 0.6f;
        float rep_penalty = 1.35f;
        const char* env_temp = std::getenv("T2S_TEMPERATURE");
        if (env_temp) temp = std::strtof(env_temp, nullptr);
        const char* env_top_k = std::getenv("T2S_TOP_K");
        if (env_top_k) top_k = (int)std::strtol(env_top_k, nullptr, 10);
        const char* env_top_p = std::getenv("T2S_TOP_P");
        if (env_top_p) top_p = std::strtof(env_top_p, nullptr);
        const char* env_rep_penalty = std::getenv("T2S_REPETITION_PENALTY");
        if (env_rep_penalty) rep_penalty = std::strtof(env_rep_penalty, nullptr);

        bool is_greedy = align_mode || (temp <= 0.0f) || (top_k == 1);
        int32_t next_token = sample_logits(host_logits, current_audio_ids, temp, top_k, top_p, rep_penalty, is_greedy);

        if (next_token == 1024) { // EOS token
            break;
        }

        if (align_mode) {
            // Under alignment cross-testing, override sampling with reference tokens
            int target_idx = total_decoded;
            if (target_idx < (int)target_word2ph.size()) {
                // Here we dummy code or alignment tests override
            }
        }

        current_audio_ids.push_back(next_token);
        generated_semantics.push_back(next_token);
        total_decoded++;
    }

    ggml_free(ctx_step);
    ggml_gallocr_free(galloc);

    return generated_semantics;
}

} // namespace gpt_sovits
