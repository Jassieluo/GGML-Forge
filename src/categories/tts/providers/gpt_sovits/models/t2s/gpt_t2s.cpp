#include "providers/gpt_sovits/models/t2s/gpt_t2s.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>
#include <map>
#include <chrono>
#include <exception>
#include <memory>

namespace gpt_sovits {

T2SInputs::T2SInputs(ggml_backend_t backend) {
    text_ids = context.empty<int32_t>("input_text_ids", {512});
    audio_ids = context.empty<int32_t>("input_audio_ids", {512});
    token = context.empty<int32_t>("input_token", {1});
    bert_features = context.empty<float>("input_bert_features", {1024, 512});
    buffer_ = ggml_backend_alloc_ctx_tensors(context.native_handle(), backend);
    if (!buffer_) throw std::runtime_error("failed to allocate T2S input tensors");
}

T2SInputs::~T2SInputs() {
    if (buffer_) ggml_backend_buffer_free(buffer_);
}
namespace {

static std::vector<float> compute_positional_embeddings(int seq_len, int hidden_dim, float alpha, int pos_offset = 0) {
    std::vector<float> data(seq_len * hidden_dim, 0.0f);
    static int cached_dim = 0;
    static std::vector<float> inv_denoms;
    if (cached_dim != hidden_dim) {
        cached_dim = hidden_dim;
        inv_denoms.resize(hidden_dim / 2);
        for (int i = 0; i < hidden_dim / 2; ++i) {
            inv_denoms[i] = 1.0f / std::pow(10000.0f, (2.0f * i) / (float)hidden_dim);
        }
    }

    for (int i_pos = 0; i_pos < seq_len; ++i_pos) {
        int pos = i_pos + pos_offset;
        for (int i = 0; i < hidden_dim / 2; ++i) {
            float val = (float)pos * inv_denoms[i];
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
    float temp, int top_k, float top_p, float rep_penalty, bool is_greedy,
    std::mt19937& rng
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

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int sampled_idx = dist(rng);
    return indexed_logits[sampled_idx].second;
}

} // namespace

bool T2SModel::read_metadata(const struct gguf_context* ctx_gguf) {
    auto read_u32 = [&](const char* key) {
        const int id = gguf_find_key(ctx_gguf, key);
        return id >= 0 && gguf_get_kv_type(ctx_gguf, id) == GGUF_TYPE_UINT32
            ? static_cast<int>(gguf_get_val_u32(ctx_gguf, id)) : 0;
    };
    family = read_u32("gpt_sovits.t2s.family");
    n_layers = read_u32("gpt_sovits.t2s.n_layers");
    metadata_hidden_dim = read_u32("gpt_sovits.t2s.hidden_dim");
    n_heads = read_u32("attention.head_count");
    if (family < 1 || family > 3 || n_layers <= 0 || metadata_hidden_dim <= 0 || n_heads <= 0) return false;

    return true;
}

T2SModel::T2SModel() = default;

bool T2SModel::load(const std::string& path, ggml_backend_t backend) {
    if (!backend) return false;

    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[T2S] " << error.what() << "\n";
        return false;
    }

    const gguf_context* metadata = source->metadata_context();
    const int64_t architecture_key = gguf_find_key(metadata, "general.architecture");
    const int64_t version_key = gguf_find_key(metadata, "gpt_sovits.version");
    if (architecture_key < 0 || gguf_get_kv_type(metadata, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata, architecture_key)) != "gpt_sovits_t2s" ||
        version_key < 0 || gguf_get_kv_type(metadata, version_key) != GGUF_TYPE_STRING) {
        std::cerr << "[T2S] Invalid architecture or missing gpt_sovits.version metadata.\n";
        return false;
    }
    version_string = canonical_model_version(gguf_get_val_str(metadata, version_key));
    version = coarse_version_from_string(version_string);
    if (version_string.empty() || version == 0 || !read_metadata(metadata)) {
        std::cerr << "[T2S] Invalid or incomplete model metadata.\n";
        return false;
    }

    const auto text_embedding_index = source->find("word_embeddings.weight");
    if (!text_embedding_index) {
        std::cerr << "[T2S] Text embedding tensor is missing.\n";
        return false;
    }
    const nn::Shape& text_embedding_shape = source->info(*text_embedding_index).logical_shape;
    if (text_embedding_shape.empty() || text_embedding_shape[0] != metadata_hidden_dim ||
        metadata_hidden_dim % n_heads != 0) {
        std::cerr << "[T2S] Invalid or incomplete Transformer topology metadata.\n";
        return false;
    }
    const int model_hidden_dim = metadata_hidden_dim;
    decoder.reset(n_layers, n_heads, model_hidden_dim / n_heads, nn::ActivationType::RELU, 1e-5f);

    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, backend);
    if (!loaded) {
        std::cerr << "[T2S] " << loaded.error << "\n";
        return false;
    }
    this->to(backend);

    head_dim = model_hidden_dim / n_heads;

    // Configure head counting parameters for all attention submodules
    for (int i = 0; i < n_layers; ++i) {
        decoder.layers[i].self_attn.n_heads = n_heads;
        decoder.layers[i].self_attn.head_dim = head_dim;
    }

    if (text_position_alpha.is_bound()) {
        ggml_backend_tensor_get(text_position_alpha.local_tensor(), &text_alpha, 0, sizeof(float));
    }
    if (audio_position_alpha.is_bound()) {
        ggml_backend_tensor_get(audio_position_alpha.local_tensor(), &audio_alpha, 0, sizeof(float));
    }
    text_position_cache = compute_positional_embeddings(512, model_hidden_dim, text_alpha);
    audio_position_cache = compute_positional_embeddings(512, model_hidden_dim, audio_alpha);

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
    ggml_backend_t backend,
    std::mt19937& rng,
    ggml_gallocr_t galloc_in
) {
    std::unique_ptr<T2SInputs> inputs;
    try {
        inputs = std::make_unique<T2SInputs>(backend);
    } catch (const std::exception& error) {
        std::cerr << "[T2S] Failed to initialize session inputs: " << error.what() << "\n";
        return {};
    }
    nn::KVCache attention_cache;
    if (!attention_cache.allocate(
            backend, head_dim, 512, n_heads, n_layers, attention_cache_config)) {
        std::cerr << "[T2S] Failed to allocate attention cache.\n";
        return {};
    }

    if (!word_embeddings.weight.is_bound() || !audio_embeddings.weight.is_bound() ||
        !bert_proj.weight.is_bound() || !bert_proj.bias.is_bound() || !predict.weight.is_bound()) {
        std::cerr << "[T2S] Error: Missing model weights in GGUF weight mapping!\n";
        return {};
    }

    int text_len = (int)(prompt_phones.size() + target_phones.size());
    std::vector<int32_t> text_ids_vec;
    text_ids_vec.reserve(text_len);
    text_ids_vec.insert(text_ids_vec.end(), prompt_phones.begin(), prompt_phones.end());
    text_ids_vec.insert(text_ids_vec.end(), target_phones.begin(), target_phones.end());

    std::vector<int32_t> current_audio_ids = prompt_semantics;
    std::vector<int32_t> generated_semantics;
    int total_decoded = 0;

    struct SchedGuard {
        ggml_backend_t blas_backend = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ~SchedGuard() {
            if (sched) ggml_backend_sched_free(sched);
            if (blas_backend) ggml_backend_free(blas_backend);
        }
    } guard;

    const char* bname = ggml_backend_name(backend);
    if (bname && std::strcmp(bname, "CPU") == 0) {
        guard.blas_backend = ggml_backend_init_by_name("BLAS", nullptr);
        if (guard.blas_backend) {
            ggml_backend_t sched_backends[2] = { guard.blas_backend, backend };
            guard.sched = ggml_backend_sched_new(sched_backends, nullptr, 2, 4096, false, false);
        }
    }

    // Context size 4MB, allocated once and reset per step to stabilize graph keys
    nn::Context step_context(4 * 1024 * 1024);

    bool is_local_galloc = false;
    ggml_gallocr_t galloc = galloc_in;
    if (!galloc && !guard.sched) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        is_local_galloc = true;
        if (!galloc) {
            std::cerr << "[T2S] Error: Failed to create graph allocator (gallocr)!\n";
            return {};
        }
    }

    std::vector<float> temp_bert;
    std::vector<float> text_pe_data;
    std::vector<float> audio_pe_data;
    std::vector<float> mask_data;
    std::vector<float> host_logits(1025);
    int32_t last_token = 0;

    const auto accept_logits = [&](ggml_tensor* logits_tensor) {
        host_logits.resize(1025);
        ggml_backend_tensor_get(logits_tensor, host_logits.data(), 0, 1025 * sizeof(float));
        const size_t greedy_vocab_size = total_decoded < 11 ? 1024 : host_logits.size();
        const int32_t greedy_token = static_cast<int32_t>(std::distance(
            host_logits.begin(),
            std::max_element(host_logits.begin(), host_logits.begin() + greedy_vocab_size)));
        const int32_t next_token = sample_next_token(host_logits, current_audio_ids, total_decoded, rng);
        if (next_token == 1024 || greedy_token == 1024) return false;
        current_audio_ids.push_back(next_token);
        generated_semantics.push_back(next_token);
        total_decoded++;
        return true;
    };

    // Prefill is request-shaped and intentionally built once per request.
    if (max_len > 0 && text_len + static_cast<int>(current_audio_ids.size()) < 512) {
        const int audio_len = static_cast<int>(current_audio_ids.size());
        step_context.reset();
        ggml_cgraph* prefill_graph = step_context.create_graph();
        ggml_tensor* prefill_logits = prefill(
            step_context, *inputs, attention_cache, prefill_graph, text_len, audio_len,
            text_ids_vec, current_audio_ids, temp_bert, text_pe_data, audio_pe_data, mask_data,
            bert_features, backend);
        bool allocated = false;
        if (guard.sched) {
            ggml_backend_sched_reset(guard.sched);
            allocated = ggml_backend_sched_alloc_graph(guard.sched, prefill_graph);
        } else {
            allocated = ggml_gallocr_alloc_graph(galloc, prefill_graph);
        }
        if (!allocated) {
            std::cerr << "[T2S] Error: Failed to allocate prefill graph.\n";
            if (galloc && is_local_galloc) ggml_gallocr_free(galloc);
            return {};
        }
        step_context.materialize();
        if (guard.sched) ggml_backend_sched_graph_compute(guard.sched, prefill_graph);
        else ggml_ops_ext::ops_backend_graph_compute(backend, prefill_graph);
        if (!accept_logits(prefill_logits)) {
            if (galloc && is_local_galloc) ggml_gallocr_free(galloc);
            return generated_semantics;
        }
    }

    if (galloc && is_local_galloc) {
        ggml_gallocr_free(galloc);
        galloc = nullptr;
    }

    if (total_decoded < max_len && text_len + static_cast<int>(current_audio_ids.size()) < 512) {
        nn::Context decode_context(4 * 1024 * 1024);
        ggml_context* ctx_decode = decode_context.native_handle();
        int32_t cache_position = 0;
        int32_t valid_length = 0;
        audio_pe_data.resize(512);
        ggml_tensor* position_tensor = decode_context.input<int32_t>(
            "decode.cache_position", {1}, nn::data::borrow(&cache_position, 1));
        ggml_tensor* length_tensor = decode_context.input<int32_t>(
            "decode.valid_length", {1}, nn::data::borrow(&valid_length, 1));
        ggml_tensor* position_embedding = decode_context.input<float>(
            "decode.audio_position", {512}, nn::data::borrow(audio_pe_data));
        ggml_tensor* token_view = ggml_view_1d(ctx_decode, inputs->token, 1, 0);
        ggml_tensor* x = audio_embeddings(ctx_decode, token_view);
        x = ggml_add(ctx_decode, x, position_embedding);
        x = decoder.decode(
            decode_context, x, attention_cache, position_tensor, length_tensor, backend);
        ggml_tensor* logits = predict(ctx_decode, x);
        ggml_cgraph* decode_graph = decode_context.build(logits);
        nn::Executor decode_executor(backend);
        decode_executor.prepare(decode_context, decode_graph);

        while (total_decoded < max_len) {
            const int audio_len = static_cast<int>(current_audio_ids.size());
            valid_length = text_len + audio_len;
            if (valid_length >= 512) break;
            cache_position = valid_length - 1;
            last_token = current_audio_ids.back();
            inputs->context.write(inputs->token, &last_token, 1);
            const auto pos_begin = audio_position_cache.begin() + (audio_len - 1) * 512;
            std::copy(pos_begin, pos_begin + 512, audio_pe_data.begin());
            decode_context.bind(position_tensor, nn::data::borrow(&cache_position, 1));
            decode_context.bind(length_tensor, nn::data::borrow(&valid_length, 1));
            decode_context.bind(position_embedding, nn::data::borrow(audio_pe_data));
            decode_executor.compute(decode_context, decode_graph);
            if (!accept_logits(logits)) break;
        }
    }
    return generated_semantics;
}

struct ggml_tensor* T2SModel::prefill(
    nn::Context& step_context,
    T2SInputs& inputs,
    nn::KVCache& attention_cache,
    struct ggml_cgraph* cgraph,
    int text_len,
    int audio_len,
    const std::vector<int32_t>& text_ids_vec,
    const std::vector<int32_t>& current_audio_ids,
    std::vector<float>& temp_bert,
    std::vector<float>& text_pe_data,
    std::vector<float>& audio_pe_data,
    std::vector<float>& mask_data,
    struct ggml_tensor* bert_features,
    ggml_backend_t backend
) {
    struct ggml_context* ctx_step = step_context.native_handle();
    struct ggml_tensor* x = nullptr;
    struct ggml_tensor* bert_features_local = nullptr;
    struct ggml_tensor* text_ids_tensor_view = nullptr;
    struct ggml_tensor* audio_ids_tensor_view = nullptr;
    struct ggml_tensor* text_pe = nullptr;
    struct ggml_tensor* audio_pe = nullptr;
    struct ggml_tensor* mask = nullptr;

    struct ggml_tensor* t_emb = nullptr;
    struct ggml_tensor* text_fused = nullptr;
    struct ggml_tensor* text_rep = nullptr;
    struct ggml_tensor* audio_rep = nullptr;

    temp_bert.resize(1024 * text_len);
        if (bert_features->buffer) {
            ggml_backend_tensor_get(bert_features, temp_bert.data(), 0, temp_bert.size() * sizeof(float));
        } else if (bert_features->data) {
            std::memcpy(temp_bert.data(), bert_features->data, temp_bert.size() * sizeof(float));
        }

        // First step: Process prompt phones and prompt semantics entirely
        inputs.context.write(inputs.bert_features, temp_bert.data(), temp_bert.size());

        bert_features_local = ggml_view_2d(ctx_step, inputs.bert_features, 1024, text_len, inputs.bert_features->nb[1], 0);

        struct ggml_tensor* bert_proj_aligned = bert_proj(ctx_step, bert_features_local);

        // Text embeddings
        inputs.context.write(inputs.text_ids, text_ids_vec.data(), text_len);

        text_ids_tensor_view = ggml_view_1d(ctx_step, inputs.text_ids, text_len, 0);

        t_emb = word_embeddings(ctx_step, text_ids_tensor_view);
        text_fused = ggml_add(ctx_step, t_emb, bert_proj_aligned);

        text_pe_data.assign(text_position_cache.begin(), text_position_cache.begin() + text_len * 512);
        text_pe = step_context.input<float>(
            "text_position", {512, text_len}, nn::data::borrow(text_pe_data));

        text_rep = ggml_add(ctx_step, text_fused, text_pe);



        // Audio embeddings
        inputs.context.write(inputs.audio_ids, current_audio_ids.data(), audio_len);

        audio_ids_tensor_view = ggml_view_1d(ctx_step, inputs.audio_ids, audio_len, 0);

        struct ggml_tensor* a_emb = audio_embeddings(ctx_step, audio_ids_tensor_view);

        audio_pe_data.assign(audio_position_cache.begin(), audio_position_cache.begin() + audio_len * 512);
        audio_pe = step_context.input<float>(
            "audio_position", {512, audio_len}, nn::data::borrow(audio_pe_data));

        audio_rep = ggml_add(ctx_step, a_emb, audio_pe);

    x = ggml_concat(ctx_step, text_rep, audio_rep, 1);

    int total_len = text_len + audio_len;

    // Generate mask
    mask_data = compute_prefix_causal_mask(total_len, text_len, n_heads);
    mask = step_context.input<float>(
        "attention_mask", {total_len, total_len, n_heads}, nn::data::borrow(mask_data));
 
    int q_len = total_len;
    const int hidden_dim = n_heads * head_dim;
 
    // Execute decoder stack
    x = decoder.prefill(step_context, x, attention_cache, mask, cgraph, backend);

    // Predict logits
    struct ggml_tensor* last_token_rep = ggml_view_2d(ctx_step, x, hidden_dim, 1, x->nb[1], (q_len - 1) * x->nb[1]);
    struct ggml_tensor* logits_tensor = predict(ctx_step, last_token_rep);
    ggml_build_forward_expand(cgraph, logits_tensor);
    return logits_tensor;
}

int32_t T2SModel::sample_next_token(
    std::vector<float>& host_logits,
    const std::vector<int32_t>& current_audio_ids,
    int total_decoded,
    std::mt19937& rng
) {
    if (total_decoded < 11) {
        host_logits.resize(1024);
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

    bool align_mode = (std::getenv("T2S_ALIGNMENT") != nullptr);
    bool is_greedy = align_mode || (temp <= 0.0f) || (top_k == 1);
    return sample_logits(host_logits, current_audio_ids, temp, top_k, top_p, rep_penalty, is_greedy, rng);
}

} // namespace gpt_sovits
