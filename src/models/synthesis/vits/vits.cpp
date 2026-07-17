#include "vits.h"
#include "ops/ops.h"
#include "nn/nn.h"
#include "nn/io/gguf.h"
#include "nn/io/load.h"
#include <cstdlib>
#include <random>
#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <iomanip>
#include <filesystem>
#include <exception>
#include <memory>
#include <optional>
#include <string_view>
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

namespace gpt_sovits {

static std::string fingerprint_file(const std::string& path) {
    const std::filesystem::path file_path = std::filesystem::u8path(path);
    std::ifstream input(file_path, std::ios::binary);
    if (!input) return "unavailable";

    uint64_t hash = 1469598103934665603ULL;
    auto mix = [&](const void* data, size_t size) {
        const auto* bytes = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;
        }
    };

    std::error_code ec;
    const uint64_t file_size = std::filesystem::file_size(file_path, ec);
    if (ec) return "unavailable";
    const auto write_time = std::filesystem::last_write_time(file_path, ec);
    if (ec) return "unavailable";
    const auto write_ticks = write_time.time_since_epoch().count();
    mix(&file_size, sizeof(file_size));
    mix(&write_ticks, sizeof(write_ticks));

    constexpr uint64_t sample_size = 1024 * 1024;
    std::vector<char> buffer(1024 * 1024);
    const uint64_t last_offset = file_size > sample_size ? file_size - sample_size : 0;
    const uint64_t offsets[] = {0, last_offset / 2, last_offset};
    for (uint64_t offset : offsets) {
        input.clear();
        input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        input.read(buffer.data(), static_cast<std::streamsize>(std::min(sample_size, file_size - offset)));
        const std::streamsize count = input.gcount();
        mix(buffer.data(), static_cast<size_t>(count));
    }
    std::ostringstream value;
    value << std::hex << std::setfill('0') << std::setw(16) << hash;
    return value.str();
}

static void append_ints_to_stream(std::ostringstream& oss, const std::vector<int>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) oss << ",";
        oss << values[i];
    }
}

bool VITSModel::read_metadata(const struct gguf_context* ctx_gguf, const std::string& exact_version) {
    auto read_string = [&](const char* key, std::string& value) {
        const int id = gguf_find_key(ctx_gguf, key);
        if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_STRING) return false;
        value = gguf_get_val_str(ctx_gguf, id);
        return true;
    };
    auto read_u32 = [&](const char* key, int& value) {
        const int id = gguf_find_key(ctx_gguf, key);
        if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_UINT32) return false;
        value = static_cast<int>(gguf_get_val_u32(ctx_gguf, id));
        return true;
    };
    auto read_bool = [&](const char* key, bool& value) {
        const int id = gguf_find_key(ctx_gguf, key);
        if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_BOOL) return false;
        value = gguf_get_val_bool(ctx_gguf, id);
        return true;
    };
    auto read_f32 = [&](const char* key, float& value) {
        const int id = gguf_find_key(ctx_gguf, key);
        if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_FLOAT32) return false;
        value = gguf_get_val_f32(ctx_gguf, id);
        return true;
    };
    auto read_u32_array = [&](const char* key, std::vector<int>& values) {
        const int id = gguf_find_key(ctx_gguf, key);
        if (id < 0 || gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(ctx_gguf, id) != GGUF_TYPE_INT32) return false;
        const size_t count = gguf_get_arr_n(ctx_gguf, id);
        if (count == 0 || count > 16) return false;
        const auto* data = static_cast<const int32_t*>(gguf_get_arr_data(ctx_gguf, id));
        values.assign(data, data + count);
        return std::all_of(values.begin(), values.end(), [](int value) { return value > 0; });
    };

    std::string kind;
    std::string implementation;
    ModelProfile metadata;
    metadata.exact_version = exact_version;
    if (!read_string("tts.artifact.kind", kind) || kind != "audio-decoder" ||
        !read_string("tts.artifact.implementation", implementation) ||
        (implementation != "classic" && implementation != "cfm") ||
        !read_string("tts.vocoder.architecture", metadata.vocoder_architecture) ||
        !read_u32("gpt_sovits.vits.api_version", metadata.vits_version) ||
        !read_u32("gpt_sovits.frontend.symbol_version", metadata.symbol_version) ||
        !read_u32("gpt_sovits.t2s.expected_family", metadata.expected_t2s_family) ||
        !read_u32("gpt_sovits.vits.semantic_frame_stride", metadata.semantic_frame_stride) ||
        !read_bool("gpt_sovits.vits.requires_speaker_embedding", metadata.requires_sv_emb) ||
        !read_u32("tts.audio.output_sample_rate", metadata.output_sampling_rate) ||
        !read_u32("tts.audio.reference_sample_rate", metadata.reference_sampling_rate) ||
        !read_u32("tts.audio.prompt_sample_rate", metadata.prompt_mel_sampling_rate) ||
        !read_u32("tts.audio.filter_length", metadata.filter_length) ||
        !read_u32("tts.audio.hop_length", metadata.hop_length) ||
        !read_u32("tts.audio.window_length", metadata.win_length) ||
        !read_u32("tts.audio.mel_channels", metadata.prompt_mel_channels) ||
        !read_u32("tts.inference.default_steps", metadata.default_inference_steps) ||
        !read_f32("tts.acoustic.feature_rate_scale", metadata.feature_rate_scale) ||
        !read_u32("tts.acoustic.max_prompt_frames", metadata.max_prompt_frames) ||
        !read_u32_array("tts.vocoder.upsample_rates", metadata.upsample_rates)) {
        std::cerr << "[VITS] Required artifact contract metadata is missing or has the wrong type." << std::endl;
        return false;
    }

    metadata.uses_cfm = implementation == "cfm";
    metadata.is_classic = implementation == "classic";
    const bool valid = metadata.vits_version >= 1 && metadata.vits_version <= 4 &&
        (metadata.symbol_version == 1 || metadata.symbol_version == 2) &&
        metadata.expected_t2s_family >= 1 && metadata.expected_t2s_family <= 3 &&
        metadata.semantic_frame_stride > 0 && metadata.output_sampling_rate > 0 &&
        metadata.reference_sampling_rate > 0 && metadata.filter_length > 0 &&
        metadata.hop_length > 0 && metadata.win_length > 0 &&
        (!metadata.uses_cfm || (metadata.prompt_mel_sampling_rate > 0 &&
         metadata.prompt_mel_channels > 0 && metadata.default_inference_steps > 0 &&
         metadata.feature_rate_scale > 0.0f && metadata.max_prompt_frames > 0 &&
         (metadata.vocoder_architecture == "bigvgan-v2" || metadata.vocoder_architecture == "hifigan"))) &&
        (metadata.uses_cfm || metadata.vocoder_architecture == "sovits");
    if (!valid) {
        std::cerr << "[VITS] Artifact contract metadata contains invalid values." << std::endl;
        return false;
    }
    profile = std::move(metadata);
    return true;
}

void VITSModel::refresh_profile() {
    ModelProfile next = profile;

    struct ggml_tensor* sv_emb_w = get_tensor("sv_emb.weight");
    struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
    struct ggml_tensor* ref_enc_out = get_tensor("ref_enc.fc.fc.bias");

    next.upsample_initial_channel = static_cast<int>(get_logical_tensor_dim("dec.conv_pre.weight", 2));

    next.ge_dim = prelu_w ? static_cast<int>(get_logical_tensor_dim("prelu.weight", 0)) : 0;
    if (!prelu_w && ref_enc_out) {
        next.ge_dim = static_cast<int>(get_logical_tensor_dim("ref_enc.fc.fc.bias", 0));
    }

    if (sv_emb_w) {
        next.sv_emb_dim = static_cast<int>(std::max(
            get_logical_tensor_dim("sv_emb.weight", 0),
            get_logical_tensor_dim("sv_emb.weight", 1)));
    }
    if (!next.requires_sv_emb) {
        next.sv_emb_dim = 0;
    }

    next.ref_enc_channels = static_cast<int>(get_logical_tensor_dim("ref_enc.spectral.0.fc.weight", 0));

    next.upsample_kernel_sizes.clear();
    for (int i = 0; i < 6; ++i) {
        struct ggml_tensor* up_w = get_tensor("dec.ups." + std::to_string(i) + ".weight");
        if (!up_w) {
            up_w = get_tensor("dec.ups." + std::to_string(i) + ".0.weight");
        }
        if (up_w) {
            const std::string name = get_tensor("dec.ups." + std::to_string(i) + ".weight")
                ? "dec.ups." + std::to_string(i) + ".weight"
                : "dec.ups." + std::to_string(i) + ".0.weight";
            next.upsample_kernel_sizes.push_back(static_cast<int>(get_logical_tensor_dim(name, 0)));
        }
    }

    std::ostringstream id;
    id << next.exact_version
       << "|backend=" << (next.uses_cfm ? "cfm" : "classic")
       << "|vocoder=" << next.vocoder_architecture
       << "|ge=" << next.ge_dim
       << "|ref=" << next.ref_enc_channels
       << "|out_sr=" << next.output_sampling_rate
       << "|sv=" << (next.requires_sv_emb ? next.sv_emb_dim : 0)
       << "|up_init=" << next.upsample_initial_channel
       << "|up_k=";
    append_ints_to_stream(id, next.upsample_kernel_sizes);
    id << "|weights=" << model_fingerprint;
    next.profile_id = id.str();

    profile = std::move(next);
    cfm_steps = profile.default_inference_steps;
}

bool VITSModel::load(const std::string& path, ggml_backend_t backend) {
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loading VITS GGUF model: " << path << std::endl;
    if (!backend) return false;
    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[VITS] " << error.what() << std::endl;
        return false;
    }

    const gguf_context* metadata = source->metadata_context();
    const int architecture_key = gguf_find_key(metadata, "general.architecture");
    const int version_key = gguf_find_key(metadata, "gpt_sovits.version");
    if (architecture_key < 0 || gguf_get_kv_type(metadata, architecture_key) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata, architecture_key)) != "gpt_sovits_vits" ||
        version_key < 0 || gguf_get_kv_type(metadata, version_key) != GGUF_TYPE_STRING) {
        std::cerr << "[VITS] Invalid architecture or version metadata." << std::endl;
        return false;
    }
    const std::string exact_version = canonical_model_version(gguf_get_val_str(metadata, version_key));
    const int coarse_version = coarse_version_from_string(exact_version);
    if (exact_version.empty() || coarse_version == 0 || !read_metadata(metadata, exact_version)) {
        std::cerr << "[VITS] Invalid model version or artifact contract." << std::endl;
        return false;
    }

    try {
        for (size_t index = 0; index < source->size(); ++index) {
            const nn::io::TensorInfo& info = source->info(index);
            artifact_parameters.define(info.name);
        }
    } catch (const std::exception& error) {
        std::cerr << "[VITS] Invalid parameter inventory: " << error.what() << std::endl;
        return false;
    }
    auto artifact_mapper = [&](std::string_view path, const nn::Parameter&) -> std::optional<std::string> {
        auto key = artifact_parameters.key_for_registered_name(path);
        return key ? std::optional<std::string>(*key) : std::nullopt;
    };
    nn::io::LoadResult loaded = nn::io::load_into(artifact_parameters, *source, backend, artifact_mapper);
    if (!loaded) {
        std::cerr << "[VITS] " << loaded.error << std::endl;
        return false;
    }
    attach_state_dict(loaded.state);
    if (profile.vocoder_architecture == "bigvgan-v2") {
        constexpr std::string_view alpha_suffix = ".act.alpha";
        for (size_t index = 0; index < source->size(); ++index) {
            const std::string& name = source->info(index).name;
            if (name.size() < alpha_suffix.size() ||
                std::string_view(name).substr(name.size() - alpha_suffix.size()) != alpha_suffix) continue;
            const std::string prefix = name.substr(0, name.size() - alpha_suffix.size());
            if (!artifact_parameters.contains(prefix + ".upsample.filter_repeated") ||
                !artifact_parameters.contains(prefix + ".downsample.filter_repeated")) {
                std::cerr << "[VITS] Missing materialized alias-free filters for " << prefix
                          << "; re-export the V3 artifact." << std::endl;
                return false;
            }
        }
    }
    model_fingerprint = fingerprint_file(path);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loaded VITS successfully. Pre-computing Weight Normalization..." << std::endl;
    
    std::unordered_map<std::string, std::string> name_map;
    if (profile.vits_version >= 3) {
        VITSModelCFM* cfm_model = dynamic_cast<VITSModelCFM*>(this);
        if (cfm_model) {
            for (int l = 0; l < 22; ++l) {
                cfm_model->transformer_blocks[l].attn.n_heads = 16;
                cfm_model->transformer_blocks[l].attn.head_dim = 64;
            }
        }
        name_map["input_proj.weight"] = "cfm.estimator.input_embed.proj.weight";
        name_map["input_proj.bias"]   = "cfm.estimator.input_embed.proj.bias";
        name_map["proj_out.weight"]   = "cfm.estimator.proj_out.weight";
        name_map["proj_out.bias"]     = "cfm.estimator.proj_out.bias";
        
        for (int l = 0; l < 22; ++l) {
            std::string cpp_prefix = "transformer_blocks." + std::to_string(l) + ".";
            std::string gguf_prefix = "cfm.estimator.transformer_blocks." + std::to_string(l) + ".";
            
            name_map[cpp_prefix + "attn_norm.linear.weight"] = gguf_prefix + "attn_norm.linear.weight";
            name_map[cpp_prefix + "attn_norm.linear.bias"]   = gguf_prefix + "attn_norm.linear.bias";
            
            name_map[cpp_prefix + "attn.q_proj.weight"] = gguf_prefix + "attn.to_q.weight";
            name_map[cpp_prefix + "attn.q_proj.bias"]   = gguf_prefix + "attn.to_q.bias";
            name_map[cpp_prefix + "attn.k_proj.weight"] = gguf_prefix + "attn.to_k.weight";
            name_map[cpp_prefix + "attn.k_proj.bias"]   = gguf_prefix + "attn.to_k.bias";
            name_map[cpp_prefix + "attn.v_proj.weight"] = gguf_prefix + "attn.to_v.weight";
            name_map[cpp_prefix + "attn.v_proj.bias"]   = gguf_prefix + "attn.to_v.bias";
            
            name_map[cpp_prefix + "attn.out_proj.weight"] = gguf_prefix + "attn.to_out.0.weight";
            name_map[cpp_prefix + "attn.out_proj.bias"]   = gguf_prefix + "attn.to_out.0.bias";
            
            name_map[cpp_prefix + "ff.w1.weight"] = gguf_prefix + "ff.ff.0.0.weight";
            name_map[cpp_prefix + "ff.w1.bias"]   = gguf_prefix + "ff.ff.0.0.bias";
            name_map[cpp_prefix + "ff.w2.weight"] = gguf_prefix + "ff.ff.2.weight";
            name_map[cpp_prefix + "ff.w2.bias"]   = gguf_prefix + "ff.ff.2.bias";
        }
    }
    auto module_mapper = [&](std::string_view path, const nn::Parameter&) -> std::optional<std::string> {
        auto mapped = name_map.find(std::string(path));
        return mapped == name_map.end() ? std::optional<std::string>(path)
                                        : std::optional<std::string>(mapped->second);
    };
    const std::string bind_error = nn::io::bind_from(*this, artifact_parameters, module_mapper);
    if (!bind_error.empty()) {
        std::cerr << "[VITS] " << bind_error << std::endl;
        return false;
    }
    this->to(backend);

    gen_weights.conv_pre_w = get_tensor("dec.conv_pre.weight");
    gen_weights.conv_pre_b = get_tensor("dec.conv_pre.bias");
    gen_weights.cond_w = get_tensor("dec.cond.weight");
    gen_weights.cond_b = get_tensor("dec.cond.bias");
    gen_weights.conv_post_w = get_tensor("dec.conv_post.weight");
    gen_weights.conv_post_b = get_tensor("dec.conv_post.bias");
    
    gen_weights.ups_w.resize(6, nullptr);
    gen_weights.ups_b.resize(6, nullptr);
    for (int i = 0; i < 6; ++i) {
        gen_weights.ups_w[i] = get_tensor("dec.ups." + std::to_string(i) + ".weight");
        gen_weights.ups_b[i] = get_tensor("dec.ups." + std::to_string(i) + ".bias");
    }
    
    for (int block_idx = 0; block_idx < 18; ++block_idx) {
        for (int l = 0; l < 3; ++l) {
            std::string prefix1 = "dec.resblocks." + std::to_string(block_idx) + ".convs1." + std::to_string(l);
            std::string prefix2 = "dec.resblocks." + std::to_string(block_idx) + ".convs2." + std::to_string(l);
            gen_weights.resblocks[block_idx][l].c1_w = get_tensor(prefix1 + ".weight");
            gen_weights.resblocks[block_idx][l].c1_b = get_tensor(prefix1 + ".bias");
            gen_weights.resblocks[block_idx][l].c2_w = get_tensor(prefix2 + ".weight");
            gen_weights.resblocks[block_idx][l].c2_b = get_tensor(prefix2 + ".bias");
        }
    }

    refresh_profile();
    const bool has_cfm = get_tensor("cfm.estimator.input_embed.proj.weight") != nullptr;
    const bool has_sv_conditioning = get_tensor("sv_emb.weight") && get_tensor("prelu.weight") &&
                                      get_tensor("ge_to512.weight");
    if (has_cfm != profile.uses_cfm || has_sv_conditioning != profile.requires_sv_emb ||
        profile.ge_dim <= 0 || profile.ref_enc_channels <= 0 ||
        profile.upsample_initial_channel <= 0 ||
        profile.upsample_kernel_sizes.size() != profile.upsample_rates.size() ||
        (profile.requires_sv_emb && profile.sv_emb_dim <= 0)) {
        std::cerr << "[VITS] GGUF tensors do not match declared version " << profile.exact_version << std::endl;
        return false;
    }
    std::cout << "[VITS Profile] " << profile.summary() << std::endl;
    if (profile.requires_sv_emb) {
        std::cout << "[VITS Profile] This model requires a " << profile.sv_emb_dim
                  << "-float ERes2Net speaker vector for Python-equivalent V2Pro conditioning." << std::endl;
    }

    ggml_backend_synchronize(backend);
    return true;
}

VITSModel::EncodeResult VITSModel::encode_semantic_base(
    nn::Context& context,
    struct ggml_tensor* phone_ids,
    struct ggml_tensor* prompt_semantics,
    struct ggml_tensor* refer_audio,
    ggml_backend_t backend
) {
    struct ggml_context* ctx_graph = context.native_handle();
    EncodeResult res;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Starting encode_semantic_base..." << std::endl;

    // Step 1: VQ Decode - semantic token IDs -> continuous features [768, N]
    struct ggml_tensor* codebook = get_tensor("quantizer.vq.layers.0._codebook.embed");
    if (!codebook) {
        std::cerr << "[VITS] Error: Missing quantizer codebook!" << std::endl;
        return res;
    }
    struct ggml_tensor* decoded = ggml_get_rows(ctx_graph, codebook, prompt_semantics);
    if (!decoded) {
        std::cerr << "[VITS] Error: VQ decode failed!" << std::endl;
        return res;
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 1 VQ Decode done." << std::endl;

    // Step 2: Interpolate from 25Hz to 50Hz (2x nearest-neighbor)
    struct ggml_tensor* interp = nn::F::interpolate_nearest_2x(ctx_graph, decoded);
    res.T_y = (int)interp->ne[1];
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 2 interpolation done." << std::endl;

    // Step 3: SSL Projection - 768 -> 192 channels via enc_p.ssl_proj
    struct ggml_tensor* ssl_proj_w = get_tensor("enc_p.ssl_proj.weight");
    struct ggml_tensor* ssl_proj_b = get_tensor("enc_p.ssl_proj.bias");
    struct ggml_tensor* y = interp;
    if (ssl_proj_w && ssl_proj_b) {
        y = nn::F::conv1d(ctx_graph, interp, ssl_proj_w, ssl_proj_b, 1, 0, 1, 1, backend);
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 3 SSL Projection done." << std::endl;

    // Step 4: Load speaker embedding (ge)
    struct ggml_tensor* ge = refer_audio;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
    } else {
        int64_t ge_dim = 512;
        struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
        if (prelu_w) {
            ge_dim = prelu_w->ne[0];
        }
        ge = context.empty<float>("vits.speaker_embedding", {ge_dim, 1});
        ge = ggml_fill(ctx_graph, ge, 0.0f);
    }
    res.ge = ge;

    struct ggml_tensor* ge_512 = ge;
    struct ggml_tensor* ge_to512_w = get_tensor("ge_to512.weight");
    struct ggml_tensor* ge_to512_b = get_tensor("ge_to512.bias");
    if (ge_to512_w && ge_to512_b && ge) {
        ge_512 = nn::F::linear(ctx_graph, ge, ge_to512_w, ge_to512_b, backend);
    }
    res.ge_512 = ge_512;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 4 speaker embedding done." << std::endl;

    int n_head = 2;
    int d_k = 96;  // 192 / 2

    // Step 5: encoder_ssl (3 layers) on ssl features
    struct ggml_tensor* y_enc = build_encoder(ctx_graph, y, *this, "enc_p.encoder_ssl", 3, n_head, d_k, res.T_y, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 5 encoder_ssl done." << std::endl;

    // Step 6: encoder_text (6 layers) on phone embeddings
    struct ggml_tensor* text_emb_w = get_tensor("enc_p.text_embedding.weight");
    int text_len = (int)phone_ids->ne[0];
    struct ggml_tensor* text_emb = ggml_get_rows(ctx_graph, text_emb_w, phone_ids);  // [192, text_len]

    struct ggml_tensor* text_enc = build_encoder(ctx_graph, text_emb, *this, "enc_p.encoder_text", 6, n_head, d_k, text_len, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 6 encoder_text done." << std::endl;

    // Step 7: MRTE - cross-attention between y_enc and text_enc with speaker conditioning
    struct ggml_tensor* mrte_out = build_mrte(ctx_graph, y_enc, text_enc, ge_512, *this, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 7 MRTE done." << std::endl;

    // Step 8: encoder2 (3 layers)
    res.y2 = build_encoder(ctx_graph, mrte_out, *this, "enc_p.encoder2", 3, n_head, d_k, res.T_y, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 8 encoder2 done." << std::endl;

    return res;
}

std::unique_ptr<VITSModel> VITSModel::create(const std::string& path) {
    struct ggml_context* ggml_ctx_backend = nullptr;
    struct gguf_init_params params_backend = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ggml_ctx_backend
    };
    struct gguf_context* ctx_gguf = gguf_init_from_file(path.c_str(), params_backend);
    if (!ctx_gguf) {
        return nullptr;
    }

    const int kid_ver = gguf_find_key(ctx_gguf, "gpt_sovits.version");
    const std::string version = kid_ver >= 0 && gguf_get_kv_type(ctx_gguf, kid_ver) == GGUF_TYPE_STRING
        ? canonical_model_version(gguf_get_val_str(ctx_gguf, kid_ver))
        : std::string{};

    gguf_free(ctx_gguf);
    if (ggml_ctx_backend) {
        ggml_free(ggml_ctx_backend);
    }

    if (version.empty()) {
        std::cerr << "[VITSModel::create] Required exact string metadata gpt_sovits.version is missing or invalid in "
                  << path << std::endl;
        return nullptr;
    }

    if (version == "v3" || version == "v4") {
        return std::make_unique<VITSModelCFM>();
    }
    return std::make_unique<VITSModelClassic>();
}

} // namespace gpt_sovits
