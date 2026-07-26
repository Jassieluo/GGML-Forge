#include "providers/gpt_sovits/models/vits/vits.h"
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
#include <utility>
#include <cstdint>
#include <iomanip>
#include <filesystem>
#include <exception>
#include <memory>
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

static ModelProfile structural_profile(const std::string& version) {
    ModelProfile profile;
    profile.exact_version = canonical_model_version(version);
    profile.vits_version = coarse_version_from_string(version);
    profile.is_classic = profile.vits_version > 0 && profile.vits_version < 3;
    profile.uses_cfm = version == "v3" || version == "v4";
    profile.requires_sv_emb = version == "v2Pro" || version == "v2ProPlus";
    profile.vocoder_architecture = version == "v3" ? "bigvgan-v2" :
                                   version == "v4" ? "hifigan" : "sovits";
    profile.upsample_rates = version == "v3" ? std::vector<int>{4, 4, 2, 2, 2, 2} :
                             version == "v4" ? std::vector<int>{10, 6, 2, 2, 2} :
                                               std::vector<int>{10, 8, 2, 2, 2};
    return profile;
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
    const auto dim = [](const nn::Parameter& parameter, size_t axis) -> int64_t {
        return parameter.is_bound() && axis < parameter.logical_shape().size()
            ? parameter.logical_shape()[axis] : 0;
    };

    next.upsample_initial_channel = static_cast<int>(dim(generator->pre.weight, 2));
    next.ge_dim = speaker ? static_cast<int>(dim(speaker->activation.weight, 0)) : 0;
    if (next.ge_dim == 0) next.ge_dim = static_cast<int>(dim(reference_encoder.output.bias, 0));

    if (speaker) {
        next.sv_emb_dim = static_cast<int>(std::max(
            dim(speaker->projection.weight, 0), dim(speaker->projection.weight, 1)));
    }
    if (!next.requires_sv_emb) {
        next.sv_emb_dim = 0;
    }

    next.ref_enc_channels = static_cast<int>(dim(reference_encoder.spectral0.weight, 0));

    next.upsample_kernel_sizes.clear();
    for (size_t i = 0; i < generator->upsample_count(); ++i) {
        next.upsample_kernel_sizes.push_back(
            static_cast<int>(dim(generator->upsample(i).weight, 0)));
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

bool VITSModel::configure_modules() {
    if (generator) return true;
    if (profile.exact_version.empty() || profile.vits_version <= 0) return false;
    if (profile.requires_sv_emb) {
        speaker = &submodule<SpeakerStack>("speaker");
    }
    generator = &submodule<vits::Generator>("generator", profile);
    return true;
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

    if (!configure_modules()) return false;

    nn::io::LoadResult loaded = nn::io::load_into(*this, *source, backend);
    if (!loaded) {
        std::cerr << "[VITS] " << loaded.error << std::endl;
        return false;
    }
    model_fingerprint = fingerprint_file(path);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS] Loaded canonical VITS parameters." << std::endl;

    this->to(backend);

    refresh_profile();
    const auto* cfm_model = dynamic_cast<const VITSModelCFM*>(this);
    const bool has_cfm = cfm_model && cfm_model->estimator.is_bound();
    const bool has_sv_conditioning = speaker && speaker->projection.weight.is_bound() &&
                                     speaker->activation.weight.is_bound() &&
                                     speaker->output_projection.weight.is_bound();
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
    if (!quantizer.codebook.weight.is_bound()) {
        std::cerr << "[VITS] Error: Missing quantizer codebook!" << std::endl;
        return res;
    }
    struct ggml_tensor* decoded = quantizer.codebook.forward(context, prompt_semantics);
    if (!decoded) {
        std::cerr << "[VITS] Error: VQ decode failed!" << std::endl;
        return res;
    }
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 1 VQ Decode done." << std::endl;

    // Step 2: Match the semantic rate expected by the acoustic decoder. V1
    // semantics are already at the decoder rate (stride 1), while V2 and
    // later artifacts carry 25 Hz semantics that must be expanded to 50 Hz.
    struct ggml_tensor* interp = decoded;
    if (profile.semantic_frame_stride == 2) {
        interp = nn::F::interpolate_nearest_2x(ctx_graph, decoded);
    } else if (profile.semantic_frame_stride != 1) {
        std::cerr << "[VITS] Unsupported semantic frame stride: "
                  << profile.semantic_frame_stride << std::endl;
        return res;
    }
    res.T_y = (int)interp->ne[1];
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 2 interpolation done." << std::endl;

    // Step 3: SSL Projection - 768 -> 192 channels via enc_p.ssl_proj
    struct ggml_tensor* y = semantic.ssl_projection.forward(context, interp, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 3 SSL Projection done." << std::endl;

    // Step 4: Load speaker embedding (ge)
    struct ggml_tensor* ge = refer_audio;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
    } else {
        int64_t ge_dim = 512;
        if (speaker && speaker->activation.weight.is_bound()) {
            ge_dim = speaker->activation.weight.tensor()->ne[0];
        }
        ge = context.empty<float>("vits.speaker_embedding", {ge_dim, 1});
        ge = ggml_fill(ctx_graph, ge, 0.0f);
    }
    res.ge = ge;

    struct ggml_tensor* ge_512 = ge;
    if (speaker && speaker->output_projection.weight.is_bound() && ge) {
        ge_512 = speaker->output_projection.forward(context, ge);
    }
    res.ge_512 = ge_512;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 4 speaker embedding done." << std::endl;

    // Step 5: encoder_ssl (3 layers) on ssl features
    struct ggml_tensor* y_enc = semantic.ssl_encoder.forward(context, y, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 5 encoder_ssl done." << std::endl;

    // Step 6: encoder_text (6 layers) on phone embeddings
    struct ggml_tensor* text_emb = semantic.text_embedding.forward(context, phone_ids);

    struct ggml_tensor* text_enc = semantic.text_encoder.forward(context, text_emb, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 6 encoder_text done." << std::endl;

    // Step 7: MRTE - cross-attention between y_enc and text_enc with speaker conditioning
    struct ggml_tensor* mrte_out = semantic.mrte.forward(context, y_enc, text_enc, ge_512, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 7 MRTE done." << std::endl;

    // Step 8: encoder2 (3 layers)
    res.y2 = semantic.output_encoder.forward(context, mrte_out, backend);
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Encode] Step 8 encoder2 done." << std::endl;

    return res;
}

std::unique_ptr<VITSModel> VITSModel::create(const std::string& path) {
    std::unique_ptr<nn::io::GGUFSource> source;
    try {
        source = std::make_unique<nn::io::GGUFSource>(path);
    } catch (const std::exception& error) {
        std::cerr << "[VITSModel::create] " << error.what() << std::endl;
        return nullptr;
    }

    const struct gguf_context* ctx_gguf = source->metadata_context();
    const int kid_ver = gguf_find_key(ctx_gguf, "gpt_sovits.version");
    const std::string version = kid_ver >= 0 && gguf_get_kv_type(ctx_gguf, kid_ver) == GGUF_TYPE_STRING
        ? canonical_model_version(gguf_get_val_str(ctx_gguf, kid_ver))
        : std::string{};

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

std::unique_ptr<VITSModel> VITSModel::create_for_version(const std::string& version) {
    const std::string canonical = canonical_model_version(version);
    if (canonical.empty()) return nullptr;
    std::unique_ptr<VITSModel> model = canonical == "v3" || canonical == "v4"
        ? std::unique_ptr<VITSModel>(std::make_unique<VITSModelCFM>())
        : std::unique_ptr<VITSModel>(std::make_unique<VITSModelClassic>());
    model->profile = structural_profile(canonical);
    return model->configure_modules() ? std::move(model) : nullptr;
}

} // namespace gpt_sovits
