#include "voice_manager.h"
// Legacy provider compatibility service.
#include "gpt_sovits_internal.h"
#include "audio/audio_io.h"
#include "audio/resample.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <type_traits>
#include <iomanip>

namespace gpt_sovits {

static constexpr uint32_t FEATURE_CACHE_MAGIC = 0x43535454; // "TTSC"
static constexpr uint64_t MAX_CACHE_FIELD_BYTES = 256ull * 1024 * 1024;

static uint64_t cache_namespace_hash(const ModelProfile& profile, int device_type) {
    uint64_t hash = 1469598103934665603ull;
    const std::string identity = profile.profile_id + "\n" + std::to_string(device_type);
    for (unsigned char byte : identity) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

static const char* cache_backend_name(int device_type) {
    if (device_type == 1) return "cuda";
    if (device_type == 2) return "sycl";
    return "cpu";
}

static std::filesystem::path versioned_features_path(
    const std::filesystem::path& root,
    const ModelProfile& profile,
    int device_type,
    const std::string& emotion
) {
    std::ostringstream profile_hash;
    profile_hash << std::hex << std::setw(16) << std::setfill('0')
                 << cache_namespace_hash(profile, device_type);
    return root / std::filesystem::u8path(profile.exact_version) /
        profile_hash.str() / cache_backend_name(device_type) /
        std::filesystem::u8path(emotion + ".features.bin");
}

// Shared log control state
extern bool g_log_enabled;

static int32_t get_backend_device_type(ggml_backend_t backend) {
    if (!backend) return 0; // CPU
    const char* name = ggml_backend_name(backend);
    if (!name) return 0;
    std::string sname(name);
    if (sname.find("CUDA") != std::string::npos) return 1;
    if (sname.find("SYCL") != std::string::npos) return 2;
    return 0; // CPU
}

static void write_string(std::ofstream& out, const std::string& value) {
    uint32_t len = static_cast<uint32_t>(value.size());
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    if (len > 0) {
        out.write(value.data(), len);
    }
}

static bool read_string(std::ifstream& in, std::string& value) {
    uint32_t len = 0;
    if (!in.read(reinterpret_cast<char*>(&len), sizeof(len))) {
        return false;
    }
    const std::streampos position = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    in.seekg(position);
    if (position < 0 || end < position || len > MAX_CACHE_FIELD_BYTES ||
        static_cast<uint64_t>(end - position) < len) {
        return false;
    }
    value.resize(len);
    if (len > 0 && !in.read(&value[0], len)) {
        return false;
    }
    return true;
}

static void write_float_vector(std::ofstream& out, const std::vector<float>& values) {
    uint32_t count = static_cast<uint32_t>(values.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (count > 0) {
        out.write(reinterpret_cast<const char*>(values.data()), count * sizeof(float));
    }
}

static bool read_float_vector(std::ifstream& in, std::vector<float>& values) {
    uint32_t count = 0;
    if (!in.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(float);
    const std::streampos position = in.tellg();
    in.seekg(0, std::ios::end);
    const std::streampos end = in.tellg();
    in.seekg(position);
    if (position < 0 || end < position || bytes > MAX_CACHE_FIELD_BYTES ||
        static_cast<uint64_t>(end - position) < bytes) {
        return false;
    }
    values.resize(count);
    if (count > 0 && !in.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))) {
        return false;
    }
    return true;
}

static bool cache_matches_current_profile(
    const PromptCache& cache,
    Impl* impl,
    int expected_device_type,
    std::string& reason
) {
    if (!impl || !impl->vits) {
        reason = "VITS model is not loaded";
        return false;
    }

    const ModelProfile& profile = impl->vits->profile;
    if (cache.vits_version != profile.vits_version) {
        reason = "coarse VITS version mismatch";
        return false;
    }
    if (cache.ge_dim != profile.ge_dim) {
        reason = "speaker embedding dimension mismatch";
        return false;
    }
    if (cache.speaker_embedding.size() != static_cast<size_t>(profile.ge_dim)) {
        reason = "speaker embedding payload size mismatch";
        return false;
    }
    if (cache.device_type != -1 && cache.device_type != expected_device_type) {
        reason = "backend device type mismatch";
        return false;
    }
    if (!cache.model_profile_id.empty() && cache.model_profile_id != profile.profile_id) {
        reason = "model profile id mismatch";
        return false;
    }
    if (!cache.model_profile_id.empty() && cache.requires_sv_emb != profile.requires_sv_emb) {
        reason = "speaker vector requirement mismatch";
        return false;
    }
    if (cache.sv_emb_dim > 0 && cache.sv_emb_dim != profile.sv_emb_dim) {
        reason = "speaker vector dimension mismatch";
        return false;
    }
    if (profile.requires_sv_emb &&
        cache.sv_emb.size() != static_cast<size_t>(profile.sv_emb_dim)) {
        reason = "missing or invalid speaker vector payload";
        return false;
    }
    if (cache.ref_enc_channels > 0 && cache.ref_enc_channels != profile.ref_enc_channels) {
        reason = "reference encoder channel count mismatch";
        return false;
    }
    if (cache.output_sampling_rate > 0 && cache.output_sampling_rate != profile.output_sampling_rate) {
        reason = "output sampling rate mismatch";
        return false;
    }
    if (profile.uses_cfm) {
        const int mel_channels = profile.prompt_mel_channels > 0 ? profile.prompt_mel_channels : 100;
        if (cache.prompt_mel.empty() || (cache.prompt_mel.size() % static_cast<size_t>(mel_channels)) != 0) {
            reason = "missing or invalid CFM prompt mel cache";
            return false;
        }
        if (cache.prompt_fea_ref.empty() || (cache.prompt_fea_ref.size() % 512) != 0) {
            reason = "missing or invalid CFM prompt encoder cache";
            return false;
        }
    }

    return true;
}

VoiceManagerImpl::VoiceManagerImpl(gpt_sovits_engine_t eng) : engine(eng) {}

bool voice_manager_parse_emotions_config(
    const std::string& json_str,
    std::string& out_name,
    std::string& out_lang,
    std::unordered_map<std::string, EmotionEntry>& out_emotions
) {
    out_name.clear();
    out_lang.clear();
    out_emotions.clear();

    auto extract_string = [](const std::string& s, size_t& pos) -> std::string {
        size_t start = s.find('"', pos);
        if (start == std::string::npos) return "";
        size_t end = s.find('"', start + 1);
        if (end == std::string::npos) return "";
        pos = end + 1;
        return s.substr(start + 1, end - start - 1);
    };

    auto skip_ws = [](const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;
    };

    size_t pos = 0;
    skip_ws(json_str, pos);
    if (pos >= json_str.size() || json_str[pos] != '{') return false;
    pos++; // skip opening {
    for (int depth = 1; depth > 0 && pos < json_str.size(); ) {
        skip_ws(json_str, pos);
        if (pos >= json_str.size()) break;

        if (json_str[pos] == '}') { depth--; pos++; continue; }
        if (json_str[pos] == ',') { pos++; continue; }

        std::string key = extract_string(json_str, pos);
        if (key.empty()) break;

        skip_ws(json_str, pos);
        if (pos >= json_str.size() || json_str[pos] != ':') break;
        pos++; // skip :
        skip_ws(json_str, pos);

        if (key == "name" || key == "lang") {
            std::string val = extract_string(json_str, pos);
            if (key == "name") out_name = val;
            else out_lang = val;
        } else if (key == "emotions") {
            if (pos >= json_str.size() || json_str[pos] != '{') break;
            pos++; // skip {
            while (pos < json_str.size()) {
                skip_ws(json_str, pos);
                if (pos >= json_str.size()) break;
                if (json_str[pos] == '}') { pos++; break; }
                if (json_str[pos] == ',') { pos++; continue; }

                std::string emo_name = extract_string(json_str, pos);
                if (emo_name.empty()) break;

                skip_ws(json_str, pos);
                if (pos >= json_str.size() || json_str[pos] != ':') break;
                pos++;
                skip_ws(json_str, pos);
                if (pos >= json_str.size() || json_str[pos] != '{') break;
                pos++; // skip inner {
                EmotionEntry entry;
                while (pos < json_str.size()) {
                    skip_ws(json_str, pos);
                    if (pos >= json_str.size()) break;
                    if (json_str[pos] == '}') { pos++; break; }
                    if (json_str[pos] == ',') { pos++; continue; }

                    std::string inner_key = extract_string(json_str, pos);
                    if (inner_key.empty()) break;
                    skip_ws(json_str, pos);
                    if (pos >= json_str.size() || json_str[pos] != ':') break;
                    pos++;
                    skip_ws(json_str, pos);
                    std::string inner_val = extract_string(json_str, pos);

                    if (inner_key == "audio") entry.audio = inner_val;
                    else if (inner_key == "text") entry.text = inner_val;
                }
                if (!entry.audio.empty() && !entry.text.empty()) {
                    out_emotions[emo_name] = entry;
                }
            }
        }
    }

    return !out_name.empty() && !out_lang.empty() && !out_emotions.empty();
}

std::vector<float> voice_manager_load_wav_file(const std::string& filename, int& sample_rate) {
    forge::media::Audio decoded;
    std::string error;
    if (!forge::media::load_wav(std::filesystem::u8path(filename), decoded, error)) {
        std::cerr << "[VoiceManager WAV Loader] " << error << ": " << filename << "\n";
        return {};
    }
    std::vector<float> audio_data = forge::media::mix_to_mono(decoded);
    sample_rate = static_cast<int>(decoded.sample_rate);
    if (sample_rate == 32000 || sample_rate == 48000) {
        audio_data = forge::media::resample_mono(
            audio_data.data(), audio_data.size(), sample_rate, 16000);
        sample_rate = audio_data.empty() ? 0 : 16000;
    }
    if (g_log_enabled && !audio_data.empty()) {
        std::cout << "[VoiceManager WAV Loader] Loaded " << filename
                  << " | Samples: " << audio_data.size() << "\n";
    }
    return audio_data;
}

bool serialize_features(const std::filesystem::path& filepath, const PromptCache& cache) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    uint32_t magic = FEATURE_CACHE_MAGIC;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    int32_t vits_ver = static_cast<int32_t>(cache.vits_version);
    int32_t ge_dim = static_cast<int32_t>(cache.ge_dim);
    int32_t device_type = static_cast<int32_t>(cache.device_type);
    out.write(reinterpret_cast<const char*>(&vits_ver), sizeof(vits_ver));
    out.write(reinterpret_cast<const char*>(&ge_dim), sizeof(ge_dim));
    out.write(reinterpret_cast<const char*>(&device_type), sizeof(device_type));

    write_string(out, cache.model_profile_id);
    write_string(out, cache.model_version);
    uint8_t requires_sv_emb = cache.requires_sv_emb ? 1 : 0;
    int32_t sv_emb_dim = static_cast<int32_t>(cache.sv_emb_dim);
    int32_t ref_enc_channels = static_cast<int32_t>(cache.ref_enc_channels);
    int32_t output_sampling_rate = static_cast<int32_t>(cache.output_sampling_rate);
    out.write(reinterpret_cast<const char*>(&requires_sv_emb), sizeof(requires_sv_emb));
    out.write(reinterpret_cast<const char*>(&sv_emb_dim), sizeof(sv_emb_dim));
    out.write(reinterpret_cast<const char*>(&ref_enc_channels), sizeof(ref_enc_channels));
    out.write(reinterpret_cast<const char*>(&output_sampling_rate), sizeof(output_sampling_rate));

    uint32_t text_len = static_cast<uint32_t>(cache.prompt_text.size());
    out.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
    out.write(cache.prompt_text.data(), text_len);

    uint32_t lang_len = static_cast<uint32_t>(cache.prompt_lang.size());
    out.write(reinterpret_cast<const char*>(&lang_len), sizeof(lang_len));
    out.write(cache.prompt_lang.data(), lang_len);

    uint32_t phone_count = static_cast<uint32_t>(cache.prompt_phones.size());
    out.write(reinterpret_cast<const char*>(&phone_count), sizeof(phone_count));
    for (const auto& phone : cache.prompt_phones) {
        uint32_t len = static_cast<uint32_t>(phone.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(phone.data(), len);
    }

    uint32_t word2ph_count = static_cast<uint32_t>(cache.prompt_word2ph.size());
    out.write(reinterpret_cast<const char*>(&word2ph_count), sizeof(word2ph_count));
    out.write(reinterpret_cast<const char*>(cache.prompt_word2ph.data()), word2ph_count * sizeof(int));

    uint32_t hubert_codes_count = static_cast<uint32_t>(cache.hubert_codes.size());
    out.write(reinterpret_cast<const char*>(&hubert_codes_count), sizeof(hubert_codes_count));
    out.write(reinterpret_cast<const char*>(cache.hubert_codes.data()), hubert_codes_count * sizeof(int32_t));

    uint32_t bert_features_count = static_cast<uint32_t>(cache.bert_features.size());
    out.write(reinterpret_cast<const char*>(&bert_features_count), sizeof(bert_features_count));
    out.write(reinterpret_cast<const char*>(cache.bert_features.data()), bert_features_count * sizeof(float));

    uint32_t speaker_embedding_count = static_cast<uint32_t>(cache.speaker_embedding.size());
    out.write(reinterpret_cast<const char*>(&speaker_embedding_count), sizeof(speaker_embedding_count));
    out.write(reinterpret_cast<const char*>(cache.speaker_embedding.data()), speaker_embedding_count * sizeof(float));

    write_float_vector(out, cache.sv_emb);
    write_float_vector(out, cache.prompt_mel);
    write_float_vector(out, cache.prompt_fea_ref);

    return out.good();
}

bool deserialize_features(const std::filesystem::path& filepath, PromptCache& cache) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for reading: " << filepath << std::endl;
        return false;
    }

    auto read_pod = [&](auto& value) {
        return static_cast<bool>(in.read(reinterpret_cast<char*>(&value), sizeof(value)));
    };
    auto read_vector = [&](auto& values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        uint32_t count = 0;
        if (!read_pod(count)) return false;
        const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(Value);
        const std::streampos position = in.tellg();
        in.seekg(0, std::ios::end);
        const std::streampos end = in.tellg();
        in.seekg(position);
        if (position < 0 || end < position || bytes > MAX_CACHE_FIELD_BYTES ||
            static_cast<uint64_t>(end - position) < bytes) {
            return false;
        }
        values.resize(count);
        return bytes == 0 || static_cast<bool>(in.read(reinterpret_cast<char*>(values.data()), bytes));
    };

    uint32_t magic = 0;
    if (!read_pod(magic) || magic != FEATURE_CACHE_MAGIC) {
        std::cerr << "[VoiceManager] Stale or invalid features cache: " << filepath << std::endl;
        return false;
    }

    int32_t vits_version = 0;
    int32_t ge_dim = 0;
    int32_t device_type = 0;
    uint8_t requires_sv_emb = 0;
    int32_t sv_emb_dim = 0;
    int32_t ref_enc_channels = 0;
    int32_t output_sampling_rate = 0;
    if (!read_pod(vits_version) || !read_pod(ge_dim) || !read_pod(device_type) ||
        !read_string(in, cache.model_profile_id) || !read_string(in, cache.model_version) ||
        !read_pod(requires_sv_emb) || !read_pod(sv_emb_dim) ||
        !read_pod(ref_enc_channels) || !read_pod(output_sampling_rate)) {
        return false;
    }
    if (requires_sv_emb > 1 || ge_dim < 0 || sv_emb_dim < 0 ||
        ref_enc_channels < 0 || output_sampling_rate < 0) {
        return false;
    }
    cache.vits_version = vits_version;
    cache.ge_dim = ge_dim;
    cache.device_type = device_type;
    cache.requires_sv_emb = requires_sv_emb != 0;
    cache.sv_emb_dim = sv_emb_dim;
    cache.ref_enc_channels = ref_enc_channels;
    cache.output_sampling_rate = output_sampling_rate;

    if (!read_string(in, cache.prompt_text) || !read_string(in, cache.prompt_lang)) return false;

    uint32_t phone_count = 0;
    if (!read_pod(phone_count) || phone_count > 1000000) return false;
    cache.prompt_phones.resize(phone_count);
    for (uint32_t i = 0; i < phone_count; ++i) {
        if (!read_string(in, cache.prompt_phones[i])) return false;
    }
    if (!read_vector(cache.prompt_word2ph) || !read_vector(cache.hubert_codes) ||
        !read_vector(cache.bert_features) || !read_vector(cache.speaker_embedding) ||
        !read_float_vector(in, cache.sv_emb) || !read_float_vector(in, cache.prompt_mel) ||
        !read_float_vector(in, cache.prompt_fea_ref)) {
        return false;
    }
    return true;
}

} // namespace gpt_sovits

using namespace gpt_sovits;

extern "C" {

gpt_sovits_voice_manager_t gpt_sovits_voice_manager_init(gpt_sovits_engine_t engine) {
    if (!engine) return nullptr;
    return new VoiceManagerImpl(engine);
}

void gpt_sovits_voice_manager_free(gpt_sovits_voice_manager_t manager) {
    if (manager) {
        delete (VoiceManagerImpl*)manager;
    }
}

bool gpt_sovits_voice_manager_register_character(
    gpt_sovits_voice_manager_t manager,
    const char* voices_root_dir,
    const char* character_id
) {
    if (!manager || !voices_root_dir || !character_id) return false;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;
    Impl* impl = (Impl*)mgr->engine;
    if (!impl) return false;

    std::string cid(character_id);
    std::filesystem::path root_dir(voices_root_dir);
    std::filesystem::path config_file = root_dir / "config.json";
    std::filesystem::path audios_dir = root_dir / "audios";
    std::filesystem::path features_dir = root_dir / "features";

    if (!std::filesystem::exists(config_file)) {
        std::cerr << "[VoiceManager] Config file not found: " << config_file.string() << std::endl;
        return false;
    }

    std::ifstream config_in(config_file.string());
    if (!config_in.is_open()) {
        std::cerr << "[VoiceManager] Failed to open config file: " << config_file.string() << std::endl;
        return false;
    }

    std::string config_content((std::istreambuf_iterator<char>(config_in)), std::istreambuf_iterator<char>());
    std::string char_name;
    std::string char_lang;
    std::unordered_map<std::string, EmotionEntry> emotions;
    if (!voice_manager_parse_emotions_config(config_content, char_name, char_lang, emotions)) {
        std::cerr << "[VoiceManager] Failed to parse config.json (expecting 'name', 'lang', 'emotions' keys)" << std::endl;
        return false;
    }

    if (g_log_enabled) std::cout << "[VoiceManager] Registering character '" << cid << "' (config name='" << char_name << "') with " << emotions.size() << " emotions" << std::endl;
    bool any_ok = false;
    std::string default_emotion;

    for (const auto& [emo_name, entry] : emotions) {
        std::string cache_id = "voice_" + char_name + "_" + emo_name;
        std::string emo_key = cid + "/" + emo_name;
        mgr->emo_to_cache_id[emo_key] = cache_id;

        if (default_emotion.empty()) {
            default_emotion = emo_name;
            mgr->char_default_emotion[cid] = emo_name;
        }
        if (!impl->vits && !impl->load_model(3)) {
            std::cerr << "[VoiceManager] Failed to load VITS before resolving feature cache namespace."
                      << std::endl;
            continue;
        }
        const int expected_device_type = get_backend_device_type(impl->vits_target_backend);
        // Prompt features depend on the exact VITS profile and backend. Keep
        // each namespace separate so switching versions never overwrites a
        // valid cache belonging to another composition.
        std::filesystem::path features_file = versioned_features_path(
            features_dir, impl->vits->profile, expected_device_type, emo_name);
        const std::filesystem::path legacy_features_file =
            features_dir / std::filesystem::u8path(emo_name + ".features.bin");
        const bool migrating_legacy = !std::filesystem::exists(features_file) &&
            std::filesystem::exists(legacy_features_file);
        const std::filesystem::path cache_to_load = migrating_legacy
            ? legacy_features_file : features_file;
        if (std::filesystem::exists(cache_to_load)) {
            if (g_log_enabled) std::cout << "[VoiceManager] Loading cached features for emotion '" << emo_name << "'..." << std::endl;
            PromptCache cache;
            if (deserialize_features(cache_to_load, cache)) {
                std::string mismatch_reason;
                if (cache_matches_current_profile(cache, impl, expected_device_type, mismatch_reason)) {
                    if (migrating_legacy) {
                        std::filesystem::create_directories(features_file.parent_path());
                        serialize_features(features_file, cache);
                    }
                    impl->prompt_caches[cache_id] = cache;
                    if (g_log_enabled) std::cout << "[VoiceManager]   OK (from cache)" << std::endl;
                    any_ok = true;
                    continue;
                } else {
                    std::cerr << "[VoiceManager]   Cached features (VITS v" << cache.vits_version << ", dim " << cache.ge_dim
                              << ", device " << cache.device_type << ", profile '" << cache.model_profile_id
                              << "') do not match current model profile '" << (impl->vits ? impl->vits->profile.profile_id : "")
                              << "' (device " << expected_device_type << "): " << mismatch_reason
                              << ". Discarding cache..." << std::endl;
                }
            } else {
                std::cerr << "[VoiceManager]   Cache corrupt, re-extracting..." << std::endl;
            }
        }
        // Load audio
        std::filesystem::path audio_path = audios_dir / std::filesystem::u8path(entry.audio);
        if (!std::filesystem::exists(audio_path)) {
            std::cerr << "[VoiceManager] Audio not found for emotion '" << emo_name << "': " << audio_path.string() << std::endl;
            continue;
        }
        int sample_rate = 0;
        std::vector<float> audio_data = voice_manager_load_wav_file(audio_path.u8string(), sample_rate);
        if (audio_data.empty()) {
            std::cerr << "[VoiceManager] Failed to load audio for emotion '" << emo_name << "'" << std::endl;
            continue;
        }

        // Check for pre-extracted speaker vector (ERes2NetV2 sv_emb) next to reference wav
        std::filesystem::path sv_path = audio_path;
        sv_path.replace_extension(".sv.bin");
        std::vector<float> sv_emb_data;
        if (impl->vits && impl->vits->profile.requires_sv_emb && std::filesystem::exists(sv_path)) {
            if (g_log_enabled) std::cout << "[VoiceManager] Found speaker vector file: " << sv_path.string() << std::endl;
            std::ifstream sv_in(sv_path, std::ios::binary);
            if (sv_in.is_open()) {
                const size_t expected_dim = static_cast<size_t>(impl->vits->profile.sv_emb_dim);
                const size_t expected_bytes = expected_dim * sizeof(float);
                const size_t actual_bytes = static_cast<size_t>(std::filesystem::file_size(sv_path));
                if (expected_dim > 0 && actual_bytes == expected_bytes) {
                    sv_emb_data.resize(expected_dim);
                    sv_in.read(reinterpret_cast<char*>(sv_emb_data.data()), expected_bytes);
                } else {
                    std::cerr << "[VoiceManager] Warning: Speaker vector size mismatch for "
                              << sv_path.string() << ": expected " << expected_bytes
                              << " bytes, got " << actual_bytes << std::endl;
                }
            } else {
                std::cerr << "[VoiceManager] Warning: Failed to open speaker vector file: " << sv_path.string() << std::endl;
            }
        }

        if (g_log_enabled) std::cout << "[VoiceManager] Extracting features for emotion '" << emo_name << "'..." << std::endl;
        gpt_sovits_get_or_create_prompt_cache(
            mgr->engine,
            cache_id.c_str(),
            audio_data.data(),
            audio_data.size(),
            sample_rate,
            entry.text.c_str(),
            char_lang.c_str(),
            sv_emb_data.empty() ? nullptr : sv_emb_data.data(),
            sv_emb_data.size()
        );

        auto it = impl->prompt_caches.find(cache_id);
        if (it != impl->prompt_caches.end()) {
            it->second.device_type = get_backend_device_type(impl->vits_target_backend);
            std::filesystem::create_directories(features_file.parent_path());
            if (serialize_features(features_file, it->second)) {
                if (g_log_enabled) std::cout << "[VoiceManager]   Cached -> " << features_file.string() << std::endl;
                any_ok = true;
            } else {
                std::cerr << "[VoiceManager]   Failed to save cache for '" << emo_name << "'" << std::endl;
            }
        } else {
            std::cerr << "[VoiceManager]   Feature extraction failed for '" << emo_name << "'" << std::endl;
        }
    }
    return any_ok;
}

const char* gpt_sovits_voice_manager_get_cache_id(
    gpt_sovits_voice_manager_t manager,
    const char* character_id
) {
    if (!manager || !character_id) return nullptr;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;

    std::string cid(character_id);
    auto it = mgr->emo_to_cache_id.find(cid);
    if (it != mgr->emo_to_cache_id.end()) {
        return it->second.c_str();
    }
    auto def_it = mgr->char_default_emotion.find(cid);
    if (def_it != mgr->char_default_emotion.end()) {
        std::string emo_key = cid + "/" + def_it->second;
        auto emo_it = mgr->emo_to_cache_id.find(emo_key);
        if (emo_it != mgr->emo_to_cache_id.end()) {
            return emo_it->second.c_str();
        }
    }
    return nullptr;
}

const float* gpt_sovits_voice_manager_synthesize(
    gpt_sovits_voice_manager_t manager,
    const char* character_id,
    const char* text,
    const char* language,
    float speed,
    int* out_num_samples
) {
    if (!manager || !character_id || !text || !language) return nullptr;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;
    const char* cache_id = gpt_sovits_voice_manager_get_cache_id(manager, character_id);
    if (!cache_id) {
        std::cerr << "[VoiceManager] Character not registered: " << character_id << std::endl;
        return nullptr;
    }
    return gpt_sovits_synthesize_with_cache(mgr->engine, text, language, cache_id, speed, out_num_samples);
}

} // extern "C"
