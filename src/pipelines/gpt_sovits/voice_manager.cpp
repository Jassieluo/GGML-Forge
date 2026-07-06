#include "voice_manager.h"
#include "gpt_sovits_internal.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <filesystem>

namespace gpt_sovits {

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
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[VoiceManager WAV Loader] Failed to open WAV file: " << filename << "\n";
        return {};
    }
    char chunk_id[4];
    file.read(chunk_id, 4);
    if (std::strncmp(chunk_id, "RIFF", 4) != 0) {
        std::cerr << "[VoiceManager WAV Loader] Invalid RIFF header\n";
        return {};
    }

    file.seekg(8, std::ios::beg);
    char format_id[4];
    file.read(format_id, 4);
    if (std::strncmp(format_id, "WAVE", 4) != 0) {
        std::cerr << "[VoiceManager WAV Loader] Not a WAVE file\n";
        return {};
    }
    short num_channels = 0;
    int s_rate = 0;
    short bits_per_sample = 0;
    int data_size = 0;

    while (file) {
        char subchunk_id[4];
        file.read(subchunk_id, 4);
        if (!file) break;
        int subchunk_size = 0;
        file.read(reinterpret_cast<char*>(&subchunk_size), 4);
        if (!file) break;

        if (std::strncmp(subchunk_id, "fmt ", 4) == 0) {
            short audio_format = 0;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&s_rate), 4);
            file.seekg(6, std::ios::cur);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (subchunk_size > 16) {
                file.seekg(subchunk_size - 16, std::ios::cur);
            }
        } else if (std::strncmp(subchunk_id, "data", 4) == 0) {
            data_size = subchunk_size;
            std::vector<float> audio_data;
            if (bits_per_sample == 16) {
                int num_samples = data_size / 2;
                std::vector<short> raw_samples(num_samples);
                file.read(reinterpret_cast<char*>(raw_samples.data()), data_size);
                audio_data.resize(num_samples);
                for (int i = 0; i < num_samples; ++i) {
                    audio_data[i] = raw_samples[i] / 32768.0f;
                }
            } else if (bits_per_sample == 32) {
                int num_samples = data_size / 4;
                audio_data.resize(num_samples);
                file.read(reinterpret_cast<char*>(audio_data.data()), data_size);
            } else {
                std::cerr << "[VoiceManager WAV Loader] Unsupported bits per sample: " << bits_per_sample << "\n";
                return {};
            }

            if (s_rate == 32000) {
                if (g_log_enabled) std::cout << "[VoiceManager WAV Loader] Downsampling 32000 Hz reference to 16000 Hz...\n";
                std::vector<float> downsampled;
                downsampled.reserve(audio_data.size() / 2);
                for (size_t i = 0; i < audio_data.size(); i += 2) {
                    downsampled.push_back(audio_data[i]);
                }
                audio_data = std::move(downsampled);
                s_rate = 16000;
            } else if (s_rate == 48000) {
                if (g_log_enabled) std::cout << "[VoiceManager WAV Loader] Downsampling 48000 Hz reference to 16000 Hz...\n";
                std::vector<float> downsampled;
                downsampled.reserve(audio_data.size() / 3);
                for (size_t i = 0; i < audio_data.size(); i += 3) {
                    downsampled.push_back(audio_data[i]);
                }
                audio_data = std::move(downsampled);
                s_rate = 16000;
            }
            sample_rate = s_rate;
            if (g_log_enabled) std::cout << "[VoiceManager WAV Loader] Loaded " << filename << " | Samples: " << audio_data.size() << "\n";
            return audio_data;
        } else {
            file.seekg(subchunk_size, std::ios::cur);
        }
    }
    return {};
}

bool serialize_features(const std::string& filepath, const PromptCache& cache) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    uint32_t magic = 0x47535646;
    uint32_t version = 3;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write compatibility headers (Format version >= 2)
    int32_t vits_ver = static_cast<int32_t>(cache.vits_version);
    int32_t ge_dim = static_cast<int32_t>(cache.ge_dim);
    int32_t device_type = static_cast<int32_t>(cache.device_type);
    out.write(reinterpret_cast<const char*>(&vits_ver), sizeof(vits_ver));
    out.write(reinterpret_cast<const char*>(&ge_dim), sizeof(ge_dim));
    out.write(reinterpret_cast<const char*>(&device_type), sizeof(device_type));

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

    return out.good();
}

bool deserialize_features(const std::string& filepath, PromptCache& cache) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for reading: " << filepath << std::endl;
        return false;
    }
    uint32_t magic = 0;
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != 0x47535646) {
        std::cerr << "[VoiceManager] Invalid magic in features file: " << filepath << std::endl;
        return false;
    }

    if (version >= 2) {
        int32_t vits_ver = 0;
        int32_t ge_dim = 0;
        in.read(reinterpret_cast<char*>(&vits_ver), sizeof(vits_ver));
        in.read(reinterpret_cast<char*>(&ge_dim), sizeof(ge_dim));
        cache.vits_version = vits_ver;
        cache.ge_dim = ge_dim;
        if (version >= 3) {
            int32_t device_type = 0;
            in.read(reinterpret_cast<char*>(&device_type), sizeof(device_type));
            cache.device_type = device_type;
        } else {
            cache.device_type = -1;
        }
    } else if (version == 1) {
        // Fallback for legacy format version 1
        cache.vits_version = 0;
        cache.ge_dim = 0;
        cache.device_type = -1;
    } else {
        std::cerr << "[VoiceManager] Unsupported features file format version: " << version << std::endl;
        return false;
    }

    uint32_t text_len = 0;
    in.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
    cache.prompt_text.resize(text_len);
    if (text_len > 0) {
        in.read(&cache.prompt_text[0], text_len);
    }
    uint32_t lang_len = 0;
    in.read(reinterpret_cast<char*>(&lang_len), sizeof(lang_len));
    cache.prompt_lang.resize(lang_len);
    if (lang_len > 0) {
        in.read(&cache.prompt_lang[0], lang_len);
    }
    uint32_t phone_count = 0;
    in.read(reinterpret_cast<char*>(&phone_count), sizeof(phone_count));
    cache.prompt_phones.resize(phone_count);
    for (uint32_t i = 0; i < phone_count; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        cache.prompt_phones[i].resize(len);
        if (len > 0) {
            in.read(&cache.prompt_phones[i][0], len);
        }
    }
    uint32_t word2ph_count = 0;
    in.read(reinterpret_cast<char*>(&word2ph_count), sizeof(word2ph_count));
    cache.prompt_word2ph.resize(word2ph_count);
    if (word2ph_count > 0) {
        in.read(reinterpret_cast<char*>(cache.prompt_word2ph.data()), word2ph_count * sizeof(int));
    }
    uint32_t hubert_codes_count = 0;
    in.read(reinterpret_cast<char*>(&hubert_codes_count), sizeof(hubert_codes_count));
    cache.hubert_codes.resize(hubert_codes_count);
    if (hubert_codes_count > 0) {
        in.read(reinterpret_cast<char*>(cache.hubert_codes.data()), hubert_codes_count * sizeof(int32_t));
    }
    uint32_t bert_features_count = 0;
    in.read(reinterpret_cast<char*>(&bert_features_count), sizeof(bert_features_count));
    cache.bert_features.resize(bert_features_count);
    if (bert_features_count > 0) {
        in.read(reinterpret_cast<char*>(cache.bert_features.data()), bert_features_count * sizeof(float));
    }
    uint32_t speaker_embedding_count = 0;
    in.read(reinterpret_cast<char*>(&speaker_embedding_count), sizeof(speaker_embedding_count));
    cache.speaker_embedding.resize(speaker_embedding_count);
    if (speaker_embedding_count > 0) {
        in.read(reinterpret_cast<char*>(cache.speaker_embedding.data()), speaker_embedding_count * sizeof(float));
    }

    return in.good();
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
        // Check for precomputed features
        std::filesystem::path features_file = features_dir / std::filesystem::u8path(emo_name + ".features.bin");
        if (std::filesystem::exists(features_file)) {
            if (g_log_enabled) std::cout << "[VoiceManager] Loading cached features for emotion '" << emo_name << "'..." << std::endl;
            PromptCache cache;
            if (!impl->vits) {
                impl->load_model(3);
            }
            int expected_vits_version = impl->vits ? impl->vits->version : 2;
            int expected_ge_dim = 512;
            if (impl->vits) {
                expected_ge_dim = impl->vits->get_tensor("prelu.weight") ? 1024 : 512;
            }
            if (deserialize_features(features_file.string(), cache)) {
                int expected_device_type = get_backend_device_type(impl->vits_target_backend);
                if (cache.vits_version == expected_vits_version &&
                    cache.ge_dim == expected_ge_dim &&
                    cache.speaker_embedding.size() == (size_t)expected_ge_dim &&
                    (cache.device_type == -1 || cache.device_type == expected_device_type)) {
                    impl->prompt_caches[cache_id] = cache;
                    if (g_log_enabled) std::cout << "[VoiceManager]   OK (from cache)" << std::endl;
                    any_ok = true;
                    continue;
                } else {
                    std::cerr << "[VoiceManager]   Cached features (VITS v" << cache.vits_version << ", dim " << cache.ge_dim
                              << ", device " << cache.device_type << ") do not match expected (VITS v" << expected_vits_version
                              << ", dim " << expected_ge_dim << ", device " << expected_device_type
                              << ") for current model/backend! Discarding cache..." << std::endl;
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
        std::vector<float> audio_data = voice_manager_load_wav_file(audio_path.string(), sample_rate);
        if (audio_data.empty()) {
            std::cerr << "[VoiceManager] Failed to load audio for emotion '" << emo_name << "'" << std::endl;
            continue;
        }

        if (g_log_enabled) std::cout << "[VoiceManager] Extracting features for emotion '" << emo_name << "'..." << std::endl;
        gpt_sovits_get_or_create_prompt_cache(
            mgr->engine,
            cache_id.c_str(),
            audio_data.data(),
            audio_data.size(),
            entry.text.c_str(),
            char_lang.c_str()
        );

        auto it = impl->prompt_caches.find(cache_id);
        if (it != impl->prompt_caches.end()) {
            it->second.device_type = get_backend_device_type(impl->vits_target_backend);
            std::filesystem::create_directories(features_dir);
            if (serialize_features(features_file.string(), it->second)) {
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
