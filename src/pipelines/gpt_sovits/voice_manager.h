#pragma once

#include "gpt_sovits.h"
#include "pipeline_types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace gpt_sovits {

struct EmotionEntry {
    std::string audio;
    std::string text;
};

struct VoiceManagerImpl {
    gpt_sovits_engine_t engine;
    // Map: "doubao/安慰" -> "voice_doubao_安慰" (cache_id)
    std::unordered_map<std::string, std::string> emo_to_cache_id;
    // Map: "doubao" -> default emotion name
    std::unordered_map<std::string, std::string> char_default_emotion;

    VoiceManagerImpl(gpt_sovits_engine_t eng);
};

// Config parse helper
bool voice_manager_parse_emotions_config(
    const std::string& json_str,
    std::string& out_name,
    std::string& out_lang,
    std::unordered_map<std::string, EmotionEntry>& out_emotions
);

// WAV loader helper
std::vector<float> voice_manager_load_wav_file(const std::string& filename, int& sample_rate);

// Prompt Cache Serialization helpers
bool serialize_features(const std::string& filepath, const PromptCache& cache);
bool deserialize_features(const std::string& filepath, PromptCache& cache);

} // namespace gpt_sovits
