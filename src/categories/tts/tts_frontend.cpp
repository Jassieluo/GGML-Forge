#include "tts_frontend.h"
#include <iostream>

namespace tts {

Frontend::Frontend(const std::string& dict_path) {
    std::cout << "[TTS Frontend] Loading phonemizer dictionary from: " << dict_path << std::endl;
    phonemizer = std::make_unique<phonemizer::Phonemizer>(dict_path);
}

Frontend::~Frontend() = default;

bool Frontend::text_to_ids(
    const std::string& text,
    const std::string& lang,
    const std::unordered_map<std::string, int32_t>& token_to_id,
    std::vector<int32_t>& out_ids
) {
    if (!phonemizer) return false;
    
    // Process text to phoneme results
    auto res = phonemizer->process(text, lang);
    
    out_ids.clear();
    out_ids.reserve(res.phones.size());
    
    for (const auto& ph : res.phones) {
        auto it = token_to_id.find(ph);
        if (it != token_to_id.end()) {
            out_ids.push_back(it->second);
        } else {
            // Map unknown tokens to UNK id (e.g. 0) or print warning
            std::cerr << "[TTS Frontend] Warning: Unknown symbol: '" << ph << "'" << std::endl;
            out_ids.push_back(0); 
        }
    }
    
    return true;
}

} // namespace tts
