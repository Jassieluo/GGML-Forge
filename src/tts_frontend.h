#pragma once

#include "phonemizer.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace tts {

class Frontend {
public:
    Frontend(const std::string& dict_path);
    ~Frontend();

    // Convert raw text into a sequence of symbol/phoneme IDs
    bool text_to_ids(
        const std::string& text,
        const std::string& lang,
        const std::unordered_map<std::string, int32_t>& token_to_id,
        std::vector<int32_t>& out_ids
    );

private:
    std::unique_ptr<phonemizer::Phonemizer> phonemizer;
};

} // namespace tts
