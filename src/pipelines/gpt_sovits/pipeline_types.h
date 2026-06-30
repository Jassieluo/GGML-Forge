#pragma once

#include <string>
#include <vector>

namespace gpt_sovits {

struct PromptCache {
    std::string prompt_text;
    std::string prompt_lang;
    std::vector<std::string> prompt_phones;
    std::vector<int> prompt_word2ph;
    std::vector<int32_t> hubert_codes;
    std::vector<float> bert_features;
    std::vector<float> speaker_embedding;
    std::vector<float> prompt_mel;
};

} // namespace gpt_sovits
