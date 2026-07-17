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
    std::vector<float> sv_emb;
    std::vector<float> prompt_fea_ref;
    
    int vits_version = 0;
    int ge_dim = 0;
    int device_type = -1;
    std::string model_profile_id;
    std::string model_version;
    bool requires_sv_emb = false;
    int sv_emb_dim = 0;
    int ref_enc_channels = 0;
    int output_sampling_rate = 0;
};

} // namespace gpt_sovits
