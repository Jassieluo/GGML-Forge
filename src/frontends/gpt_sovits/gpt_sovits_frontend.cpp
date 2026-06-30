#include "gpt_sovits_frontend.h"
#include "text_utils.h"
#include "phonemizer.h"
#include "models/text/bert/bert.h"
#include "include/symbols.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace gpt_sovits {

// Helper: safe read tensor from backend
static void safe_ggml_backend_tensor_get(const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (!tensor) return;
    if (tensor->buffer == nullptr) {
        std::memcpy(data, (const char*)tensor->data + offset, size);
    } else {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
}

struct LangSegment {
    std::string text;
    std::string lang;
};

static std::vector<LangSegment> split_zh_en(const std::u32string& text) {
    if (text.empty()) return {};
    
    std::vector<LangSegment> segments;
    std::u32string current_text;
    std::string current_lang = "";
    
    auto is_cjk = [](char32_t cp) {
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
        if (cp >= 0x3400 && cp <= 0x4DBF) return true;
        if (cp == U'，' || cp == U'。' || cp == U'？' || cp == U'！' || cp == U'、' || 
            cp == U'；' || cp == U'：' || cp == U'“' || cp == U'”' || cp == U'…') {
            return true;
        }
        return false;
    };
    
    auto is_english_char = [](char32_t cp) {
        if (cp >= U'A' && cp <= U'Z') return true;
        if (cp >= U'a' && cp <= U'z') return true;
        return false;
    };
    
    for (char32_t cp : text) {
        std::string char_lang = "";
        if (is_cjk(cp)) {
            char_lang = "zh";
        } else if (is_english_char(cp)) {
            char_lang = "en";
        } else {
            char_lang = current_lang.empty() ? "zh" : current_lang;
        }
        
        if (current_lang.empty()) {
            current_lang = char_lang;
            current_text.push_back(cp);
        } else if (current_lang == char_lang) {
            current_text.push_back(cp);
        } else {
            segments.push_back({phonemizer::utf32_to_utf8(current_text), current_lang});
            current_text = {cp};
            current_lang = char_lang;
        }
    }
    if (!current_text.empty()) {
        segments.push_back({phonemizer::utf32_to_utf8(current_text), current_lang});
    }
    
    std::vector<LangSegment> merged;
    for (const auto& seg : segments) {
        if (merged.empty()) {
            merged.push_back(seg);
        } else if (merged.back().lang == seg.lang) {
            merged.back().text += seg.text;
        } else {
            merged.push_back(seg);
        }
    }
    return merged;
}

GPTSoVITSFrontend::GPTSoVITSFrontend(const std::string& dict_dir)
    : dict_dir_(dict_dir) {}

GPTSoVITSFrontend::~GPTSoVITSFrontend() = default;

bool GPTSoVITSFrontend::initialize() {
    phonemizer_ = std::make_unique<phonemizer::Phonemizer>(dict_dir_);
    
    const auto& syms = get_phone_symbols();
    for (size_t i = 0; i < syms.size(); ++i) {
        phone_to_id_map_[syms[i]] = static_cast<int32_t>(i);
    }

    std::string vocab_path = dict_dir_ + "/bert_vocab.txt";
    load_bert_vocab(vocab_path);

    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "[Frontend Debug] phone_to_id_map_ size: " << phone_to_id_map_.size() << "\n";
        std::cout << "[Frontend Debug] phone_to_id_map_['f'] = " << phone_to_id_map_["f"] << "\n";
        std::cout << "[Frontend Debug] phone_to_id_map_['an2'] = " << phone_to_id_map_["an2"] << "\n";
        std::fflush(stdout);
    }
    
    return true;
}

int32_t GPTSoVITSFrontend::phone_to_id(const std::string& phone) const {
    auto it = phone_to_id_map_.find(phone);
    if (it != phone_to_id_map_.end()) {
        return it->second;
    }
    // Return UNK symbol index if not found (default is 93 or 0)
    auto unk_it = phone_to_id_map_.find("UNK");
    if (unk_it != phone_to_id_map_.end()) {
        return unk_it->second;
    }
    return 0;
}

bool GPTSoVITSFrontend::load_bert_vocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "[GPTSoVITSFrontend] Failed to open BERT vocabulary file: " << vocab_path << "\n";
        return false;
    }
    
    bert_vocab_.clear();
    std::string line;
    int32_t idx = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        bert_vocab_[line] = idx++;
    }
    
    return true;
}

std::vector<int32_t> GPTSoVITSFrontend::bert_tokenize(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.push_back(101); // [CLS]
    
    std::u32string u32_chars = phonemizer::utf8_to_utf32(text);
    for (char32_t c : u32_chars) {
        std::string utf8_char = phonemizer::utf32_char_to_utf8(c);
        auto it = bert_vocab_.find(utf8_char);
        if (it != bert_vocab_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(100); // [UNK]
        }
    }
    
    ids.push_back(102); // [SEP]
    return ids;
}

std::vector<std::string> GPTSoVITSFrontend::split_text(const std::string& text, const std::string& split_method) const {
    std::u32string u32_text = phonemizer::utf8_to_utf32(text);
    u32_text = text::clean_formatted_decimals(u32_text);
    
    std::vector<std::u32string> u32_segments;
    if (split_method == "cut0") {
        u32_segments = text::cut0(u32_text);
    } else if (split_method == "cut1") {
        u32_segments = text::cut1(u32_text);
    } else if (split_method == "cut2") {
        u32_segments = text::cut2(u32_text);
    } else if (split_method == "cut3") {
        u32_segments = text::cut3(u32_text);
    } else if (split_method == "cut4") {
        u32_segments = text::cut4(u32_text);
    } else if (split_method == "cut5" || split_method == "5") {
        u32_segments = text::cut5(u32_text);
    } else {
        u32_segments = text::cut0(u32_text);
    }
    
    std::vector<std::u32string> final_u32_segments;
    for (const auto& seg : u32_segments) {
        std::vector<std::u32string> sub_segs = text::split_by_sentence_ends(seg);
        for (const auto& sub : sub_segs) {
            if (!text::is_subset_of_punctuation(sub)) {
                final_u32_segments.push_back(sub);
            }
        }
    }
    
    std::vector<std::string> results;
    for (const auto& seg : final_u32_segments) {
        results.push_back(phonemizer::utf32_to_utf8(seg));
    }
    return results;
}

bool GPTSoVITSFrontend::process(
    const std::string& text,
    const std::string& language,
    BertModel* bert_model,
    struct ggml_context* ctx_graph,
    ggml_backend_t bert_backend,
    FrontendResult& out_result
) {
    out_result.phones.clear();
    out_result.phone_ids.clear();
    out_result.bert_features.clear();
    
    if (!phonemizer_) {
        std::cerr << "[GPTSoVITSFrontend] Error: Phonemizer is not initialized.\n";
        return false;
    }
    
    if (language != "zh" && language != "zh_en") {
        auto res = phonemizer_->process(text, language);
        out_result.phones = res.phones;
        out_result.word2ph = res.word2ph;
        out_result.phone_ids.reserve(res.phones.size());
        for (const auto& ph : res.phones) {
            out_result.phone_ids.push_back(phone_to_id(ph));
        }
        out_result.bert_features.assign(1024 * res.phones.size(), 0.0f);
        return true;
    }
    
    // Mixed Mode segmentation
    std::u32string u32_text = phonemizer::utf8_to_utf32(text);
    std::vector<LangSegment> segments = split_zh_en(u32_text);
    
    std::vector<std::string> combined_phones;
    std::vector<int> combined_word2ph;
    std::vector<float> combined_bert_aligned;
    
    for (const auto& seg : segments) {
        phonemizer::PhonemizerResult seg_res = phonemizer_->process(seg.text, seg.lang);
        
        combined_phones.insert(combined_phones.end(), seg_res.phones.begin(), seg_res.phones.end());
        combined_word2ph.insert(combined_word2ph.end(), seg_res.word2ph.begin(), seg_res.word2ph.end());
        
        int seg_phone_len = (int)seg_res.phones.size();
        
        if (seg.lang == "zh") {
            if (!bert_model) {
                std::cerr << "[GPTSoVITSFrontend] Warning: BERT model is null, filling zeroes for BERT features.\n";
                size_t start_idx = combined_bert_aligned.size();
                combined_bert_aligned.resize(start_idx + 1024 * seg_phone_len, 0.0f);
                continue;
            }
            
            std::vector<int32_t> target_bert_ids = bert_tokenize(seg_res.norm_text);
            struct ggml_tensor* seg_bert_out = bert_model->forward(ctx_graph, target_bert_ids, bert_backend);
            
            int seg_bert_len = seg_bert_out ? seg_bert_out->ne[1] : 0;
            std::vector<float> seg_bert_data(1024 * seg_bert_len, 0.0f);
            if (seg_bert_out) {
                safe_ggml_backend_tensor_get(seg_bert_out, seg_bert_data.data(), 0, 1024 * seg_bert_len * sizeof(float));
            }
            
            std::vector<int> seg_phone_to_word;
            for (size_t w_idx = 0; w_idx < seg_res.word2ph.size(); ++w_idx) {
                int num_phones = seg_res.word2ph[w_idx];
                for (int p = 0; p < num_phones; ++p) {
                    seg_phone_to_word.push_back((int)w_idx);
                }
            }
            
            for (int p_idx = 0; p_idx < seg_phone_len; ++p_idx) {
                int w_idx = 0;
                if (p_idx < (int)seg_phone_to_word.size()) {
                    w_idx = seg_phone_to_word[p_idx] + 1; // +1 to skip [CLS]
                }
                if (w_idx >= seg_bert_len) w_idx = seg_bert_len - 1;
                if (w_idx < 0) w_idx = 0;
                
                size_t start_idx = combined_bert_aligned.size();
                combined_bert_aligned.resize(start_idx + 1024);
                std::memcpy(combined_bert_aligned.data() + start_idx, seg_bert_data.data() + w_idx * 1024, 1024 * sizeof(float));
            }
        } else {
            size_t start_idx = combined_bert_aligned.size();
            combined_bert_aligned.resize(start_idx + 1024 * seg_phone_len, 0.0f);
        }
    }
    
    out_result.phones = combined_phones;
    out_result.word2ph = combined_word2ph;
    out_result.phone_ids.reserve(combined_phones.size());
    for (const auto& ph : combined_phones) {
        out_result.phone_ids.push_back(phone_to_id(ph));
    }
    out_result.bert_features = combined_bert_aligned;
    return true;
}

} // namespace gpt_sovits
