#include "phonemizer.h"
#include "cppjieba/Jieba.hpp"
#include "categories/tts/symbols.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

namespace phonemizer {

Phonemizer::Phonemizer(const std::string& dict_dir) {
    // 1. Initialize cppjieba
    jieba = std::make_unique<cppjieba::Jieba>(
        dict_dir + "/jieba.dict.utf8",
        dict_dir + "/hmm_model.utf8",
        dict_dir + "/user.dict.utf8",
        dict_dir + "/idf.utf8",
        dict_dir + "/stop_words.utf8"
    );

    // 2. Initialize TextNormalizer and ToneSandhi
    normalizer = std::make_unique<TextNormalizer>();
    tone_modifier = std::make_unique<ToneSandhi>();

    // 3. Load dictionaries
    load_pinyin_dicts(dict_dir);
    load_opencpop_strict(dict_dir);
    set_symbol_version(2);
    load_english_dict(dict_dir);

    // 4. Connect ToneSandhi with pinyin maps
    tone_modifier->init(&phrase_pinyin_map, &single_pinyin_map);

    // 5. Setup Erhua word lists
    must_erhua = {U"小院儿", U"胡同儿", U"范儿", U"老汉儿", U"撒欢儿", U"寻老礼儿", U"妥妥儿", U"媳妇儿"};
    not_erhua = {
        U"虐儿", U"为儿", U"护儿", U"瞒儿", U"救儿", U"替儿", U"有儿", U"一儿", U"我儿", U"俺儿",
        U"妻儿", U"拐儿", U"聋儿", U"乞儿", U"患儿", U"幼儿", U"孤儿", U"婴儿", U"婴幼儿", U"连体儿",
        U"脑瘫儿", U"流浪儿", U"体弱儿", U"混血儿", U"蜜雪儿", U"舫儿", U"祖儿", U"美儿", U"应采儿", U"可儿",
        U"侄儿", U"孙儿", U"侄孙儿", U"女儿", U"男儿", U"红孩儿", U"花儿", U"虫儿", U"马儿", U"鸟儿",
        U"猪儿", U"猫儿", U"狗儿", U"少儿"
    };
}

Phonemizer::~Phonemizer() = default;

void Phonemizer::load_pinyin_dicts(const std::string& dict_dir) {
    // Load single characters
    std::string single_path = dict_dir + "/pinyin_single.txt";
    std::ifstream sf(single_path);
    if (sf.is_open()) {
        std::string line;
        while (std::getline(sf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string char_utf8 = line.substr(0, tab_pos);
                std::string pinyins_str = line.substr(tab_pos + 1);
                
                std::u32string char_u32 = utf8_to_utf32(char_utf8);
                if (!char_u32.empty()) {
                     char32_t cp = char_u32[0];
                     std::vector<std::string> pinyins;
                     std::stringstream ss(pinyins_str);
                     std::string item;
                     while (std::getline(ss, item, ',')) {
                         pinyins.push_back(item);
                     }
                     single_pinyin_map[cp] = pinyins;
                }
            }
        }
        sf.close();
    } else {
        std::cerr << "Warning: Failed to load single character dict from: " << single_path << std::endl;
    }

    // Load phrases
    std::string phrase_path = dict_dir + "/pinyin_phrase.txt";
    std::ifstream pf(phrase_path);
    if (pf.is_open()) {
        std::string line;
        while (std::getline(pf, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string phrase = line.substr(0, tab_pos);
                std::string pinyins_str = line.substr(tab_pos + 1);
                
                std::vector<std::string> pinyins;
                std::stringstream ss(pinyins_str);
                std::string item;
                while (std::getline(ss, item, ' ')) {
                    pinyins.push_back(item);
                }
                phrase_pinyin_map[phrase] = pinyins;
            }
        }
        pf.close();
    } else {
        std::cerr << "Warning: Failed to load phrase dict from: " << phrase_path << std::endl;
    }
}

void Phonemizer::load_opencpop_strict(const std::string& dict_dir) {
    std::string path = dict_dir + "/opencpop-strict.txt";
    std::ifstream f(path);
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            size_t tab_pos = line.find('\t');
            if (tab_pos != std::string::npos) {
                std::string py = line.substr(0, tab_pos);
                std::string syms_str = line.substr(tab_pos + 1);
                
                size_t space_pos = syms_str.find(' ');
                if (space_pos != std::string::npos) {
                    std::string init = syms_str.substr(0, space_pos);
                    std::string fin = syms_str.substr(space_pos + 1);
                    pinyin_to_symbol_map[py] = {init, fin};
                }
            }
        }
        f.close();
    } else {
        std::cerr << "Warning: Failed to load opencpop-strict from: " << path << std::endl;
    }
}

void Phonemizer::load_symbols() {
    symbols.clear();
    const auto& syms = get_phone_symbols(version_);
    symbols.insert(syms.begin(), syms.end());
}

void Phonemizer::set_symbol_version(int version) {
    version_ = version;
    load_symbols();
}

void Phonemizer::load_english_dict(const std::string& dict_dir) {
    // Load cmudict-fast.rep format: WORD PH1 PH2 ...
    // (space separated, all uppercase, one word per line)
    auto load_rep_file = [this](const std::string& path, bool overwrite) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[Phonemizer] Warning: Failed to load English dict from: " << path << std::endl;
            return;
        }
        std::string line;
        int loaded = 0;
        while (std::getline(f, line)) {
            // Strip \r for Windows line endings
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == ';' || line[0] == '#') continue;

            // Find first space - everything before is the word
            size_t space_pos = line.find(' ');
            if (space_pos == std::string::npos) continue;

            std::string word = line.substr(0, space_pos);
            // Lowercase the key
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            // Skip variants like "WORD(2)" - they have pronunciation alternatives
            // We take only the primary (no parenthesis suffix)
            if (word.find('(') != std::string::npos) continue;

            // Only overwrite if overwrite==true (for hot dict), else skip existing
            if (!overwrite && english_dict.find(word) != english_dict.end()) continue;

            // Parse phones
            std::vector<std::string> phones;
            std::string phones_str = line.substr(space_pos + 1);
            std::istringstream pss(phones_str);
            std::string ph;
            while (pss >> ph) {
                phones.push_back(ph);
            }
            if (!phones.empty()) {
                english_dict[word] = phones;
                ++loaded;
            }
        }
        f.close();
        std::cout << "[Phonemizer] Loaded " << loaded << " entries from: " << path << std::endl;
    };

    // Load main CMUDict (fast variant, space-separated)
    load_rep_file(dict_dir + "/cmudict-fast.rep", false);
    // Load hot override dict (overwrites main dict)
    load_rep_file(dict_dir + "/engdict-hot.rep", true);

    std::cout << "[Phonemizer] English dict total entries: " << english_dict.size() << std::endl;
}

std::pair<std::vector<std::string>, std::vector<std::string>> Phonemizer::get_initials_finals(const std::string& word) const {
    std::vector<std::string> base_pinyins = tone_modifier->get_base_pinyins(word);
    std::vector<std::string> initials;
    std::vector<std::string> finals;

    for (const auto& py : base_pinyins) {
        if (py.empty()) {
            initials.push_back("");
            finals.push_back("");
            continue;
        }

        char tone = py.back();
        std::string pinyin_no_tone = py;
        if (tone >= '1' && tone <= '5') {
            pinyin_no_tone = py.substr(0, py.size() - 1);
        } else {
            tone = '5'; // fallback neutral tone
        }

        // Apply pypinyin post process replacements matching chinese2.py lines 268-293
        std::string initial = "";
        std::string final_val = pinyin_no_tone;

        if (pinyin_no_tone == "er") {
            initial = "";
            final_val = "er";
        } else {
            // Approximate initials split: check first 2 characters, then first 1
            static const std::vector<std::string> possible_initials = {
                "ch", "sh", "zh", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "w", "x", "y", "z"
            };
            for (const auto& init : possible_initials) {
                if (pinyin_no_tone.rfind(init, 0) == 0) {
                    initial = init;
                    final_val = pinyin_no_tone.substr(init.size());
                    break;
                }
            }
        }

        // chinese2.py mapping rules
        if (pinyin_no_tone != "er") {
            if (!initial.empty()) {
                static const std::unordered_map<std::string, std::string> v_rep_map = {
                    {"uei", "ui"}, {"iou", "iu"}, {"uen", "un"}
                };
                if (v_rep_map.find(final_val) != v_rep_map.end()) {
                    final_val = v_rep_map.at(final_val);
                }
            } else {
                static const std::unordered_map<std::string, std::string> pinyin_rep_map = {
                    {"ing", "ying"}, {"i", "yi"}, {"in", "yin"}, {"u", "wu"}
                };
                std::string py_full = final_val;
                if (pinyin_rep_map.find(py_full) != pinyin_rep_map.end()) {
                    std::string mapped = pinyin_rep_map.at(py_full);
                    initial = mapped.substr(0, 1); // approximate split
                    final_val = mapped.substr(1);
                } else {
                    static const std::unordered_map<char, std::string> single_rep_map = {
                        {'v', "yu"}, {'e', "e"}, {'i', "y"}, {'u', "w"}
                    };
                    if (!py_full.empty() && single_rep_map.find(py_full[0]) != single_rep_map.end()) {
                        std::string mapped = single_rep_map.at(py_full[0]) + py_full.substr(1);
                        if (mapped.rfind("yu", 0) == 0) {
                            initial = "y";
                            final_val = mapped.substr(1);
                        } else {
                            initial = mapped.substr(0, 1);
                            final_val = mapped.substr(1);
                        }
                    }
                }
            }
        }

        initials.push_back(initial);
        finals.push_back(final_val + tone);
    }
    return {initials, finals};
}

std::pair<std::vector<std::string>, std::vector<std::string>> Phonemizer::merge_erhua(
    const std::vector<std::string>& initials,
    const std::vector<std::string>& finals,
    const std::string& word,
    const std::string& pos) const {

    std::vector<std::string> adj_finals = finals;

    // fix er1
    std::u32string word_u32 = utf8_to_utf32(word);
    for (size_t i = 0; i < adj_finals.size(); ++i) {
        if (i == adj_finals.size() - 1 && word_u32[i] == U'儿' && adj_finals[i] == "er1") {
            adj_finals[i] = "er2";
        }
    }

    if (must_erhua.find(word_u32) == must_erhua.end() && (not_erhua.find(word_u32) != not_erhua.end() || pos == "a" || pos == "j" || pos == "nr")) {
        return {initials, adj_finals};
    }

    if (adj_finals.size() != word_u32.size()) {
        return {initials, adj_finals};
    }

    std::vector<std::string> new_initials;
    std::vector<std::string> new_finals;
    new_initials.reserve(initials.size());
    new_finals.reserve(adj_finals.size());

    for (size_t i = 0; i < adj_finals.size(); ++i) {
        std::string phn = adj_finals[i];
        if (i == adj_finals.size() - 1 && word_u32[i] == U'儿' && (phn == "er2" || phn == "er5") &&
            word_u32.size() >= 2 && not_erhua.find(word_u32.substr(word_u32.size() - 2)) == not_erhua.end() &&
            !new_finals.empty()) {
            
            char prev_tone = new_finals.back().back();
            phn = "er";
            phn.push_back(prev_tone);
        }
        new_initials.push_back(initials[i]);
        new_finals.push_back(phn);
    }

    return {new_initials, new_finals};
}

PhonemizerResult Phonemizer::process_en(const std::string& text) {
    PhonemizerResult result;

    // --- English normalization: simple punctuation replacement ---
    // Map special punctuations to standard ASCII equivalents
    std::string norm = text;
    // Replace em dash and similar to hyphen
    {
        // Work on UTF-32 for punctuation mapping
        std::u32string u32 = utf8_to_utf32(norm);
        std::u32string out;
        out.reserve(u32.size());
        static const std::unordered_map<char32_t, char32_t> punc_norm_map = {
            {U'\u2014', U'-'},  // em dash -> hyphen
            {U'\u2013', U'-'},  // en dash -> hyphen
            {U'\u2018', U'\''}, // left single quote -> apostrophe
            {U'\u2019', U'\''}, // right single quote -> apostrophe
            {U'\u201c', U'"'}, // left double quote
            {U'\u201d', U'"'}, // right double quote
            {U'\u2026', U'.'}, // ellipsis -> period
            {U'\uff0c', U','}, // fullwidth comma
            {U'\u3002', U'.'}, // ideographic full stop
        };
        for (char32_t cp : u32) {
            auto it = punc_norm_map.find(cp);
            out.push_back(it != punc_norm_map.end() ? it->second : cp);
        }
        norm = utf32_to_utf8(out);
    }
    result.norm_text = norm;

    // --- Tokenize: split into words and punctuation tokens ---
    // Words = sequences of [A-Za-z'] (apostrophe for contractions)
    // Punctuation = individual non-space, non-alpha characters
    struct Token {
        std::string text;
        bool is_word; // true=word, false=punctuation
    };
    std::vector<Token> tokens;
    {
        size_t i = 0;
        while (i < norm.size()) {
            char c = norm[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++i;
                continue;
            }
            // Check if start of a word (ASCII letter or apostrophe at start of token)
            if (std::isalpha((unsigned char)c)) {
                size_t start = i;
                while (i < norm.size() && (std::isalpha((unsigned char)norm[i]) || norm[i] == '\'')) {
                    ++i;
                }
                // Trim trailing apostrophe
                std::string word = norm.substr(start, i - start);
                while (!word.empty() && word.back() == '\'') word.pop_back();
                if (!word.empty()) {
                    tokens.push_back({word, true});
                }
            } else {
                // Punctuation (single byte, ASCII)
                tokens.push_back({std::string(1, c), false});
                ++i;
            }
        }
    }

    // --- Map tokens to phonemes ---
    // Punctuation maps: keep only valid GPT-SoVITS punctuation symbols
    static const std::unordered_map<std::string, std::string> punc_symbol_map = {
        {",", ","}, {".", "."}, {"!", "!"}, {"?", "?"}, {"-", "-"},
        {";", ","}, {":", ","}, {"'", "-"}, {"~", "…"}
    };

    for (const auto& tok : tokens) {
        if (!tok.is_word) {
            // Punctuation
            auto pit = punc_symbol_map.find(tok.text);
            if (pit != punc_symbol_map.end()) {
                result.phones.push_back(pit->second);
                result.word2ph.push_back(1);
            }
            // else: skip unknown punctuation
            continue;
        }

        // Word: lowercase for lookup
        std::string lower = tok.text;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        auto dict_it = english_dict.find(lower);
        if (dict_it != english_dict.end()) {
            // Found in CMUDict
            const auto& phones = dict_it->second;
            int count = 0;
            for (const auto& ph : phones) {
                if (symbols.find(ph) != symbols.end()) {
                    result.phones.push_back(ph);
                    ++count;
                } else {
                    // Not in valid symbols - skip silently (e.g. rare arpa variants)
                }
            }
            if (count == 0) count = 1; // avoid zero-phone words for BERT alignment
            result.word2ph.push_back(count);
        } else {
            // OOV word: spell it out letter by letter
            int count = 0;
            for (char ch : lower) {
                if (!std::isalpha((unsigned char)ch)) continue;
                std::string letter(1, ch);
                auto lit = english_dict.find(letter);
                if (lit != english_dict.end()) {
                    for (const auto& ph : lit->second) {
                        if (symbols.find(ph) != symbols.end()) {
                            result.phones.push_back(ph);
                            ++count;
                        }
                    }
                } else {
                    // Fallback: push UNK
                    result.phones.push_back("UNK");
                    ++count;
                }
            }
            if (count == 0) count = 1;
            result.word2ph.push_back(count);
        }
    }

    // Ensure at least one phone (Python does: if len(phones) < 4: phones = [","] + phones)
    if (result.phones.size() < 4) {
        result.phones.insert(result.phones.begin(), ",");
        result.word2ph.insert(result.word2ph.begin(), 1);
    }

    return result;
}

PhonemizerResult Phonemizer::process(const std::string& text, const std::string& lang) {
    // Dispatch to English processor for 'en' language
    if (lang == "en") {
        return process_en(text);
    }

    PhonemizerResult result;
    result.norm_text = normalizer->normalize_sentence(text);

    std::vector<std::pair<std::string, std::string>> tagged_words;
    jieba->Tag(result.norm_text, tagged_words);

    std::vector<std::pair<std::string, std::string>> merged_seg = tone_modifier->pre_merge_for_modify(tagged_words);

    for (const auto& pair : merged_seg) {
        std::string word = pair.first;
        std::string pos = pair.second;

        std::u32string word_u32 = utf8_to_utf32(word);
        if (word_u32.empty()) continue;

        // If it's punctuation
        if (is_punctuation(word_u32[0])) {
            std::string punc_str = utf32_char_to_utf8(word_u32[0]);
            
            // Map standard punctuations
            static const std::unordered_map<std::string, std::string> punc_map = {
                {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, {"！", "!"}, {"？", "?"},
                {"、", ","}, {"·", ","}, {"/", ","}, {"—", "-"}, {"~", "…"}, {"～", "…"}
            };
            if (punc_map.find(punc_str) != punc_map.end()) {
                punc_str = punc_map.at(punc_str);
            }

            result.phones.push_back(punc_str);
            result.word2ph.push_back(1);
            continue;
        }

        // Get initials and finals
        auto init_fin = get_initials_finals(word);
        std::vector<std::string> initials = init_fin.first;
        std::vector<std::string> finals = init_fin.second;

        // Apply tone sandhi
        finals = tone_modifier->modified_tone(word, pos, finals);

        // Apply erhua (only for V2/V2Pro)
        std::vector<std::string> new_initials = initials;
        std::vector<std::string> new_finals = finals;
        if (version_ != 1) {
            auto erhua_res = merge_erhua(initials, finals, word, pos);
            new_initials = erhua_res.first;
            new_finals = erhua_res.second;
        }



        // Construct final phones
        for (size_t i = 0; i < new_finals.size(); ++i) {
            std::string init = new_initials[i];
            std::string fin = new_finals[i];

            if (init.empty() && fin.empty()) continue;

            char tone = fin.back();
            std::string fin_no_tone = fin.substr(0, fin.size() - 1);
            std::string py = init + fin_no_tone;

            // Map pinyin to strict symbols
            std::string mapped_init = init;
            std::string mapped_fin = fin_no_tone;
            if (pinyin_to_symbol_map.find(py) != pinyin_to_symbol_map.end()) {
                mapped_init = pinyin_to_symbol_map.at(py).first;
                mapped_fin = pinyin_to_symbol_map.at(py).second;
            }

            std::string final_phone = mapped_fin + tone;

            // Check validity against symbols
            int word_ph_count = 0;
            if (symbols.find(mapped_init) != symbols.end()) {
                result.phones.push_back(mapped_init);
                word_ph_count++;
            }
            if (symbols.find(final_phone) != symbols.end()) {
                result.phones.push_back(final_phone);
                word_ph_count++;
            } else {
                result.phones.push_back("UNK");
                word_ph_count++;
            }
            result.word2ph.push_back(word_ph_count);
        }
    }

    return result;
}

} // namespace phonemizer
