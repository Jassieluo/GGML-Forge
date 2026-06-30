#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "utf8_utils.hpp"
#include "tone_sandhi_words.hpp"

namespace phonemizer {

class ToneSandhi {
private:
    const std::unordered_map<std::string, std::vector<std::string>>* phrase_map;
    const std::unordered_map<char32_t, std::vector<std::string>>* single_map;
    std::u32string punc_u32;

public:
    ToneSandhi() : phrase_map(nullptr), single_map(nullptr) {
        punc_u32 = utf8_to_utf32("：，；。？！“”‘’':,;.?!");
    }

    void init(const std::unordered_map<std::string, std::vector<std::string>>* p_map,
              const std::unordered_map<char32_t, std::vector<std::string>>* s_map) {
        phrase_map = p_map;
        single_map = s_map;
    }

    std::vector<std::string> get_base_pinyins(const std::string& word) const {
        if (phrase_map && phrase_map->find(word) != phrase_map->end()) {
            return phrase_map->at(word);
        }
        std::u32string word_u32 = utf8_to_utf32(word);
        if (word_u32.size() > 1 && word_u32.back() == U'儿') {
            std::string prefix = utf32_to_utf8(word_u32.substr(0, word_u32.size() - 1));
            if (phrase_map && phrase_map->find(prefix) != phrase_map->end()) {
                std::vector<std::string> result = phrase_map->at(prefix);
                char32_t cp = U'儿';
                if (single_map && single_map->find(cp) != single_map->end() && !single_map->at(cp).empty()) {
                    result.push_back(single_map->at(cp)[0]);
                } else {
                    result.push_back("er2");
                }
                return result;
            }
        }
        std::vector<std::string> result;
        for (char32_t cp : word_u32) {
            if (single_map && single_map->find(cp) != single_map->end() && !single_map->at(cp).empty()) {
                // Return first (most common) pinyin
                result.push_back(single_map->at(cp)[0]);
            } else {
                result.push_back("");
            }
        }
        return result;
    }

    bool all_tone_three(const std::vector<std::string>& pinyins) const {
        if (pinyins.empty()) return false;
        for (const auto& p : pinyins) {
            if (p.empty() || p.back() != '3') return false;
        }
        return true;
    }

    bool is_reduplication(const std::u32string& word) const {
        return word.size() == 2 && word[0] == word[1];
    }

    inline bool is_numeric_python(char32_t cp) const {
        if (cp >= '0' && cp <= '9') return true;
        static const std::u32string cn_nums = U"一二三四五六七八九十百千万亿零";
        return cn_nums.find(cp) != std::u32string::npos;
    }

    std::vector<std::pair<std::string, std::string>> pre_merge_for_modify(
        const std::vector<std::pair<std::string, std::string>>& seg) const {
        
        // Split mis-segmented "儿 + locative" tokens (e.g. "儿里" -> "儿" + "里")
        std::vector<std::pair<std::string, std::string>> split_seg;
        for (const auto& pair : seg) {
            std::u32string w_u32 = utf8_to_utf32(pair.first);
            if (w_u32.size() >= 2 && w_u32[0] == U'儿') {
                char32_t next = w_u32[1];
                if (next == U'里' || next == U'外' || next == U'上' || next == U'下' || 
                    next == U'边' || next == U'旁' || next == U'中' || next == U'后' || next == U'前') {
                    split_seg.push_back({utf32_to_utf8(U"儿"), "x"});
                    split_seg.push_back({utf32_to_utf8(w_u32.substr(1)), "f"});
                    continue;
                }
            }
            split_seg.push_back(pair);
        }

        // 0. Pre-merge "V 不 C" structures like "看不懂", "做不好"
        std::vector<std::pair<std::string, std::string>> seg0;
        size_t idx = 0;
        while (idx < split_seg.size()) {
            if (idx + 2 < split_seg.size() && split_seg[idx + 1].first == "不") {
                std::u32string w0_u32 = utf8_to_utf32(split_seg[idx].first);
                std::u32string w2_u32 = utf8_to_utf32(split_seg[idx + 2].first);
                if (w0_u32.size() == 1 && w2_u32.size() == 1 && 
                    !is_punctuation(w0_u32[0]) && !is_punctuation(w2_u32[0])) {
                    std::string merged_word = split_seg[idx].first + "不" + split_seg[idx + 2].first;
                    seg0.push_back({merged_word, "v"});
                    idx += 3;
                    continue;
                }
            }
            seg0.push_back(split_seg[idx]);
            idx += 1;
        }

        // 1. _merge_bu
        std::vector<std::pair<std::string, std::string>> seg1;
        std::string last_word = "";
        std::string last_pos = "";
        for (const auto& pair : seg0) {
            std::string word = pair.first;
            std::string pos = pair.second;
            if (last_word == "不") {
                word = last_word + word;
            }
            if (word != "不") {
                seg1.push_back({word, pos});
            }
            last_word = pair.first;
            last_pos = pair.second;
        }
        if (last_word == "不") {
            seg1.push_back({last_word, "d"});
        }

        // 2. _merge_yi
        std::vector<std::pair<std::string, std::string>> seg2;
        size_t i = 0;
        // Function 1
        while (i < seg1.size()) {
            std::string word = seg1[i].first;
            std::string pos = seg1[i].second;
            bool merged = false;
            if (i >= 1 && word == "一" && i + 1 < seg1.size()) {
                auto last = !seg2.empty() ? seg2.back() : seg1[i - 1];
                if (last.first == seg1[i + 1].first && last.second == "v" && seg1[i + 1].second == "v") {
                    std::string combined = last.first + "一" + seg1[i + 1].first;
                    if (!seg2.empty()) {
                        seg2.back() = {combined, last.second};
                    } else {
                        seg2.push_back({combined, last.second});
                    }
                    i += 2;
                    merged = true;
                }
            }
            if (!merged) {
                seg2.push_back({word, pos});
                i += 1;
            }
        }
        // Function 2
        std::vector<std::pair<std::string, std::string>> seg3;
        for (const auto& pair : seg2) {
            if (!seg3.empty() && seg3.back().first == "一") {
                seg3.back().first = seg3.back().first + pair.first;
            } else {
                seg3.push_back(pair);
            }
        }

        // 3. _merge_reduplication
        std::vector<std::pair<std::string, std::string>> seg4;
        for (const auto& pair : seg3) {
            if (!seg4.empty() && pair.first == seg4.back().first && !is_punctuation(utf8_to_utf32(pair.first)[0])) {
                seg4.back().first = seg4.back().first + pair.first;
            } else {
                seg4.push_back(pair);
            }
        }

        // 4. _merge_continuous_three_tones & _merge_continuous_three_tones_2
        std::vector<std::pair<std::string, std::string>> seg5;
        std::vector<std::vector<std::string>> sub_finals_list;
        for (const auto& pair : seg4) {
            sub_finals_list.push_back(get_base_pinyins(pair.first));
        }

        std::vector<bool> merge_last(seg4.size(), false);
        for (size_t k = 0; k < seg4.size(); ++k) {
            std::string word = seg4[k].first;
            std::string pos = seg4[k].second;
            std::u32string word_u32 = utf8_to_utf32(word);

            if (k >= 1 && all_tone_three(sub_finals_list[k - 1]) && all_tone_three(sub_finals_list[k]) && !merge_last[k - 1]) {
                std::u32string last_u32 = utf8_to_utf32(seg4[k - 1].first);
                if (!is_reduplication(last_u32) && last_u32.size() + word_u32.size() <= 3) {
                    seg5.back().first = seg5.back().first + word;
                    merge_last[k] = true;
                } else {
                    seg5.push_back({word, pos});
                }
            } else {
                seg5.push_back({word, pos});
            }
        }

        // Three tones 2
        std::vector<std::pair<std::string, std::string>> seg6;
        std::vector<std::vector<std::string>> sub_finals_list2;
        for (const auto& pair : seg5) {
            sub_finals_list2.push_back(get_base_pinyins(pair.first));
        }
        std::vector<bool> merge_last2(seg5.size(), false);
        for (size_t k = 0; k < seg5.size(); ++k) {
            std::string word = seg5[k].first;
            std::string pos = seg5[k].second;
            std::u32string word_u32 = utf8_to_utf32(word);

            if (k >= 1 && !sub_finals_list2[k - 1].empty() && !sub_finals_list2[k - 1].back().empty() &&
                !sub_finals_list2[k].empty() && !sub_finals_list2[k][0].empty() &&
                sub_finals_list2[k - 1].back().back() == '3' && sub_finals_list2[k][0].back() == '3' && !merge_last2[k - 1]) {
                std::u32string last_u32 = utf8_to_utf32(seg5[k - 1].first);
                if (!is_reduplication(last_u32) && last_u32.size() + word_u32.size() <= 3) {
                    seg6.back().first = seg6.back().first + word;
                    merge_last2[k] = true;
                } else {
                    seg6.push_back({word, pos});
                }
            } else {
                seg6.push_back({word, pos});
            }
        }

        // 5. _merge_er
        std::vector<std::pair<std::string, std::string>> seg7;
        for (size_t k = 0; k < seg6.size(); ++k) {
            std::string word = seg6[k].first;
            std::string pos = seg6[k].second;
            if (k >= 1 && word == "儿" && seg6[k - 1].first != "#" && !seg7.empty()) {
                seg7.back().first = seg7.back().first + word;
            } else {
                seg7.push_back({word, pos});
            }
        }

        return seg7;
    }

    std::vector<std::string> bu_sandhi(const std::u32string& word, std::vector<std::string> finals) const {
        if (word.size() == 3 && word[1] == U'不') {
            if (finals.size() > 1) {
                finals[1] = finals[1].substr(0, finals[1].size() - 1) + "5";
            }
        } else {
            for (size_t i = 0; i < word.size(); ++i) {
                if (word[i] == U'不' && i + 1 < word.size() && finals.size() > i + 1) {
                    if (finals[i + 1].back() == '4') {
                        finals[i] = finals[i].substr(0, finals[i].size() - 1) + "2";
                    }
                }
            }
        }
        return finals;
    }

    std::vector<std::string> yi_sandhi(const std::u32string& word, std::vector<std::string> finals) const {
        bool all_num = true;
        bool has_yi = false;
        for (char32_t cp : word) {
            if (cp == U'一') has_yi = true;
            else if (!is_numeric_python(cp)) all_num = false;
        }

        if (has_yi && all_num) {
            return finals;
        } else if (word.size() == 3 && word[1] == U'一' && word[0] == word[2]) {
            if (finals.size() > 1) {
                finals[1] = finals[1].substr(0, finals[1].size() - 1) + "5";
            }
        } else if (word.rfind(U"第一", 0) == 0) {
            if (finals.size() > 1) {
                finals[1] = finals[1].substr(0, finals[1].size() - 1) + "1";
            }
        } else {
            for (size_t i = 0; i < word.size(); ++i) {
                if (word[i] == U'一' && i + 1 < word.size() && finals.size() > i + 1) {
                    if (finals[i + 1].back() == '4') {
                        finals[i] = finals[i].substr(0, finals[i].size() - 1) + "2";
                    } else {
                        if (punc_u32.find(word[i + 1]) == std::u32string::npos) {
                            finals[i] = finals[i].substr(0, finals[i].size() - 1) + "4";
                        }
                    }
                }
            }
        }
        return finals;
    }

    std::vector<std::string> split_word(const std::u32string& word) const {
        // Find split cut: split 3-character word into shortest components
        if (word.size() <= 2) return {utf32_to_utf8(word)};
        std::u32string part0 = word.substr(0, 2);
        std::u32string part1 = word.substr(2);
        return {utf32_to_utf8(part0), utf32_to_utf8(part1)};
    }

    std::vector<std::string> neural_sandhi(const std::u32string& word, const std::string& pos, std::vector<std::string> finals) const {
        // Reduplication words for n., v., a., e.g. 试试, 奶奶
        for (size_t j = 1; j < word.size(); ++j) {
            if (word[j] == word[j - 1] && !pos.empty() && (pos[0] == 'n' || pos[0] == 'v' || pos[0] == 'a')) {
                std::string word_utf8 = utf32_to_utf8(word);
                if (MUST_NOT_NEURAL_TONE_WORDS.find(word_utf8) == MUST_NOT_NEURAL_TONE_WORDS.end()) {
                    if (finals.size() > j) {
                        finals[j] = finals[j].substr(0, finals[j].size() - 1) + "5";
                    }
                }
            }
        }

        // Particle suffix
        std::u32string particles = U"吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶的地得";
        if (!word.empty() && particles.find(word.back()) != std::u32string::npos) {
            if (!finals.empty()) {
                finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
            }
        }

        // 了着过
        if (word.size() == 1 && (word[0] == U'了' || word[0] == U'着' || word[0] == U'过') && (pos == "ul" || pos == "uz" || pos == "ug")) {
            if (!finals.empty()) {
                finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
            }
        }

        // 们子
        if (word.size() > 1 && (word.back() == U'们' || word.back() == U'子') && (pos[0] == 'r' || pos[0] == 'n')) {
            std::string word_utf8 = utf32_to_utf8(word);
            if (MUST_NOT_NEURAL_TONE_WORDS.find(word_utf8) == MUST_NOT_NEURAL_TONE_WORDS.end()) {
                if (!finals.empty()) {
                    finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
                }
            }
        }

        // 上下里
        if (word.size() > 1 && (word.back() == U'上' || word.back() == U'下' || word.back() == U'里') && (pos[0] == 's' || pos[0] == 'l' || pos[0] == 'f')) {
            if (!finals.empty()) {
                finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
            }
        }

        // 来去 direction
        if (word.size() > 1 && (word.back() == U'来' || word.back() == U'去')) {
            char32_t prev = word[word.size() - 2];
            std::u32string dirs = U"上下进出回过起开";
            if (dirs.find(prev) != std::u32string::npos) {
                if (!finals.empty()) {
                    finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
                }
            }
        }

        // 个 quantifier
        size_t ge_idx = word.find(U'个');
        if (ge_idx != std::u32string::npos) {
            bool matches_ge = false;
            if (ge_idx >= 1) {
                char32_t prev = word[ge_idx - 1];
                if (is_numeric_python(prev) || std::u32string(U"几有两半多各整每做是").find(prev) != std::u32string::npos) {
                    matches_ge = true;
                }
            } else if (word == U"个") {
                matches_ge = true;
            }
            if (matches_ge && finals.size() > ge_idx) {
                finals[ge_idx] = finals[ge_idx].substr(0, finals[ge_idx].size() - 1) + "5";
            }
        }

        // Check overall MUST_NEURAL_TONE_WORDS
        std::string word_utf8 = utf32_to_utf8(word);
        if (MUST_NEURAL_TONE_WORDS.find(word_utf8) != MUST_NEURAL_TONE_WORDS.end() ||
            (word_utf8.size() >= 6 && MUST_NEURAL_TONE_WORDS.find(word_utf8.substr(word_utf8.size() - 6)) != MUST_NEURAL_TONE_WORDS.end())) {
            if (!finals.empty()) {
                finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
            }
        }

        // Split word check
        std::vector<std::string> word_list = split_word(word);
        if (word_list.size() == 2) {
            std::u32string p0 = utf8_to_utf32(word_list[0]);
            std::u32string p1 = utf8_to_utf32(word_list[1]);
            if (MUST_NEURAL_TONE_WORDS.find(word_list[0]) != MUST_NEURAL_TONE_WORDS.end() ||
                (word_list[0].size() >= 6 && MUST_NEURAL_TONE_WORDS.find(word_list[0].substr(word_list[0].size() - 6)) != MUST_NEURAL_TONE_WORDS.end())) {
                if (finals.size() > p0.size() - 1) {
                    finals[p0.size() - 1] = finals[p0.size() - 1].substr(0, finals[p0.size() - 1].size() - 1) + "5";
                }
            }
            if (MUST_NEURAL_TONE_WORDS.find(word_list[1]) != MUST_NEURAL_TONE_WORDS.end() ||
                (word_list[1].size() >= 6 && MUST_NEURAL_TONE_WORDS.find(word_list[1].substr(word_list[1].size() - 6)) != MUST_NEURAL_TONE_WORDS.end())) {
                if (!finals.empty()) {
                    finals.back() = finals.back().substr(0, finals.back().size() - 1) + "5";
                }
            }
        }

        return finals;
    }

    std::vector<std::string> three_sandhi(const std::u32string& word, std::vector<std::string> finals) const {
        if (word.size() == 2 && all_tone_three(finals)) {
            if (finals.size() > 0) {
                finals[0] = finals[0].substr(0, finals[0].size() - 1) + "2";
            }
        } else if (word.size() == 3) {
            std::vector<std::string> word_list = split_word(word);
            std::u32string p0 = utf8_to_utf32(word_list[0]);
            std::u32string p1 = utf8_to_utf32(word_list[1]);
            
            if (all_tone_three(finals)) {
                if (p0.size() == 2) { // 2 + 1
                    if (finals.size() > 1) {
                        finals[0] = finals[0].substr(0, finals[0].size() - 1) + "2";
                        finals[1] = finals[1].substr(0, finals[1].size() - 1) + "2";
                    }
                } else if (p0.size() == 1) { // 1 + 2
                    if (finals.size() > 1) {
                        finals[1] = finals[1].substr(0, finals[1].size() - 1) + "2";
                    }
                }
            } else {
                std::vector<std::string> f0(finals.begin(), finals.begin() + p0.size());
                std::vector<std::string> f1(finals.begin() + p0.size(), finals.end());
                if (all_tone_three(f0) && f0.size() == 2) {
                    finals[0] = finals[0].substr(0, finals[0].size() - 1) + "2";
                }
                if (all_tone_three(f1) && f1.size() == 2) {
                    finals[p0.size()] = finals[p0.size()].substr(0, finals[p0.size()].size() - 1) + "2";
                }
                else if (!f0.empty() && !f1.empty() && !all_tone_three(f1) && f1[0].back() == '3' && f0.back().back() == '3') {
                    finals[p0.size() - 1] = finals[p0.size() - 1].substr(0, finals[p0.size() - 1].size() - 1) + "2";
                }
            }
        } else if (word.size() == 4) {
            // Split into two 2-char components
            if (finals.size() >= 4) {
                std::vector<std::string> f0 = {finals[0], finals[1]};
                std::vector<std::string> f1 = {finals[2], finals[3]};
                if (all_tone_three(f0)) {
                    finals[0] = finals[0].substr(0, finals[0].size() - 1) + "2";
                }
                if (all_tone_three(f1)) {
                    finals[2] = finals[2].substr(0, finals[2].size() - 1) + "2";
                }
            }
        }
        return finals;
    }

    std::vector<std::string> modified_tone(const std::string& word_utf8, const std::string& pos, std::vector<std::string> finals) const {
        std::u32string word_u32 = utf8_to_utf32(word_utf8);
        finals = bu_sandhi(word_u32, finals);
        finals = yi_sandhi(word_u32, finals);
        finals = neural_sandhi(word_u32, pos, finals);
        finals = three_sandhi(word_u32, finals);
        return finals;
    }
};

} // namespace phonemizer
