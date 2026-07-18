#include "providers/gpt_sovits/frontend/text_utils.h"
#include <algorithm>

namespace gpt_sovits {
namespace text {

static std::u32string replace_all_u32(std::u32string str, const std::u32string& from, const std::u32string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::u32string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

const std::unordered_set<char32_t> splits_dict = {
    U'，', U'。', U'？', U'！', U',', U'.', U'?', U'!', U'~', U':', U'：', U'—', U'…', U'、', U';', U'；'
};

bool is_subset_of_punctuation(const std::u32string& str) {
    static const std::unordered_set<char32_t> puncs = {
        U'!', U'?', U'…', U',', U'.', U'-', U' ', U'\t', U'\r', U'\n',
        U'，', U'。', U'？', U'！', U'~', U':', U'：', U'—', U'、', U';', U'；'
    };
    for (char32_t cp : str) {
        if (puncs.find(cp) == puncs.end()) {
            return false;
        }
    }
    return true;
}

bool is_decimal_point(const std::u32string& inp, size_t i) {
    if (inp[i] != U'.') return false;
    if (i == 0 || i + 1 >= inp.size()) return false;
    char32_t prev = inp[i - 1];
    if (prev < U'0' || prev > U'9') return false;
    for (size_t j = i + 1; j < inp.size(); ++j) {
        char32_t next = inp[j];
        if (next >= U'0' && next <= U'9') {
            return true;
        }
        if ((next >= 0x4E00 && next <= 0x9FFF) ||
            (next >= U'A' && next <= U'Z') ||
            (next >= U'a' && next <= U'z') ||
            next == U'。' || next == U'？' || next == U'！' || next == U'?' || next == U'!' ||
            next == U'，' || next == U',' || next == U'、' || next == U'；' || next == U';') {
            break;
        }
    }
    return false;
}

std::u32string clean_formatted_decimals(const std::u32string& text) {
    std::u32string result;
    result.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        char32_t cp = text[i];
        result.push_back(cp);
        if (cp == U'.' && result.size() >= 2 && result[result.size() - 2] >= U'0' && result[result.size() - 2] <= U'9') {
            size_t next_digit_idx = 0;
            for (size_t j = i + 1; j < text.size(); ++j) {
                char32_t n = text[j];
                if (n >= U'0' && n <= U'9') {
                    next_digit_idx = j;
                    break;
                }
                if (n != U' ' && n != U'\t' && n != U'\r' && n != U'\n' && n != U'|' && n != U'│') {
                    break;
                }
            }
            if (next_digit_idx > 0) {
                i = next_digit_idx - 1;
            }
        }
    }
    return result;
}

std::vector<std::u32string> split_python(std::u32string todo_text) {
    if (todo_text.empty()) return {};
    todo_text = replace_all_u32(todo_text, U"……", U"。");
    todo_text = replace_all_u32(todo_text, U"——", U"，");
    if (splits_dict.find(todo_text.back()) == splits_dict.end()) {
        todo_text += U"。";
    }
    size_t i_split_head = 0;
    size_t i_split_tail = 0;
    size_t len_text = todo_text.size();
    std::vector<std::u32string> todo_texts;
    while (true) {
        if (i_split_head >= len_text) break;
        if (splits_dict.find(todo_text[i_split_head]) != splits_dict.end()) {
            i_split_head++;
            todo_texts.push_back(todo_text.substr(i_split_tail, i_split_head - i_split_tail));
            i_split_tail = i_split_head;
        } else {
            i_split_head++;
        }
    }
    return todo_texts;
}

std::vector<std::u32string> cut0(const std::u32string& inp) {
    if (!is_subset_of_punctuation(inp)) return { inp };
    return {};
}

std::vector<std::u32string> cut1(const std::u32string& inp) {
    std::vector<std::u32string> inps = split_python(inp);
    if (inps.empty()) return {};
    std::vector<std::u32string> opts;
    for (size_t idx = 0; idx < inps.size(); idx += 4) {
        std::u32string merged = U"";
        for (size_t k = 0; k < 4 && idx + k < inps.size(); ++k) {
            merged += inps[idx + k];
        }
        opts.push_back(merged);
    }
    std::vector<std::u32string> filtered_opts;
    for (const auto& item : opts) {
        if (!is_subset_of_punctuation(item)) filtered_opts.push_back(item);
    }
    return filtered_opts;
}

std::vector<std::u32string> cut2(const std::u32string& inp) {
    std::vector<std::u32string> inps = split_python(inp);
    if (inps.size() < 2) {
        if (!is_subset_of_punctuation(inp)) return { inp };
        return {};
    }
    std::vector<std::u32string> opts;
    size_t summ = 0;
    std::u32string tmp_str = U"";
    for (size_t i = 0; i < inps.size(); ++i) {
        summ += inps[i].size();
        tmp_str += inps[i];
        if (summ > 50) {
            summ = 0;
            opts.push_back(tmp_str);
            tmp_str = U"";
        }
    }
    if (!tmp_str.empty()) opts.push_back(tmp_str);
    if (opts.size() > 1 && opts.back().size() < 50) {
        opts[opts.size() - 2] = opts[opts.size() - 2] + opts.back();
        opts.pop_back();
    }
    std::vector<std::u32string> filtered_opts;
    for (const auto& item : opts) {
        if (!is_subset_of_punctuation(item)) filtered_opts.push_back(item);
    }
    return filtered_opts;
}

std::vector<std::u32string> cut3(const std::u32string& inp) {
    std::vector<std::u32string> opts;
    std::u32string current = U"";
    for (char32_t cp : inp) {
        if (cp == U'。') {
            if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);
            current = U"";
        } else {
            current += cp;
        }
    }
    if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);
    return opts;
}

std::vector<std::u32string> cut4(const std::u32string& inp) {
    std::vector<std::u32string> opts;
    std::u32string current = U"";
    for (size_t i = 0; i < inp.size(); ++i) {
        char32_t cp = inp[i];
        if (cp == U'.') {
            if (is_decimal_point(inp, i)) {
                current += cp;
            } else {
                if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);
                current = U"";
            }
        } else {
            current += cp;
        }
    }
    if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);
    return opts;
}

std::vector<std::u32string> cut5(const std::u32string& inp) {
    static const std::unordered_set<char32_t> punds = {
        U',', U'.', U';', U'?', U'!', U'、', U'，', U'。', U'？', U'！', U'；', U'：', U'…'
    };
    std::vector<std::u32string> mergeitems;
    std::u32string items = U"";
    for (size_t i = 0; i < inp.size(); ++i) {
        char32_t cp = inp[i];
        if (punds.find(cp) != punds.end()) {
            if (is_decimal_point(inp, i)) {
                items += cp;
            } else {
                items += cp;
                mergeitems.push_back(items);
                items = U"";
            }
        } else {
            items += cp;
        }
    }
    if (!items.empty()) mergeitems.push_back(items);
    std::vector<std::u32string> opts;
    for (const auto& item : mergeitems) {
        if (!is_subset_of_punctuation(item)) opts.push_back(item);
    }
    return opts;
}

std::vector<std::u32string> split_by_sentence_ends(const std::u32string& inp) {
    static const std::unordered_set<char32_t> sentence_ends = {
        U'。', U'？', U'！', U'.', U'?', U'!'
    };
    std::vector<std::u32string> res;
    std::u32string current = U"";

    for (size_t i = 0; i < inp.size(); ++i) {
        char32_t cp = inp[i];
        current += cp;

        if (sentence_ends.find(cp) != sentence_ends.end()) {
            if (!is_decimal_point(inp, i)) {
                res.push_back(current);
                current = U"";
            }
        }
    }

    if (!current.empty()) {
        res.push_back(current);
    }
    return res;
}

} // namespace text
} // namespace gpt_sovits
