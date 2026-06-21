#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <iostream>
#include "utf8_utils.hpp"
#include "char_convert.hpp"

namespace phonemizer {

class TextNormalizer {
private:
    std::unordered_map<char32_t, char32_t> t2s_map;
    std::unordered_map<char, std::string> digits_map;
    std::vector<std::pair<int, std::string>> units;

public:
    TextNormalizer() {
        // Load traditional to simplified map
        std::u32string simp = utf8_to_utf32(SIMPLIFIED_CHARS);
        std::u32string trad = utf8_to_utf32(TRADITIONAL_CHARS);
        for (size_t i = 0; i < simp.size() && i < trad.size(); ++i) {
            t2s_map[trad[i]] = simp[i];
        }

        // Digits map
        std::string d_str = "零一二三四五六七八九";
        std::u32string d_u32 = utf8_to_utf32(d_str);
        for (int i = 0; i < 10; ++i) {
            digits_map['0' + i] = utf32_char_to_utf8(d_u32[i]);
        }

        // Units map: (power, unit_string)
        units = {
            {1, "十"},
            {2, "百"},
            {3, "千"},
            {4, "万"},
            {8, "亿"}
        };
    }

    std::u32string to_simplified(const std::u32string& text) {
        std::u32string result;
        result.reserve(text.size());
        for (char32_t cp : text) {
            auto it = t2s_map.find(cp);
            if (it != t2s_map.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(cp);
            }
        }
        return result;
    }

    std::u32string fullwidth_to_halfwidth(const std::u32string& text) {
        std::u32string result;
        result.reserve(text.size());
        for (char32_t cp : text) {
            if (cp == 0x3000) {
                result.push_back(0x0020); // fullwidth space -> halfwidth space
            } else if (cp >= 0xFF01 && cp <= 0xFF5E) {
                result.push_back(cp - 0xFEE0); // fullwidth ASCII -> halfwidth ASCII
            } else {
                result.push_back(cp);
            }
        }
        return result;
    }

    std::string verbalize_digit(const std::string& num_str, bool alt_one = false) {
        std::string result = "";
        for (char c : num_str) {
            if (c >= '0' && c <= '9') {
                std::string digit = digits_map[c];
                if (alt_one && digit == "一") {
                    result += "幺";
                } else {
                    result += digit;
                }
            }
        }
        return result;
    }

    std::vector<std::string> get_cardinal_value(std::string num_str, bool use_zero = true) {
        // Strip leading zeros
        size_t first_non_zero = num_str.find_first_not_of('0');
        if (first_non_zero == std::string::npos) {
            return {};
        }
        std::string stripped = num_str.substr(first_non_zero);
        if (stripped.empty()) {
            return {};
        }

        if (stripped.size() == 1) {
            if (use_zero && stripped.size() < num_str.size()) {
                return {digits_map['0'], digits_map[stripped[0]]};
            } else {
                return {digits_map[stripped[0]]};
            }
        } else {
            int largest_unit = 0;
            std::string largest_unit_str = "";
            for (auto it = units.rbegin(); it != units.rend(); ++it) {
                if (it->first < (int)stripped.size()) {
                    largest_unit = it->first;
                    largest_unit_str = it->second;
                    break;
                }
            }

            std::string first_part = num_str.substr(0, num_str.size() - largest_unit);
            std::string second_part = num_str.substr(num_str.size() - largest_unit);

            std::vector<std::string> first_val = get_cardinal_value(first_part, true);
            std::vector<std::string> second_val = get_cardinal_value(second_part, true);

            std::vector<std::string> result = first_val;
            result.push_back(largest_unit_str);
            result.insert(result.end(), second_val.begin(), second_val.end());
            return result;
        }
    }

    std::string verbalize_cardinal(const std::string& num_str) {
        size_t first_non_zero = num_str.find_first_not_of('0');
        if (first_non_zero == std::string::npos) {
            return digits_map['0'];
        }
        std::string stripped = num_str.substr(first_non_zero);
        if (stripped.empty()) {
            return digits_map['0'];
        }

        std::vector<std::string> result_symbols = get_cardinal_value(stripped, true);
        if (result_symbols.size() >= 2 && result_symbols[0] == "一" && result_symbols[1] == "十") {
            result_symbols.erase(result_symbols.begin());
        }

        std::string result = "";
        for (const auto& sym : result_symbols) {
            result += sym;
        }
        return result;
    }

    std::string num2str(const std::string& num_str) {
        size_t dot_pos = num_str.find('.');
        std::string integer = "";
        std::string decimal = "";
        if (dot_pos == std::string::npos) {
            integer = num_str;
        } else {
            integer = num_str.substr(0, dot_pos);
            decimal = num_str.substr(dot_pos + 1);
        }

        std::string result = verbalize_cardinal(integer);

        // Strip trailing zeros from decimal, but keep at least one zero if it's all zeros
        if (!decimal.empty()) {
            size_t last_non_zero = decimal.find_last_not_of('0');
            if (last_non_zero == std::string::npos) {
                decimal = "0";
            } else {
                decimal = decimal.substr(0, last_non_zero + 1);
            }

            result = result.empty() ? "零" : result;
            result += "点" + verbalize_digit(decimal);
        }

        return result;
    }

    std::string normalize_sentence(const std::string& sentence_utf8) {
        std::u32string text_u32 = utf8_to_utf32(sentence_utf8);
        text_u32 = to_simplified(text_u32);
        text_u32 = fullwidth_to_halfwidth(text_u32);

        std::string text = utf32_to_utf8(text_u32);

        // Replace Greek letters
        static const std::vector<std::pair<std::string, std::string>> greek_replacements = {
            {"①", "一"}, {"②", "二"}, {"③", "三"}, {"④", "四"}, {"⑤", "五"},
            {"⑥", "六"}, {"⑦", "七"}, {"⑧", "八"}, {"⑨", "九"}, {"⑩", "十"},
            {"α", "阿尔法"}, {"β", "贝塔"}, {"γ", "伽玛"}, {"Γ", "伽玛"},
            {"δ", "德尔塔"}, {"Δ", "德尔塔"}, {"ε", "艾普西龙"}, {"ζ", "捷塔"},
            {"η", "依塔"}, {"θ", "西塔"}, {"Θ", "西塔"}, {"ι", "艾欧塔"},
            {"κ", "喀帕"}, {"λ", "拉姆达"}, {"Λ", "拉姆达"}, {"μ", "缪"},
            {"ν", "拗"}, {"ξ", "克西"}, {"Ξ", "克西"}, {"ο", "欧米克伦"},
            {"π", "派"}, {"Π", "派"}, {"ρ", "肉"}, {"ς", "西格玛"},
            {"Σ", "西格玛"}, {"σ", "西格玛"}, {"τ", "套"}, {"υ", "宇普西龙"},
            {"φ", "服艾"}, {"Φ", "服艾"}, {"χ", "器"}, {"ψ", "普赛"},
            {"Ψ", "普赛"}, {"ω", "欧米伽"}, {"Ω", "欧米伽"}
        };
        for (const auto& pair : greek_replacements) {
            size_t pos = 0;
            while ((pos = text.find(pair.first, pos)) != std::string::npos) {
                text.replace(pos, pair.first.size(), pair.second);
                pos += pair.second.size();
            }
        }

        // 1. Percentage replacement: (\d+(\.\d+)?)% -> 百分之...
        std::regex percent_regex(R"((-?)(\d+(?:\.\d+)?)%)");
        text = std::regex_replace(text, percent_regex, "百分之$2");

        // 2. Fraction replacement: (\d+)/(\d+) -> ...分之...
        std::regex frac_regex(R"((-?)(\d+)/(\d+))");
        // We can do custom match to handle nominator/denominator order
        std::smatch match;
        std::string search_str = text;
        std::string frac_result = "";
        while (std::regex_search(search_str, match, frac_regex)) {
            frac_result += match.prefix().str();
            std::string sign = match[1].str().empty() ? "" : "负";
            std::string nominator = num2str(match[2].str());
            std::string denominator = num2str(match[3].str());
            frac_result += sign + denominator + "分之" + nominator;
            search_str = match.suffix().str();
        }
        frac_result += search_str;
        text = frac_result;

        // 3. Mathematical operators replacement
        static const std::vector<std::pair<std::string, std::string>> math_ops = {
            {"+", "加"}, {"-", "减"}, {"×", "乘"}, {"÷", "除"}, {"=", "等于"}
        };
        for (const auto& pair : math_ops) {
            size_t pos = 0;
            while ((pos = text.find(pair.first, pos)) != std::string::npos) {
                text.replace(pos, pair.first.size(), pair.second);
                pos += pair.second.size();
            }
        }

        // 4. Default number normalization: sequences of digits (cardinal conversion)
        std::regex num_regex(R"((-?)(\d+(?:\.\d+)?))");
        search_str = text;
        std::string num_result = "";
        while (std::regex_search(search_str, match, num_regex)) {
            num_result += match.prefix().str();
            std::string sign = match[1].str().empty() ? "" : "负";
            std::string val = match[2].str();
            
            // PaddleSpeech normalization rules:
            // 1. Long sequences (>= 7 digits)
            // 2. Starts with zero (e.g. phone/zip code "010")
            // 3. Size >= 3 and does not end with "00" (e.g. "123" -> "一二三", but "100" -> "一百")
            bool is_digit_seq = (val.size() >= 7) || 
                                (val.size() > 1 && val[0] == '0' && val[1] != '.') ||
                                (val.size() >= 3 && !(val.size() >= 3 && val.substr(val.size() - 2) == "00"));
            
            if (is_digit_seq) {
                num_result += sign + verbalize_digit(val, true);
            } else {
                num_result += sign + num2str(val);
            }
            search_str = match.suffix().str();
        }
        num_result += search_str;
        text = num_result;

        // Clean up remaining special characters
        std::u32string final_u32 = utf8_to_utf32(text);
        std::u32string clean_u32;
        clean_u32.reserve(final_u32.size());
        for (char32_t cp : final_u32) {
            if (cp == '/' || cp == '\\' || cp == '<' || cp == '>' || cp == '{' || cp == '}' || 
                cp == '(' || cp == ')' || cp == '#' || cp == '&' || cp == '@' || cp == '^' || cp == '_' ||
                cp == '|' || cp == 0x2502) {
                continue; // filter out
            }
            clean_u32.push_back(cp);
        }

        return utf32_to_utf8(clean_u32);
    }
};

} // namespace phonemizer
