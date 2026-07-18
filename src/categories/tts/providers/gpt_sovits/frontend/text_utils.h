#pragma once

#include <string>
#include <vector>
#include <unordered_set>

namespace gpt_sovits {
namespace text {

// Provider-local multilingual text splitting utilities.

extern const std::unordered_set<char32_t> splits_dict;

bool is_subset_of_punctuation(const std::u32string& str);
bool is_decimal_point(const std::u32string& inp, size_t i);
std::u32string clean_formatted_decimals(const std::u32string& text);
std::vector<std::u32string> split_python(std::u32string todo_text);

std::vector<std::u32string> cut0(const std::u32string& inp);
std::vector<std::u32string> cut1(const std::u32string& inp);
std::vector<std::u32string> cut2(const std::u32string& inp);
std::vector<std::u32string> cut3(const std::u32string& inp);
std::vector<std::u32string> cut4(const std::u32string& inp);
std::vector<std::u32string> cut5(const std::u32string& inp);
std::vector<std::u32string> split_by_sentence_ends(const std::u32string& inp);

} // namespace text
} // namespace gpt_sovits
