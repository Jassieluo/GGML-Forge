#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include "utf8_utils.hpp"
#include "text_normalizer.hpp"
#include "tone_sandhi.hpp"

#if defined(_WIN32)
#  if defined(PHONEMIZER_BUILD_SHARED)
#    define PHONEMIZER_API __declspec(dllexport)
#  elif defined(PHONEMIZER_USE_SHARED)
#    define PHONEMIZER_API __declspec(dllimport)
#  else
#    define PHONEMIZER_API
#  endif
#else
#  define PHONEMIZER_API
#endif

namespace cppjieba {
    class Jieba;
}

namespace phonemizer {

struct PhonemizerResult {
    std::vector<std::string> phones;
    std::vector<int> word2ph;
    std::string norm_text;
};

class PHONEMIZER_API Phonemizer {
private:
    std::unique_ptr<cppjieba::Jieba> jieba;
    std::unique_ptr<TextNormalizer> normalizer;
    std::unique_ptr<ToneSandhi> tone_modifier;

    std::unordered_map<std::string, std::vector<std::string>> phrase_pinyin_map;
    std::unordered_map<char32_t, std::vector<std::string>> single_pinyin_map;
    std::unordered_map<std::string, std::pair<std::string, std::string>> pinyin_to_symbol_map;

    std::unordered_set<std::u32string> must_erhua;
    std::unordered_set<std::u32string> not_erhua;
    std::unordered_set<std::string> symbols;

    // English G2P: word -> list of ARPAbet phonemes
    // Key is lowercase word, value is space-separated ARPAbet phones (e.g. ["W", "EH1", "L", "K", "AH0", "M"])
    std::unordered_map<std::string, std::vector<std::string>> english_dict;

    void load_pinyin_dicts(const std::string& dict_dir);
    void load_opencpop_strict(const std::string& dict_dir);
    void load_symbols();
    void load_english_dict(const std::string& dict_dir);

    std::pair<std::vector<std::string>, std::vector<std::string>> get_initials_finals(const std::string& word) const;
    std::pair<std::vector<std::string>, std::vector<std::string>> merge_erhua(
        const std::vector<std::string>& initials,
        const std::vector<std::string>& finals,
        const std::string& word,
        const std::string& pos) const;

    PhonemizerResult process_en(const std::string& text);

public:
    Phonemizer(const std::string& dict_dir);
    ~Phonemizer();

    PhonemizerResult process(const std::string& text, const std::string& lang = "zh");
};

} // namespace phonemizer
