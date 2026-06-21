#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace phonemizer {

inline std::u32string utf8_to_utf32(const std::string& utf8) {
    std::u32string utf32;
    utf32.reserve(utf8.size());
    size_t i = 0;
    while (i < utf8.size()) {
        uint32_t cp = 0;
        uint8_t b = utf8[i];
        if (b < 0x80) {
            cp = b;
            i += 1;
        } else if ((b & 0xE0) == 0xC0) {
            if (i + 1 >= utf8.size()) break;
            cp = (b & 0x1F) << 6;
            cp |= (utf8[i + 1] & 0x3F);
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            if (i + 2 >= utf8.size()) break;
            cp = (b & 0x0F) << 12;
            cp |= (utf8[i + 1] & 0x3F) << 6;
            cp |= (utf8[i + 2] & 0x3F);
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            if (i + 3 >= utf8.size()) break;
            cp = (b & 0x07) << 18;
            cp |= (utf8[i + 1] & 0x3F) << 12;
            cp |= (utf8[i + 2] & 0x3F) << 6;
            cp |= (utf8[i + 3] & 0x3F);
            i += 4;
        } else {
            // invalid UTF-8, skip byte
            i += 1;
            continue;
        }
        utf32.push_back(cp);
    }
    return utf32;
}

inline std::string utf32_to_utf8(const std::u32string& utf32) {
    std::string utf8;
    utf8.reserve(utf32.size() * 3);
    for (char32_t cp : utf32) {
        if (cp < 0x80) {
            utf8.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            utf8.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            utf8.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            utf8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x200000) {
            utf8.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
            utf8.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return utf8;
}

inline std::string utf32_char_to_utf8(char32_t cp) {
    std::u32string s(1, cp);
    return utf32_to_utf8(s);
}

inline bool is_cjk_char(char32_t cp) {
    // Standard CJK Ideographs ranges
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0x2A700 && cp <= 0x2B73F) return true;
    if (cp >= 0x2B740 && cp <= 0x2B81F) return true;
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return true;
    if (cp >= 0x2CEB0 && cp <= 0x2EBEF) return true;
    if (cp >= 0x30000 && cp <= 0x3134F) return true;
    if (cp >= 0x31350 && cp <= 0x323AF) return true;
    if (cp >= 0x2EBF0 && cp <= 0x2EE5F) return true;
    return false;
}

inline bool is_numeric(char32_t cp) {
    return cp >= '0' && cp <= '9';
}

inline bool is_punctuation(char32_t cp) {
    // Match common Chinese and English punctuations
    static const std::u32string puncs = U"!?,.:;!？，。：；、…“”‘’'-~～/()（）[]{}[【】]";
    return puncs.find(cp) != std::u32string::npos;
}

} // namespace phonemizer
