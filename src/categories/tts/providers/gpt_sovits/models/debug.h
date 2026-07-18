#pragma once

#include <cstdlib>
#include <string>

namespace gpt_sovits {

// Provider-local diagnostics shared by GPT-SoVITS model components.

inline bool is_debug_enabled() {
    const char* value = std::getenv("GPT_SOVITS_DEBUG");
    if (!value) return false;
    std::string setting(value);
    const size_t first = setting.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return false;
    const size_t last = setting.find_last_not_of(" \t\r\n");
    setting = setting.substr(first, last - first + 1);
    return setting != "0" && setting != "false" && setting != "off";
}

} // namespace gpt_sovits

#ifndef GPT_SOVITS_DEBUG_ENABLED
#define GPT_SOVITS_DEBUG_ENABLED() (gpt_sovits::is_debug_enabled())
#endif
