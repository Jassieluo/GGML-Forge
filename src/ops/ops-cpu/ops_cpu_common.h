#pragma once
#include "ggml.h"
#include <type_traits>

template <typename T>
inline float read_val(const T* ptr) {
    if constexpr (std::is_same_v<T, ggml_fp16_t>) {
        return ggml_fp16_to_fp32(*ptr);
    } else {
        return (float)(*ptr);
    }
}

template <typename T>
inline void write_val(T* ptr, float val) {
    if constexpr (std::is_same_v<T, ggml_fp16_t>) {
        *ptr = ggml_fp32_to_fp16(val);
    } else {
        *ptr = (T)val;
    }
}
