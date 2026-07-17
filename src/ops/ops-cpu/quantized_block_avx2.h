#pragma once

#include "ggml.h"
#include "ggml-common.h"
#include <immintrin.h>

namespace ggml_ops_ext {
namespace cpu {

static inline float horizontal_sum_avx2(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static inline float dot_q8_0_block_avx2(const block_q8_0& block, const float* input) {
    __m256 sum = _mm256_setzero_ps();
    for (int offset = 0; offset < QK8_0; offset += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs + offset));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
        sum = _mm256_fmadd_ps(values, _mm256_loadu_ps(input + offset), sum);
    }
    return ggml_fp16_to_fp32(block.d) * horizontal_sum_avx2(sum);
}

static inline float dot_q4_0_block_avx2(const block_q4_0& block, const float* input) {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.qs));
    const __m128i mask = _mm_set1_epi8(0x0f);
    const __m128i low = _mm_and_si128(packed, mask);
    const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
    const __m256i offset8 = _mm256_set1_epi32(8);
    __m256 sum = _mm256_setzero_ps();
    for (int half = 0; half < 2; ++half) {
        const __m128i low8 = half ? _mm_srli_si128(low, 8) : low;
        const __m128i high8 = half ? _mm_srli_si128(high, 8) : high;
        const __m256 low_values = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(low8), offset8));
        const __m256 high_values = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(high8), offset8));
        sum = _mm256_fmadd_ps(low_values, _mm256_loadu_ps(input + half * 8), sum);
        sum = _mm256_fmadd_ps(high_values, _mm256_loadu_ps(input + 16 + half * 8), sum);
    }
    return ggml_fp16_to_fp32(block.d) * horizontal_sum_avx2(sum);
}

static inline void q4_k_group_params(const block_q4_K& block, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = block.scales[group] & 63;
        minimum = block.scales[group + 4] & 63;
    } else {
        scale = (block.scales[group + 4] & 0x0f) | ((block.scales[group - 4] >> 6) << 4);
        minimum = (block.scales[group + 4] >> 4) | ((block.scales[group] >> 6) << 4);
    }
}

static inline __m256 q4_k_values_avx2(const block_q4_K& block, int group, int offset) {
    const int group64 = group / 2;
    const __m128i bytes = _mm_loadl_epi64(
        reinterpret_cast<const __m128i*>(block.qs + group64 * 32 + offset));
    __m256i values = _mm256_cvtepu8_epi32(bytes);
    values = (group & 1) ? _mm256_srli_epi32(values, 4)
                         : _mm256_and_si256(values, _mm256_set1_epi32(0x0f));
    return _mm256_cvtepi32_ps(values);
}

static inline float dot_q4_k_block_avx2(const block_q4_K& block, const float* input) {
    const float d = ggml_fp16_to_fp32(block.data.data.d);
    const float dmin = ggml_fp16_to_fp32(block.data.data.dmin);
    __m256 sum = _mm256_setzero_ps();
    for (int group = 0; group < 8; ++group) {
        int scale;
        int minimum;
        q4_k_group_params(block, group, scale, minimum);
        const __m256 d8 = _mm256_set1_ps(d * scale);
        const __m256 min8 = _mm256_set1_ps(dmin * minimum);
        for (int offset = 0; offset < 32; offset += 8) {
            const __m256 weight = _mm256_sub_ps(
                _mm256_mul_ps(d8, q4_k_values_avx2(block, group, offset)), min8);
            sum = _mm256_fmadd_ps(weight, _mm256_loadu_ps(input + group * 32 + offset), sum);
        }
    }
    return horizontal_sum_avx2(sum);
}

struct quantized_32_microtile {
    alignas(32) float values[32];
    float d;
    float dmin;
    int scale;
    int minimum;
};

static inline void decode_quantized_32_avx2(
    ggml_type type, const void* block_data, int64_t block_offset, quantized_32_microtile& tile
) {
    if (type == GGML_TYPE_Q4_0) {
        const auto& block = *static_cast<const block_q4_0*>(block_data);
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(block.qs));
        const __m128i mask = _mm_set1_epi8(0x0f);
        const __m128i low = _mm_and_si128(packed, mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
        const __m256i offset8 = _mm256_set1_epi32(8);
        tile.d = ggml_fp16_to_fp32(block.d);
        tile.dmin = 0.0f;
        tile.scale = 1;
        tile.minimum = 0;
        for (int half = 0; half < 2; ++half) {
            const __m128i low8 = half ? _mm_srli_si128(low, 8) : low;
            const __m128i high8 = half ? _mm_srli_si128(high, 8) : high;
            const __m256 low_values = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_cvtepu8_epi32(low8), offset8));
            const __m256 high_values = _mm256_cvtepi32_ps(
                _mm256_sub_epi32(_mm256_cvtepu8_epi32(high8), offset8));
            _mm256_store_ps(tile.values + half * 8, low_values);
            _mm256_store_ps(tile.values + 16 + half * 8, high_values);
        }
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        const auto& block = *static_cast<const block_q8_0*>(block_data);
        tile.d = ggml_fp16_to_fp32(block.d);
        tile.dmin = 0.0f;
        tile.scale = 1;
        tile.minimum = 0;
        for (int offset = 0; offset < QK8_0; offset += 8) {
            const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.qs + offset));
            const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
            _mm256_store_ps(tile.values + offset, values);
        }
        return;
    }
    if (type == GGML_TYPE_Q4_K) {
        const auto& block = *static_cast<const block_q4_K*>(block_data);
        const int group = static_cast<int>(block_offset / 32);
        tile.d = ggml_fp16_to_fp32(block.data.data.d);
        tile.dmin = ggml_fp16_to_fp32(block.data.data.dmin);
        q4_k_group_params(block, group, tile.scale, tile.minimum);
        for (int offset = 0; offset < 32; offset += 8) {
            _mm256_store_ps(tile.values + offset, q4_k_values_avx2(block, group, offset));
        }
    }
}

static inline void axpy_quantized_32_microtile_avx2(
    ggml_type type, const quantized_32_microtile& tile, float input, float* output
) {
    if (type == GGML_TYPE_Q4_K) {
        const __m256 factor = _mm256_set1_ps((input * tile.d) * tile.scale);
        const __m256 minimum = _mm256_set1_ps((input * tile.dmin) * tile.minimum);
        for (int offset = 0; offset < 32; offset += 8) {
            const __m256 contribution = _mm256_sub_ps(
                _mm256_mul_ps(factor, _mm256_load_ps(tile.values + offset)), minimum);
            float* out = output + offset;
            _mm256_storeu_ps(out, _mm256_add_ps(_mm256_loadu_ps(out), contribution));
        }
        return;
    }
    const __m256 factor = _mm256_set1_ps(input * tile.d);
    for (int offset = 0; offset < 32; offset += 8) {
        float* out = output + offset;
        _mm256_storeu_ps(out, _mm256_fmadd_ps(
            factor, _mm256_load_ps(tile.values + offset), _mm256_loadu_ps(out)));
    }
}

static inline float dot_quantized_row_avx2(
    ggml_type type, const void* row_data, const float* input, int64_t elements
) {
    float sum = 0.0f;
    if (type == GGML_TYPE_Q4_0) {
        const auto* blocks = static_cast<const block_q4_0*>(row_data);
        for (int64_t i = 0; i < elements / QK4_0; ++i) sum += dot_q4_0_block_avx2(blocks[i], input + i * QK4_0);
        return sum;
    }
    if (type == GGML_TYPE_Q8_0) {
        const auto* blocks = static_cast<const block_q8_0*>(row_data);
        for (int64_t i = 0; i < elements / QK8_0; ++i) sum += dot_q8_0_block_avx2(blocks[i], input + i * QK8_0);
        return sum;
    }
    if (type == GGML_TYPE_Q4_K) {
        const auto* blocks = static_cast<const block_q4_K*>(row_data);
        for (int64_t i = 0; i < elements / QK_K; ++i) sum += dot_q4_k_block_avx2(blocks[i], input + i * QK_K);
        return sum;
    }
    return 0.0f;
}

} // namespace cpu
} // namespace ggml_ops_ext
