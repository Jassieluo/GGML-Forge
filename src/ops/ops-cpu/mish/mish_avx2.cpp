#include "ggml-cpu/vec.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX2__)
inline static __m256 ggml_v_mish_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 max_val = _mm256_set1_ps(20.0f);
    const __m256 min_val = _mm256_set1_ps(-20.0f);
    
    // Clamp to prevent overflow/underflow
    __m256 x_clamped = _mm256_min_ps(_mm256_max_ps(x, min_val), max_val);
    
    // exp(x)
    __m256 ex = ggml_v_expf(x_clamped);
    
    // (ex + 1)^2
    __m256 ex1 = _mm256_add_ps(ex, one);
    __m256 ex1_sq = _mm256_mul_ps(ex1, ex1);
    
    // (ex1_sq - 1) / (ex1_sq + 1)
    __m256 num = _mm256_sub_ps(ex1_sq, one);
    __m256 den = _mm256_add_ps(ex1_sq, one);
    __m256 mul = _mm256_div_ps(num, den);
    
    return _mm256_mul_ps(x, mul);
}
#endif

void ggml_vec_ext_mish_f32_avx2(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_mish_avx2(_mm256_loadu_ps(x + i)));
    }
#endif
    // Fallback scalar
    for (; i < n; ++i) {
        float val = x[i];
        float clamped = std::max(-20.0f, std::min(val, 20.0f));
        float ex = std::exp(clamped);
        float ex1 = ex + 1.0f;
        float ex1_sq = ex1 * ex1;
        y[i] = val * (ex1_sq - 1.0f) / (ex1_sq + 1.0f);
    }
}

} // namespace cpu
} // namespace ggml_ops_ext
