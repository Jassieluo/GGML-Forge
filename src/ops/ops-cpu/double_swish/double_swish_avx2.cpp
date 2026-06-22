#include "ggml-cpu/vec.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX2__)
inline static __m256 ggml_v_double_swish_avx2(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 max_val = _mm256_set1_ps(20.0f);
    const __m256 min_val = _mm256_set1_ps(-20.0f);
    
    // x_minus_1 = x - 1
    __m256 x_minus_1 = _mm256_sub_ps(x, one);
    // neg_x_minus_1 = -(x - 1)
    __m256 neg_x_minus_1 = _mm256_sub_ps(_mm256_setzero_ps(), x_minus_1);
    
    // Clamp to prevent overflow/underflow
    __m256 clamped = _mm256_min_ps(_mm256_max_ps(neg_x_minus_1, min_val), max_val);
    
    // expf(clamped)
    __m256 e_neg = ggml_v_expf(clamped);
    
    // 1 + expf(clamped)
    __m256 den = _mm256_add_ps(one, e_neg);
    
    // x / (1 + expf(clamped))
    return _mm256_div_ps(x, den);
}
#endif

void ggml_vec_ext_double_swish_f32_avx2(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_double_swish_avx2(_mm256_loadu_ps(x + i)));
    }
#endif
    // Fallback scalar
    for (; i < n; ++i) {
        float val = x[i];
        float neg_xm1 = -(val - 1.0f);
        float clamped = std::max(-20.0f, std::min(neg_xm1, 20.0f));
        y[i] = val / (1.0f + std::exp(clamped));
    }
}

} // namespace cpu
} // namespace ggml_ops_ext
