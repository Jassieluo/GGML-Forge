#include "ggml-cpu/vec.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX512F__)
inline static __m512 ggml_v_double_swish_avx512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 max_val = _mm512_set1_ps(20.0f);
    const __m512 min_val = _mm512_set1_ps(-20.0f);
    
    // x_minus_1 = x - 1
    __m512 x_minus_1 = _mm512_sub_ps(x, one);
    // neg_x_minus_1 = -(x - 1)
    __m512 neg_x_minus_1 = _mm512_sub_ps(_mm512_setzero_ps(), x_minus_1);
    
    // Clamp to prevent overflow/underflow
    __m512 clamped = _mm512_min_ps(_mm512_max_ps(neg_x_minus_1, min_val), max_val);
    
    // expf(clamped)
    __m512 e_neg = ggml_v_expf(clamped);
    
    // 1 + expf(clamped)
    __m512 den = _mm512_add_ps(one, e_neg);
    
    // x / (1 + expf(clamped))
    return _mm512_div_ps(x, den);
}
#endif

void ggml_vec_ext_double_swish_f32_avx512(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_double_swish_avx512(_mm512_loadu_ps(x + i)));
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
