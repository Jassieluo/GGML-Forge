#include "ggml-cpu/vec.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX512F__)
inline static __m512 ggml_v_mish_avx512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 max_val = _mm512_set1_ps(20.0f);
    const __m512 min_val = _mm512_set1_ps(-20.0f);
    
    // Clamp to prevent overflow/underflow
    __m512 x_clamped = _mm512_min_ps(_mm512_max_ps(x, min_val), max_val);
    
    // exp(x)
    __m512 ex = ggml_v_expf(x_clamped);
    
    // (ex + 1)^2
    __m512 ex1 = _mm512_add_ps(ex, one);
    __m512 ex1_sq = _mm512_mul_ps(ex1, ex1);
    
    // (ex1_sq - 1) / (ex1_sq + 1)
    __m512 num = _mm512_sub_ps(ex1_sq, one);
    __m512 den = _mm512_add_ps(ex1_sq, one);
    __m512 mul = _mm512_div_ps(num, den);
    
    return _mm512_mul_ps(x, mul);
}
#endif

void ggml_vec_ext_mish_f32_avx512(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_mish_avx512(_mm512_loadu_ps(x + i)));
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
