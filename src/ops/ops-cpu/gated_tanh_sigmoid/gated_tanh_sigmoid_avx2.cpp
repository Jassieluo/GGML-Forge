#include "ggml-cpu/vec.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX2__)
inline static __m256 ggml_v_gated_tanh_sigmoid_avx2(__m256 xa, __m256 xb) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 max_val = _mm256_set1_ps(20.0f);
    const __m256 min_val = _mm256_set1_ps(-20.0f);

    // 2 * xa clamped
    __m256 xa2 = _mm256_mul_ps(xa, _mm256_set1_ps(2.0f));
    __m256 xa2_clamped = _mm256_min_ps(_mm256_max_ps(xa2, min_val), max_val);

    // -xb clamped
    __m256 neg_xb = _mm256_sub_ps(zero, xb);
    __m256 neg_xb_clamped = _mm256_min_ps(_mm256_max_ps(neg_xb, min_val), max_val);

    // exp(2*xa) and exp(-xb)
    __m256 e_2a = ggml_v_expf(xa2_clamped);
    __m256 e_nb = ggml_v_expf(neg_xb_clamped);

    // tanh(xa) = (e_2a - 1) / (e_2a + 1)
    __m256 tanh_val = _mm256_div_ps(_mm256_sub_ps(e_2a, one), _mm256_add_ps(e_2a, one));

    // sigmoid(xb) = 1 / (1 + e_nb)
    __m256 sig_val = _mm256_div_ps(one, _mm256_add_ps(one, e_nb));

    return _mm256_mul_ps(tanh_val, sig_val);
}
#endif

void ggml_vec_ext_gated_tanh_sigmoid_f32_avx2(const int n, float * y, const float * xa, const float * xb) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_gated_tanh_sigmoid_avx2(_mm256_loadu_ps(xa + i), _mm256_loadu_ps(xb + i)));
    }
#endif
    // Fallback scalar
    for (; i < n; ++i) {
        float val_a = xa[i];
        float val_b = xb[i];
        float ta = std::tanh(val_a);
        float sig = 1.0f / (1.0f + std::exp(-val_b));
        y[i] = ta * sig;
    }
}

} // namespace cpu
} // namespace ggml_ops_ext
