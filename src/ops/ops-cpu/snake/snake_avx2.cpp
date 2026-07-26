// AVX2 Snake / SnakeBeta: y = x + sin^2(alpha*x)/beta.
// The vocoder hot path spends most of its time in std::sin, which does not
// auto-vectorize, so the sine is evaluated with a vectorized Cody-Waite range
// reduction plus the classic cephes minimax polynomials (~1e-7 relative).
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace ggml_ops_ext {
namespace cpu {

namespace {

// Cody-Waite splitting of pi/2 — three terms keep the reduction exact for the
// magnitudes activations reach; larger inputs fall back to scalar sin.
constexpr float kPiOver2A = 1.5707962512969971f;
constexpr float kPiOver2B = 7.549789948768648e-8f;
constexpr float kPiOver2C = 5.390302529957765e-15f;
constexpr float kTwoOverPi = 0.63661977236758134f;
constexpr float kReductionLimit = 1.0e5f;

inline __m256 sin_avx2(__m256 x) {
    // n = round(x * 2/pi), the quadrant index.
    __m256 quadrant_f = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kTwoOverPi)),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256i quadrant = _mm256_cvtps_epi32(quadrant_f);

    __m256 r = _mm256_fnmadd_ps(quadrant_f, _mm256_set1_ps(kPiOver2A), x);
    r = _mm256_fnmadd_ps(quadrant_f, _mm256_set1_ps(kPiOver2B), r);
    r = _mm256_fnmadd_ps(quadrant_f, _mm256_set1_ps(kPiOver2C), r);

    const __m256 z = _mm256_mul_ps(r, r);

    // sin(r) on [-pi/4, pi/4]
    __m256 sin_poly = _mm256_set1_ps(-1.9515295891e-4f);
    sin_poly = _mm256_fmadd_ps(sin_poly, z, _mm256_set1_ps(8.3321608736e-3f));
    sin_poly = _mm256_fmadd_ps(sin_poly, z, _mm256_set1_ps(-1.6666654611e-1f));
    sin_poly = _mm256_fmadd_ps(_mm256_mul_ps(sin_poly, z), r, r);

    // cos(r) on [-pi/4, pi/4]
    __m256 cos_poly = _mm256_set1_ps(2.443315711809948e-5f);
    cos_poly = _mm256_fmadd_ps(cos_poly, z, _mm256_set1_ps(-1.388731625493765e-3f));
    cos_poly = _mm256_fmadd_ps(cos_poly, z, _mm256_set1_ps(4.166664568298827e-2f));
    cos_poly = _mm256_mul_ps(_mm256_mul_ps(cos_poly, z), z);
    cos_poly = _mm256_add_ps(cos_poly, _mm256_fnmadd_ps(_mm256_set1_ps(0.5f), z, _mm256_set1_ps(1.0f)));

    // Odd quadrants swap sin and cos; quadrants 2 and 3 negate.
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i two = _mm256_set1_epi32(2);
    const __m256i swap_mask = _mm256_cmpeq_epi32(_mm256_and_si256(quadrant, one), one);
    __m256 result = _mm256_blendv_ps(sin_poly, cos_poly, _mm256_castsi256_ps(swap_mask));
    const __m256i negate_mask = _mm256_cmpeq_epi32(_mm256_and_si256(quadrant, two), two);
    const __m256 sign_bit =
        _mm256_and_ps(_mm256_castsi256_ps(negate_mask), _mm256_set1_ps(-0.0f));
    return _mm256_xor_ps(result, sign_bit);
}

// True when every lane is small enough for the three-term reduction.
inline bool reduction_safe(__m256 values) {
    const __m256 magnitude =
        _mm256_and_ps(values, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    const __m256 over = _mm256_cmp_ps(magnitude, _mm256_set1_ps(kReductionLimit), _CMP_GT_OQ);
    return _mm256_movemask_ps(over) == 0;
}

} // namespace

void ggml_vec_ext_snake_f32_avx2(const int n, float* y, const float* x, const float alpha) {
    const float inv_alpha = 1.0f / alpha;
    const __m256 alpha_v = _mm256_set1_ps(alpha);
    const __m256 inv_alpha_v = _mm256_set1_ps(inv_alpha);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        const __m256 value = _mm256_loadu_ps(x + i);
        const __m256 scaled = _mm256_mul_ps(value, alpha_v);
        if (!reduction_safe(scaled)) break;
        const __m256 sine = sin_avx2(scaled);
        _mm256_storeu_ps(y + i,
                         _mm256_fmadd_ps(_mm256_mul_ps(sine, sine), inv_alpha_v, value));
    }
    for (; i < n; ++i) {
        const float value = x[i];
        const float sine = std::sin(alpha * value);
        y[i] = value + sine * sine * inv_alpha;
    }
}

void ggml_vec_ext_snake_beta_f32_avx2(const int n, float* y, const float* x, const float alpha,
                                      const float beta) {
    const float inv_beta = 1.0f / beta;
    const __m256 alpha_v = _mm256_set1_ps(alpha);
    const __m256 inv_beta_v = _mm256_set1_ps(inv_beta);

    int i = 0;
    for (; i <= n - 8; i += 8) {
        const __m256 value = _mm256_loadu_ps(x + i);
        const __m256 scaled = _mm256_mul_ps(value, alpha_v);
        if (!reduction_safe(scaled)) break;
        const __m256 sine = sin_avx2(scaled);
        _mm256_storeu_ps(y + i,
                         _mm256_fmadd_ps(_mm256_mul_ps(sine, sine), inv_beta_v, value));
    }
    for (; i < n; ++i) {
        const float value = x[i];
        const float sine = std::sin(alpha * value);
        y[i] = value + sine * sine * inv_beta;
    }
}

} // namespace cpu
} // namespace ggml_ops_ext
