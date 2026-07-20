#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif
#include <omp.h>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

template <typename T>
static void compute_instance_norm(const T* input, const float* gamma, const float* beta, T* output, int64_t length,
                                  int64_t channels, int64_t groups, float eps, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t group = 0; group < groups; ++group) {
        const T* input_group = input + group * length;
        T* output_group = output + group * length;
        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (int64_t index = 0; index < length; ++index) {
            sum += read_val(input_group + index);
        }
        const float mean = sum / static_cast<float>(length);
        float squared_sum = 0.0f;
#pragma omp simd reduction(+ : squared_sum)
        for (int64_t index = 0; index < length; ++index) {
            const float difference = read_val(input_group + index) - mean;
            squared_sum += difference * difference;
        }
        const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(length) + eps);
        const int64_t channel = group % channels;
        const float scale = gamma ? gamma[channel] : 1.0f;
        const float shift = beta ? beta[channel] : 0.0f;
#pragma omp simd
        for (int64_t index = 0; index < length; ++index) {
            write_val(output_group + index, (read_val(input_group + index) - mean) * inverse_std * scale + shift);
        }
    }
}

#if defined(__AVX2__) || defined(_M_AVX2)
static float horizontal_sum(__m256 values) {
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static __m256 load_f16x8(const ggml_fp16_t* source) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(source)));
}

static void compute_instance_norm_f16_avx2(const ggml_fp16_t* input, const float* gamma, const float* beta,
                                           ggml_fp16_t* output, int64_t length, int64_t channels, int64_t groups,
                                           float eps, int threads) {
    const int64_t vector_length = length & ~int64_t(7);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t group = 0; group < groups; ++group) {
        const ggml_fp16_t* input_group = input + group * length;
        __m256 sum_vector = _mm256_setzero_ps();
        int64_t index = 0;
        for (; index < vector_length; index += 8) {
            sum_vector = _mm256_add_ps(sum_vector, load_f16x8(input_group + index));
        }
        float sum = horizontal_sum(sum_vector);
        for (; index < length; ++index)
            sum += ggml_fp16_to_fp32(input_group[index]);
        const float mean = sum / static_cast<float>(length);
        const __m256 mean_vector = _mm256_set1_ps(mean);
        __m256 squared_sum_vector = _mm256_setzero_ps();
        index = 0;
        for (; index < vector_length; index += 8) {
            const __m256 difference = _mm256_sub_ps(load_f16x8(input_group + index), mean_vector);
            squared_sum_vector = _mm256_add_ps(squared_sum_vector, _mm256_mul_ps(difference, difference));
        }
        float squared_sum = horizontal_sum(squared_sum_vector);
        for (; index < length; ++index) {
            const float difference = ggml_fp16_to_fp32(input_group[index]) - mean;
            squared_sum += difference * difference;
        }
        const int64_t channel = group % channels;
        const __m256 inverse_std = _mm256_set1_ps(1.0f / std::sqrt(squared_sum / static_cast<float>(length) + eps));
        const __m256 scale = _mm256_set1_ps(gamma ? gamma[channel] : 1.0f);
        const __m256 shift = _mm256_set1_ps(beta ? beta[channel] : 0.0f);
        ggml_fp16_t* output_group = output + group * length;
        index = 0;
        for (; index < vector_length; index += 8) {
            const __m256 normalized =
                _mm256_mul_ps(_mm256_sub_ps(load_f16x8(input_group + index), mean_vector), inverse_std);
            const __m256 result = _mm256_add_ps(_mm256_mul_ps(normalized, scale), shift);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output_group + index),
                             _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
        const float inverse_std_scalar = _mm256_cvtss_f32(inverse_std);
        const float scale_scalar = _mm256_cvtss_f32(scale);
        const float shift_scalar = _mm256_cvtss_f32(shift);
        for (; index < length; ++index) {
            output_group[index] = ggml_fp32_to_fp16(
                (ggml_fp16_to_fp32(input_group[index]) - mean) * inverse_std_scalar * scale_scalar + shift_scalar);
        }
    }
}
#endif

bool ops_cpu_op_instance_norm(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);
    if ((int)node->op != GGML_OP_OPS_VIRT_INSTANCE_NORM)
        return false;

    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params))
        return false;

    struct ggml_tensor* x = params.x;
    struct ggml_tensor* gamma = params.gamma;
    struct ggml_tensor* beta = params.beta;
    struct ggml_tensor* dst = node;
    float eps = params.eps;

    const int64_t length = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t groups = ggml_nelements(x) / length;

    std::vector<float> gamma_f32;
    if (gamma) {
        if (gamma->type == GGML_TYPE_F16) {
            gamma_f32.resize(channels);
            const ggml_fp16_t* p = (const ggml_fp16_t*)gamma->data;
            ggml_fp16_to_fp32_row(p, gamma_f32.data(), channels);
        }
    }
    const float* gamma_ptr =
        gamma ? (gamma->type == GGML_TYPE_F32 ? (const float*)gamma->data : gamma_f32.data()) : nullptr;

    std::vector<float> beta_f32;
    if (beta) {
        if (beta->type == GGML_TYPE_F16) {
            beta_f32.resize(channels);
            const ggml_fp16_t* p = (const ggml_fp16_t*)beta->data;
            ggml_fp16_to_fp32_row(p, beta_f32.data(), channels);
        }
    }
    const float* beta_ptr = beta ? (beta->type == GGML_TYPE_F32 ? (const float*)beta->data : beta_f32.data()) : nullptr;

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        compute_instance_norm((const float*)x->data, gamma_ptr, beta_ptr, (float*)dst->data, length, channels, groups,
                              eps, omp_threads);
    } else if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
#if defined(__AVX2__) || defined(_M_AVX2)
        compute_instance_norm_f16_avx2((const ggml_fp16_t*)x->data, gamma_ptr, beta_ptr, (ggml_fp16_t*)dst->data,
                                       length, channels, groups, eps, omp_threads);
#else
        compute_instance_norm((const ggml_fp16_t*)x->data, gamma_ptr, beta_ptr, (ggml_fp16_t*)dst->data, length,
                              channels, groups, eps, omp_threads);
#endif
    } else {
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
