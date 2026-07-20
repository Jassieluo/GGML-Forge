#include "ggml.h"
#include "ops/cpu.h"
#include "ops/ops.h"

#include <cmath>
#include <cstring>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif
#include <omp.h>
#include <type_traits>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

#if defined(__AVX2__) || defined(_M_AVX2)
static float horizontal_sum(__m256 values) {
    const __m128 low = _mm256_castps256_ps128(values);
    const __m128 high = _mm256_extractf128_ps(values, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static __m256 load_f16x8(const ggml_fp16_t* source) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(source)));
}

static bool execute_layer_norm_f16_avx2(ggml_backend_t backend, const ggml_tensor* input, const ggml_tensor* gamma,
                                        const ggml_tensor* beta, ggml_tensor* output, float eps) {
    const int64_t width = input->ne[0];
    const int64_t rows = ggml_nelements(input) / width;
    const auto* source = static_cast<const ggml_fp16_t*>(input->data);
    const auto* scale = static_cast<const ggml_fp16_t*>(gamma->data);
    const auto* shift = static_cast<const ggml_fp16_t*>(beta->data);
    auto* destination = static_cast<ggml_fp16_t*>(output->data);
    const int64_t vector_width = width & ~int64_t(7);
    const int threads = backend_thread_count(backend);

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const ggml_fp16_t* input_row = source + row * width;
        __m256 sum_vector = _mm256_setzero_ps();
        int64_t column = 0;
        for (; column < vector_width; column += 8) {
            sum_vector = _mm256_add_ps(sum_vector, load_f16x8(input_row + column));
        }
        float sum = horizontal_sum(sum_vector);
        for (; column < width; ++column) {
            sum += ggml_fp16_to_fp32(input_row[column]);
        }
        const float mean = sum / static_cast<float>(width);
        const __m256 mean_vector = _mm256_set1_ps(mean);

        __m256 squared_sum_vector = _mm256_setzero_ps();
        column = 0;
        for (; column < vector_width; column += 8) {
            const __m256 difference = _mm256_sub_ps(load_f16x8(input_row + column), mean_vector);
            squared_sum_vector = _mm256_add_ps(squared_sum_vector, _mm256_mul_ps(difference, difference));
        }
        float squared_sum = horizontal_sum(squared_sum_vector);
        for (; column < width; ++column) {
            const float difference = ggml_fp16_to_fp32(input_row[column]) - mean;
            squared_sum += difference * difference;
        }
        const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(width) + eps);
        const __m256 inverse_std_vector = _mm256_set1_ps(inverse_std);

        ggml_fp16_t* output_row = destination + row * width;
        column = 0;
        for (; column < vector_width; column += 8) {
            const __m256 normalized =
                _mm256_mul_ps(_mm256_sub_ps(load_f16x8(input_row + column), mean_vector), inverse_std_vector);
            const __m256 result =
                _mm256_add_ps(_mm256_mul_ps(normalized, load_f16x8(scale + column)), load_f16x8(shift + column));
            const __m128i packed = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output_row + column), packed);
        }
        for (; column < width; ++column) {
            const float normalized = (ggml_fp16_to_fp32(input_row[column]) - mean) * inverse_std;
            output_row[column] =
                ggml_fp32_to_fp16(normalized * ggml_fp16_to_fp32(scale[column]) + ggml_fp16_to_fp32(shift[column]));
        }
    }
    return true;
}
#endif

template <typename T>
static bool execute_layer_norm(ggml_backend_t backend, const ggml_tensor* input, const ggml_tensor* gamma,
                               const ggml_tensor* beta, ggml_tensor* output, float eps) {
    const int64_t width = input->ne[0];
    const int64_t rows = ggml_nelements(input) / width;
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);

    std::vector<float> gamma_f32;
    std::vector<float> beta_f32;
    const float* scale = nullptr;
    const float* shift = nullptr;
    if constexpr (std::is_same_v<T, float>) {
        scale = static_cast<const float*>(gamma->data);
        shift = static_cast<const float*>(beta->data);
    } else {
        gamma_f32.resize(static_cast<size_t>(width));
        beta_f32.resize(static_cast<size_t>(width));
        const T* gamma_values = static_cast<const T*>(gamma->data);
        const T* beta_values = static_cast<const T*>(beta->data);
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(gamma_values), gamma_f32.data(), width);
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(beta_values), beta_f32.data(), width);
        scale = gamma_f32.data();
        shift = beta_f32.data();
    }

    const int threads = backend_thread_count(backend);
    std::vector<float> scratch;
    if constexpr (!std::is_same_v<T, float>) {
        scratch.resize(static_cast<size_t>(threads) * static_cast<size_t>(width) * 2);
    }
#pragma omp parallel num_threads(threads)
    {
        float* converted = nullptr;
        float* converted_output = nullptr;
        if constexpr (!std::is_same_v<T, float>) {
            converted = scratch.data() + static_cast<size_t>(omp_get_thread_num()) * static_cast<size_t>(width) * 2;
            converted_output = converted + width;
        }
#pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            const T* input_row = source + row * width;
            const float* values = nullptr;
            if constexpr (std::is_same_v<T, float>) {
                values = input_row;
            } else {
                ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(input_row), converted, width);
                values = converted;
            }

            float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
            for (int64_t column = 0; column < width; ++column) {
                sum += values[column];
            }
            const float mean = sum / static_cast<float>(width);
            float squared_sum = 0.0f;
#pragma omp simd reduction(+ : squared_sum)
            for (int64_t column = 0; column < width; ++column) {
                const float difference = values[column] - mean;
                squared_sum += difference * difference;
            }
            const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(width) + eps);
            T* output_row = destination + row * width;
            if constexpr (std::is_same_v<T, float>) {
#pragma omp simd
                for (int64_t column = 0; column < width; ++column) {
                    output_row[column] = (values[column] - mean) * inverse_std * scale[column] + shift[column];
                }
            } else {
#pragma omp simd
                for (int64_t column = 0; column < width; ++column) {
                    converted_output[column] = (values[column] - mean) * inverse_std * scale[column] + shift[column];
                }
                ggml_fp32_to_fp16_row(converted_output, reinterpret_cast<ggml_fp16_t*>(output_row), width);
            }
        }
    }
    return true;
}

bool ops_cpu_op_layer_norm(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0] || !node->src[1] || !node->src[2]) {
        return false;
    }
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(eps));
    if (node->type == GGML_TYPE_F32) {
        return execute_layer_norm<float>(backend, node->src[0], node->src[1], node->src[2], node, eps);
    }
    if (node->type == GGML_TYPE_F16) {
#if defined(__AVX2__) || defined(_M_AVX2)
        return execute_layer_norm_f16_avx2(backend, node->src[0], node->src[1], node->src[2], node, eps);
#else
        return execute_layer_norm<ggml_fp16_t>(backend, node->src[0], node->src[1], node->src[2], node, eps);
#endif
    }
    return false;
}

static size_t ada_parameter_row_offset(const ggml_tensor* parameter, int64_t row, const ggml_tensor* output) {
    const int64_t i1 = row % output->ne[1];
    const int64_t remainder = row / output->ne[1];
    const int64_t i2 = remainder % output->ne[2];
    const int64_t i3 = remainder / output->ne[2];
    const int64_t p1 = parameter->ne[2] > 1 ? i1 % parameter->ne[1]
                       : parameter->ne[1] > 1
                           ? (parameter->ne[1] == output->ne[2] ? i2 : i1 % parameter->ne[1])
                           : 0;
    const int64_t p2 = parameter->ne[2] > 1 ? i2 % parameter->ne[2] : 0;
    const int64_t p3 = parameter->ne[3] > 1 ? i3 % parameter->ne[3] : 0;
    return p1 * parameter->nb[1] + p2 * parameter->nb[2] + p3 * parameter->nb[3];
}

template <typename T>
static float ada_load(const T* value) {
    return static_cast<float>(*value);
}

template <>
[[maybe_unused]] float ada_load(const ggml_fp16_t* value) {
    return ggml_fp16_to_fp32(*value);
}

template <typename T>
static void ada_store(T* destination, float value) {
    *destination = static_cast<T>(value);
}

template <>
[[maybe_unused]] void ada_store(ggml_fp16_t* destination, float value) {
    *destination = ggml_fp32_to_fp16(value);
}

template <typename T>
static bool execute_ada_ln(ggml_backend_t backend, const ggml_tensor* input, const ggml_tensor* scale,
                           const ggml_tensor* shift, ggml_tensor* output, float eps) {
    const int64_t width = input->ne[0];
    const int64_t rows = ggml_nelements(input) / width;
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);
    const int threads = backend_thread_count(backend);

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const T* input_row = source + row * width;
        const auto* scale_row = reinterpret_cast<const T*>(
            static_cast<const char*>(scale->data) + ada_parameter_row_offset(scale, row, output));
        const auto* shift_row = reinterpret_cast<const T*>(
            static_cast<const char*>(shift->data) + ada_parameter_row_offset(shift, row, output));
        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (int64_t column = 0; column < width; ++column) {
            sum += ada_load(input_row + column);
        }
        const float mean = sum / static_cast<float>(width);
        float squared_sum = 0.0f;
#pragma omp simd reduction(+ : squared_sum)
        for (int64_t column = 0; column < width; ++column) {
            const float difference = ada_load(input_row + column) - mean;
            squared_sum += difference * difference;
        }
        const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(width) + eps);
        T* output_row = destination + row * width;
#pragma omp simd
        for (int64_t column = 0; column < width; ++column) {
            const float normalized = (ada_load(input_row + column) - mean) * inverse_std;
            ada_store(output_row + column,
                      normalized * (1.0f + ada_load(scale_row + column)) + ada_load(shift_row + column));
        }
    }
    return true;
}

#if defined(__AVX2__) || defined(_M_AVX2)
static bool execute_ada_ln_f16_avx2(ggml_backend_t backend, const ggml_tensor* input,
                                     const ggml_tensor* scale, const ggml_tensor* shift,
                                     ggml_tensor* output, float eps) {
    const int64_t width = input->ne[0];
    const int64_t rows = ggml_nelements(input) / width;
    const int64_t vector_width = width & ~int64_t(7);
    const auto* source = static_cast<const ggml_fp16_t*>(input->data);
    auto* destination = static_cast<ggml_fp16_t*>(output->data);
    const int threads = backend_thread_count(backend);

#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const ggml_fp16_t* input_row = source + row * width;
        const auto* scale_row = reinterpret_cast<const ggml_fp16_t*>(
            static_cast<const char*>(scale->data) + ada_parameter_row_offset(scale, row, output));
        const auto* shift_row = reinterpret_cast<const ggml_fp16_t*>(
            static_cast<const char*>(shift->data) + ada_parameter_row_offset(shift, row, output));
        __m256 sum_vector = _mm256_setzero_ps();
        int64_t column = 0;
        for (; column < vector_width; column += 8) {
            sum_vector = _mm256_add_ps(sum_vector, load_f16x8(input_row + column));
        }
        float sum = horizontal_sum(sum_vector);
        for (; column < width; ++column) {
            sum += ggml_fp16_to_fp32(input_row[column]);
        }
        const float mean = sum / static_cast<float>(width);
        const __m256 mean_vector = _mm256_set1_ps(mean);

        __m256 squared_sum_vector = _mm256_setzero_ps();
        column = 0;
        for (; column < vector_width; column += 8) {
            const __m256 difference = _mm256_sub_ps(load_f16x8(input_row + column), mean_vector);
            squared_sum_vector = _mm256_add_ps(squared_sum_vector, _mm256_mul_ps(difference, difference));
        }
        float squared_sum = horizontal_sum(squared_sum_vector);
        for (; column < width; ++column) {
            const float difference = ggml_fp16_to_fp32(input_row[column]) - mean;
            squared_sum += difference * difference;
        }
        const __m256 inverse_std_vector =
            _mm256_set1_ps(1.0f / std::sqrt(squared_sum / static_cast<float>(width) + eps));
        ggml_fp16_t* output_row = destination + row * width;
        column = 0;
        for (; column < vector_width; column += 8) {
            const __m256 normalized =
                _mm256_mul_ps(_mm256_sub_ps(load_f16x8(input_row + column), mean_vector), inverse_std_vector);
            const __m256 affine_scale = _mm256_add_ps(_mm256_set1_ps(1.0f), load_f16x8(scale_row + column));
            const __m256 result = _mm256_add_ps(_mm256_mul_ps(normalized, affine_scale),
                                                load_f16x8(shift_row + column));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output_row + column),
                             _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
        const float inverse_std = _mm256_cvtss_f32(inverse_std_vector);
        for (; column < width; ++column) {
            const float normalized = (ggml_fp16_to_fp32(input_row[column]) - mean) * inverse_std;
            output_row[column] = ggml_fp32_to_fp16(
                normalized * (1.0f + ggml_fp16_to_fp32(scale_row[column])) +
                ggml_fp16_to_fp32(shift_row[column]));
        }
    }
    return true;
}
#endif

bool ops_cpu_op_ada_ln(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0] || !node->src[1] || !node->src[2]) {
        return false;
    }
    float eps;
    std::memcpy(&eps, node->op_params, sizeof(eps));
    if (node->type == GGML_TYPE_F32) {
        return execute_ada_ln<float>(backend, node->src[0], node->src[1], node->src[2], node, eps);
    }
    if (node->type == GGML_TYPE_F16) {
#if defined(__AVX2__) || defined(_M_AVX2)
        return execute_ada_ln_f16_avx2(backend, node->src[0], node->src[1], node->src[2], node, eps);
#else
        return execute_ada_ln<ggml_fp16_t>(backend, node->src[0], node->src[1], node->src[2], node, eps);
#endif
    }
    return false;
}

} // namespace cpu
} // namespace ggml_ops_ext
