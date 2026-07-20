#include "ops/cpu.h"
#include "ops/ops.h"

#include <cmath>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif
#include <omp.h>

namespace ggml_ops_ext::cpu {
namespace {

void compute_glu_f32(const float* input, float* output, int64_t width, int64_t rows, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const float* value = input + row * 2 * width;
        const float* gate = value + width;
        float* output_row = output + row * width;
#pragma omp simd
        for (int64_t column = 0; column < width; ++column) {
            output_row[column] = value[column] / (1.0f + std::exp(-gate[column]));
        }
    }
}

void compute_glu_f16(const ggml_fp16_t* input, ggml_fp16_t* output, int64_t width, int64_t rows, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const ggml_fp16_t* value = input + row * 2 * width;
        const ggml_fp16_t* gate = value + width;
        ggml_fp16_t* output_row = output + row * width;
        int64_t column = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
        alignas(32) float value_f32[8];
        alignas(32) float gate_f32[8];
        alignas(32) float result_f32[8];
        const int64_t vector_width = width & ~int64_t(7);
        for (; column < vector_width; column += 8) {
            _mm256_store_ps(value_f32,
                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(value + column))));
            _mm256_store_ps(gate_f32,
                            _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gate + column))));
#pragma omp simd
            for (int lane = 0; lane < 8; ++lane) {
                result_f32[lane] = value_f32[lane] / (1.0f + std::exp(-gate_f32[lane]));
            }
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(output_row + column),
                _mm256_cvtps_ph(_mm256_load_ps(result_f32), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }
#endif
        for (; column < width; ++column) {
            const float value_f32 = ggml_fp16_to_fp32(value[column]);
            const float gate_f32 = ggml_fp16_to_fp32(gate[column]);
            output_row[column] = ggml_fp32_to_fp16(value_f32 / (1.0f + std::exp(-gate_f32)));
        }
    }
}

} // namespace

bool ops_cpu_op_glu(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0] || static_cast<int>(node->op) != GGML_OP_OPS_VIRT_GLU) {
        return false;
    }
    const int64_t width = node->ne[0];
    const int64_t rows = ggml_nelements(node) / width;
    const int threads = backend_thread_count(backend);
    if (node->type == GGML_TYPE_F32) {
        compute_glu_f32(static_cast<const float*>(node->src[0]->data), static_cast<float*>(node->data), width, rows,
                        threads);
        return true;
    }
    if (node->type == GGML_TYPE_F16) {
        compute_glu_f16(static_cast<const ggml_fp16_t*>(node->src[0]->data), static_cast<ggml_fp16_t*>(node->data),
                        width, rows, threads);
        return true;
    }
    return false;
}

} // namespace ggml_ops_ext::cpu
