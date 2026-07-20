#include "ops/cpu.h"
#include "ops/ops.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

void convert_f16_to_f32(const ggml_fp16_t *source, float *destination,
                        int64_t width) {
  int64_t index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
  const int64_t vector_width = width & ~int64_t(7);
  for (; index < vector_width; index += 8) {
    _mm256_storeu_ps(destination + index,
                     _mm256_cvtph_ps(_mm_loadu_si128(
                         reinterpret_cast<const __m128i *>(source + index))));
  }
#endif
  for (; index < width; ++index)
    destination[index] = ggml_fp16_to_fp32(source[index]);
}

void convert_f32_to_f16(const float *source, ggml_fp16_t *destination,
                        int64_t width) {
  int64_t index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
  const int64_t vector_width = width & ~int64_t(7);
  for (; index < vector_width; index += 8) {
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(destination + index),
        _mm256_cvtps_ph(_mm256_loadu_ps(source + index),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
#endif
  for (; index < width; ++index)
    destination[index] = ggml_fp32_to_fp16(source[index]);
}

float maximum_absolute(const float *values) {
#if defined(__AVX2__) || defined(_M_AVX2)
  const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 maximum = _mm256_setzero_ps();
  for (int index = 0; index < 32; index += 8) {
    maximum = _mm256_max_ps(
        maximum, _mm256_and_ps(_mm256_loadu_ps(values + index), sign_mask));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, maximum);
  float result = 0.0f;
  for (float lane : lanes)
    result = std::max(result, lane);
  return result;
#else
  float result = 0.0f;
  for (int index = 0; index < 32; ++index)
    result = std::max(result, std::abs(values[index]));
  return result;
#endif
}

void quantize_q8_0(const float *source, block_q8_0 *destination,
                   int64_t width) {
  for (int64_t block = 0; block < width / QK8_0; ++block) {
    const float *values = source + block * QK8_0;
    const float delta = maximum_absolute(values) / 127.0f;
    destination[block].d = ggml_fp32_to_fp16(delta);
    const float inverse_delta = delta == 0.0f ? 0.0f : 1.0f / delta;
    for (int index = 0; index < QK8_0; ++index) {
      destination[block].qs[index] =
          static_cast<int8_t>(std::lrint(values[index] * inverse_delta));
    }
  }
}

void quantize_q4_0(const float *source, block_q4_0 *destination,
                   int64_t width) {
  for (int64_t block = 0; block < width / QK4_0; ++block) {
    const float *values = source + block * QK4_0;
    const float delta = maximum_absolute(values) / 8.0f;
    destination[block].d = ggml_fp32_to_fp16(delta);
    const float inverse_delta = delta == 0.0f ? 0.0f : 1.0f / delta;
    for (int index = 0; index < QK4_0 / 2; ++index) {
      const int low = std::clamp(
          static_cast<int>(std::lrint(values[index] * inverse_delta)) + 8, 0,
          15);
      const int high =
          std::clamp(static_cast<int>(std::lrint(values[index + QK4_0 / 2] *
                                                 inverse_delta)) +
                         8,
                     0, 15);
      destination[block].qs[index] = static_cast<uint8_t>(low | (high << 4));
    }
  }
}

bool write_cache_rows(ggml_tensor *cache, const ggml_tensor *values,
                      int32_t position, int threads) {
  const int64_t width = cache->ne[0];
  const int64_t tokens = values->ne[1];
  const int64_t rows = values->ne[3] * values->ne[2] * tokens;
  if (position < 0 || position + tokens > cache->ne[1])
    return false;
  if (cache->type != GGML_TYPE_F32 && cache->type != GGML_TYPE_F16 &&
      cache->type != GGML_TYPE_Q8_0 && cache->type != GGML_TYPE_Q4_0) {
    return false;
  }

#pragma omp parallel num_threads(threads)
  {
    std::vector<float> converted;
    if (values->type == GGML_TYPE_F16 && cache->type != GGML_TYPE_F16) {
      converted.resize(static_cast<size_t>(width));
    }
#pragma omp for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
      const int64_t token = row % tokens;
      const int64_t head = (row / tokens) % values->ne[2];
      const int64_t batch = row / (tokens * values->ne[2]);
      const char *source = static_cast<const char *>(values->data) +
                           batch * values->nb[3] + head * values->nb[2] +
                           token * values->nb[1];
      char *destination = static_cast<char *>(cache->data) +
                          batch * cache->nb[3] + head * cache->nb[2] +
                          (position + token) * cache->nb[1];

      if (cache->type == values->type) {
        std::memcpy(destination, source, ggml_row_size(cache->type, width));
        continue;
      }
      if (cache->type == GGML_TYPE_F32) {
        convert_f16_to_f32(reinterpret_cast<const ggml_fp16_t *>(source),
                           reinterpret_cast<float *>(destination), width);
        continue;
      }
      if (cache->type == GGML_TYPE_F16) {
        convert_f32_to_f16(reinterpret_cast<const float *>(source),
                           reinterpret_cast<ggml_fp16_t *>(destination), width);
        continue;
      }

      const float *float_source = nullptr;
      if (values->type == GGML_TYPE_F32) {
        float_source = reinterpret_cast<const float *>(source);
      } else {
        convert_f16_to_f32(reinterpret_cast<const ggml_fp16_t *>(source),
                           converted.data(), width);
        float_source = converted.data();
      }
      if (cache->type == GGML_TYPE_Q8_0) {
        quantize_q8_0(float_source, reinterpret_cast<block_q8_0 *>(destination),
                      width);
      } else {
        quantize_q4_0(float_source, reinterpret_cast<block_q4_0 *>(destination),
                      width);
      }
    }
  }
  return true;
}

} // namespace

bool ops_cpu_op_kv_cache_update(ggml_backend_t backend, ggml_tensor *node) {
  ops_kv_cache_update_params params;
  if (!ops_extract_kv_cache_update_params(node, params))
    return false;
  const int32_t position = *static_cast<const int32_t *>(params.position->data);
  const int threads = backend_thread_count(backend);
  if (!write_cache_rows(params.cache_k, params.new_k, position, threads) ||
      !write_cache_rows(params.cache_v, params.new_v, position, threads)) {
    return false;
  }
  *static_cast<int32_t *>(node->data) =
      position + static_cast<int32_t>(params.new_k->ne[1]);
  return true;
}

} // namespace ggml_ops_ext::cpu
