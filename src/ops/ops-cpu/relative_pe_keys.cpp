#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cstring>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

namespace ggml_ops_ext::cpu {
namespace {

template <typename TQ, typename TR>
float dot_product(const TQ *query, const TR *relative, int64_t length) {
  float sum = 0.0f;
  int64_t index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
  __m256 accumulator = _mm256_setzero_ps();
  const int64_t vector_length = length & ~int64_t(7);
  for (; index < vector_length; index += 8) {
    __m256 query_values;
    __m256 relative_values;
    if constexpr (std::is_same_v<TQ, float>) {
      query_values = _mm256_loadu_ps(query + index);
    } else {
      query_values = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(query + index)));
    }
    if constexpr (std::is_same_v<TR, float>) {
      relative_values = _mm256_loadu_ps(relative + index);
    } else {
      relative_values = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(relative + index)));
    }
    accumulator = _mm256_fmadd_ps(query_values, relative_values, accumulator);
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, accumulator);
  for (float lane : lanes)
    sum += lane;
#endif
  for (; index < length; ++index)
    sum += read_val(query + index) * read_val(relative + index);
  return sum;
}

// emb_head_stride is the per-head row stride into the embedding table: 0 when
// the table is shared across heads (emb ne[2] == 1), relative_length otherwise.
template <typename TQ, typename TR>
void compute_relative_keys(const TQ *query, const TR *relative, TQ *output,
                           int64_t width, int64_t tokens, int64_t heads,
                           int64_t emb_head_stride, float scale, int32_t window,
                           int threads) {
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
  for (int64_t head = 0; head < heads; ++head) {
    for (int64_t token = 0; token < tokens; ++token) {
      TQ *output_row = output + (head * tokens + token) * tokens;
      std::memset(output_row, 0, static_cast<size_t>(tokens) * sizeof(TQ));
      const TQ *query_row = query + (head * tokens + token) * width;
      const int64_t key_begin = std::max<int64_t>(0, token - window);
      const int64_t key_end = std::min<int64_t>(tokens, token + window + 1);
      for (int64_t key = key_begin; key < key_end; ++key) {
        const int64_t relative_index = key - token + window;
        const TR *relative_row =
            relative + (head * emb_head_stride + relative_index) * width;
        write_val(output_row + key,
                  dot_product(query_row, relative_row, width) * scale);
      }
    }
  }
}

template <typename TQ>
bool dispatch_relative_type(const TQ *query, const ggml_tensor *relative,
                            TQ *output, int64_t width, int64_t tokens,
                            int64_t heads, int64_t emb_head_stride, float scale,
                            int32_t window, int threads) {
  if (relative->type == GGML_TYPE_F32) {
    compute_relative_keys(query, static_cast<const float *>(relative->data),
                          output, width, tokens, heads, emb_head_stride, scale,
                          window, threads);
    return true;
  }
  if (relative->type == GGML_TYPE_F16) {
    compute_relative_keys(
        query, static_cast<const ggml_fp16_t *>(relative->data), output, width,
        tokens, heads, emb_head_stride, scale, window, threads);
    return true;
  }
  return false;
}

} // namespace

bool ops_cpu_op_relative_pe_keys(ggml_backend_t backend, ggml_tensor *node) {
  ops_relative_pe_keys_params params;
  if (!ops_extract_relative_pe_keys_params(node, params))
    return false;
  const ggml_tensor *query = params.q;
  const int64_t width = query->ne[0];
  const int64_t tokens = query->ne[1];
  const int64_t heads = query->ne[2];
  const int64_t emb_head_stride =
      params.emb_rel_k->ne[2] == 1 ? 0 : params.emb_rel_k->ne[1];
  const int threads = backend_thread_count(backend);
  if (query->type == GGML_TYPE_F32) {
    return dispatch_relative_type(
        static_cast<const float *>(query->data), params.emb_rel_k,
        static_cast<float *>(node->data), width, tokens, heads, emb_head_stride,
        params.scale, params.window_size, threads);
  }
  if (query->type == GGML_TYPE_F16) {
    return dispatch_relative_type(
        static_cast<const ggml_fp16_t *>(query->data), params.emb_rel_k,
        static_cast<ggml_fp16_t *>(node->data), width, tokens, heads,
        emb_head_stride, params.scale, params.window_size, threads);
  }
  return false;
}

} // namespace ggml_ops_ext::cpu
