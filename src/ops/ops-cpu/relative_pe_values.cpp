#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <algorithm>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

template <typename TW, typename TD>
void compute_relative_values(const TW *weights, const float *relative,
                             TD *output, int64_t tokens, int64_t width,
                             int64_t heads, int64_t relative_length,
                             int32_t window, int threads) {
#pragma omp parallel num_threads(threads)
  {
    std::vector<float> accumulator(static_cast<size_t>(width));
#pragma omp for collapse(2) schedule(static)
    for (int64_t token = 0; token < tokens; ++token) {
      for (int64_t head = 0; head < heads; ++head) {
        std::fill(accumulator.begin(), accumulator.end(), 0.0f);
        const int64_t key_begin = std::max<int64_t>(0, token - window);
        const int64_t key_end = std::min<int64_t>(tokens, token + window + 1);
        for (int64_t key = key_begin; key < key_end; ++key) {
          const float weight =
              read_val(weights + (head * tokens + token) * tokens + key);
          const int64_t relative_index = key - token + window;
          const float *relative_row =
              relative + (head * relative_length + relative_index) * width;
#pragma omp simd
          for (int64_t channel = 0; channel < width; ++channel) {
            accumulator[channel] += weight * relative_row[channel];
          }
        }
        TD *output_row = output + (token * heads + head) * width;
#pragma omp simd
        for (int64_t channel = 0; channel < width; ++channel) {
          write_val(output_row + channel, accumulator[channel]);
        }
      }
    }
  }
}

template <typename TW>
bool dispatch_output(const TW *weights, const float *relative,
                     ggml_tensor *output, int64_t tokens, int64_t width,
                     int64_t heads, int64_t relative_length, int32_t window,
                     int threads) {
  if (output->type == GGML_TYPE_F32) {
    compute_relative_values(weights, relative,
                            static_cast<float *>(output->data), tokens, width,
                            heads, relative_length, window, threads);
    return true;
  }
  if (output->type == GGML_TYPE_F16) {
    compute_relative_values(weights, relative,
                            static_cast<ggml_fp16_t *>(output->data), tokens,
                            width, heads, relative_length, window, threads);
    return true;
  }
  return false;
}

} // namespace

bool ops_cpu_op_relative_pe_values(ggml_backend_t backend, ggml_tensor *node) {
  ops_relative_pe_values_params params;
  if (!ops_extract_relative_pe_values_params(node, params))
    return false;
  const ggml_tensor *weights = params.attn_w;
  const ggml_tensor *relative = params.emb_rel_v;
  const int64_t tokens = weights->ne[1];
  const int64_t width = relative->ne[0];
  const int64_t heads = weights->ne[2];
  const int64_t relative_length = relative->ne[1];
  const int threads = backend_thread_count(backend);

  std::vector<float> converted_relative;
  const float *relative_data = nullptr;
  if (relative->type == GGML_TYPE_F32) {
    relative_data = static_cast<const float *>(relative->data);
  } else if (relative->type == GGML_TYPE_F16) {
    const size_t count = static_cast<size_t>(ggml_nelements(relative));
    converted_relative.resize(count);
    const auto *source = static_cast<const ggml_fp16_t *>(relative->data);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      converted_relative[index] = ggml_fp16_to_fp32(source[index]);
    }
    relative_data = converted_relative.data();
  } else {
    return false;
  }

  if (weights->type == GGML_TYPE_F32) {
    return dispatch_output(static_cast<const float *>(weights->data),
                           relative_data, node, tokens, width, heads,
                           relative_length, params.window_size, threads);
  }
  if (weights->type == GGML_TYPE_F16) {
    return dispatch_output(static_cast<const ggml_fp16_t *>(weights->data),
                           relative_data, node, tokens, width, heads,
                           relative_length, params.window_size, threads);
  }
  return false;
}

} // namespace ggml_ops_ext::cpu
