#include "ops/ops.h"

#include "ggml-common.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define OPS_IMPORT extern "C" __declspec(dllimport)
#else
#define OPS_IMPORT extern "C"
#endif

OPS_IMPORT void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
OPS_IMPORT void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
OPS_IMPORT void ggml_ops_ext_sycl_init();
#endif

namespace {

void set_float_tensor(ggml_tensor *tensor, const std::vector<float> &values) {
  if (tensor->type == GGML_TYPE_F32) {
    ggml_backend_tensor_set(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return;
  }
  std::vector<ggml_fp16_t> converted(values.size());
  std::transform(values.begin(), values.end(), converted.begin(),
                 ggml_fp32_to_fp16);
  ggml_backend_tensor_set(tensor, converted.data(), 0,
                          converted.size() * sizeof(ggml_fp16_t));
}

std::vector<float> read_cache(const ggml_tensor *cache) {
  const int64_t width = cache->ne[0];
  const int64_t rows = cache->ne[1] * cache->ne[2] * cache->ne[3];
  const size_t row_bytes = ggml_row_size(cache->type, width);
  std::vector<uint8_t> raw(row_bytes * static_cast<size_t>(rows));
  ggml_backend_tensor_get(cache, raw.data(), 0, raw.size());
  std::vector<float> values(static_cast<size_t>(rows * width));
  for (int64_t row = 0; row < rows; ++row) {
    const void *source = raw.data() + static_cast<size_t>(row) * row_bytes;
    float *destination = values.data() + row * width;
    if (cache->type == GGML_TYPE_F32) {
      std::memcpy(destination, source,
                  static_cast<size_t>(width) * sizeof(float));
    } else if (cache->type == GGML_TYPE_F16) {
      const auto *input = static_cast<const ggml_fp16_t *>(source);
      for (int64_t channel = 0; channel < width; ++channel) {
        destination[channel] = ggml_fp16_to_fp32(input[channel]);
      }
    } else if (cache->type == GGML_TYPE_Q8_0) {
      dequantize_row_q8_0(static_cast<const block_q8_0 *>(source), destination,
                          width);
    } else {
      dequantize_row_q4_0(static_cast<const block_q4_0 *>(source), destination,
                          width);
    }
  }
  return values;
}

bool check_cache(const std::vector<float> &actual,
                 const std::vector<float> &source, int64_t width,
                 int64_t capacity, int64_t heads, int64_t batches,
                 int64_t tokens, int32_t position, float tolerance) {
  float maximum_error = 0.0f;
  for (int64_t batch = 0; batch < batches; ++batch) {
    for (int64_t head = 0; head < heads; ++head) {
      for (int64_t cache_token = 0; cache_token < capacity; ++cache_token) {
        for (int64_t channel = 0; channel < width; ++channel) {
          const size_t cache_index = static_cast<size_t>(
              ((batch * heads + head) * capacity + cache_token) * width +
              channel);
          float expected = 0.0f;
          if (cache_token >= position && cache_token < position + tokens) {
            const int64_t source_token = cache_token - position;
            const size_t source_index = static_cast<size_t>(
                ((batch * heads + head) * tokens + source_token) * width +
                channel);
            expected = source[source_index];
          }
          maximum_error =
              std::max(maximum_error, std::abs(actual[cache_index] - expected));
        }
      }
    }
  }
  if (maximum_error > tolerance) {
    std::cerr << "KV cache maximum error=" << maximum_error
              << " tolerance=" << tolerance << '\n';
  }
  return maximum_error <= tolerance;
}

bool run_case(ggml_backend_t backend, const std::string &backend_name,
              ggml_type cache_type, ggml_type source_type, int64_t width) {
  constexpr int64_t capacity = 7;
  constexpr int64_t heads = 2;
  constexpr int64_t batches = 2;
  constexpr int64_t tokens = 3;
  constexpr int32_t position_value = 2;
  const size_t source_count =
      static_cast<size_t>(width * tokens * heads * batches);
  std::vector<float> keys(source_count);
  std::vector<float> values(source_count);
  for (size_t index = 0; index < source_count; ++index) {
    keys[index] = 0.9f * std::sin(static_cast<float>(index) * 0.037f);
    values[index] = 0.8f * std::cos(static_cast<float>(index) * 0.043f);
    if (source_type == GGML_TYPE_F16) {
      keys[index] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(keys[index]));
      values[index] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[index]));
    }
  }

  ggml_context *context = ggml_init({4 * 1024 * 1024, nullptr, true});
  ggml_tensor *cache_k =
      ggml_new_tensor_4d(context, cache_type, width, capacity, heads, batches);
  ggml_tensor *cache_v =
      ggml_new_tensor_4d(context, cache_type, width, capacity, heads, batches);
  ggml_tensor *new_k =
      ggml_new_tensor_4d(context, source_type, width, tokens, heads, batches);
  ggml_tensor *new_v =
      ggml_new_tensor_4d(context, source_type, width, tokens, heads, batches);
  ggml_tensor *position = ggml_new_tensor_1d(context, GGML_TYPE_I32, 1);
  ggml_tensor *result = ggml_ops_kv_cache_update(
      context, cache_k, cache_v, new_k, new_v, position, backend);
  if (!result)
    return false;
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(context, backend);
  std::vector<uint8_t> zeros_k(ggml_nbytes(cache_k), 0);
  std::vector<uint8_t> zeros_v(ggml_nbytes(cache_v), 0);
  ggml_backend_tensor_set(cache_k, zeros_k.data(), 0, zeros_k.size());
  ggml_backend_tensor_set(cache_v, zeros_v.data(), 0, zeros_v.size());
  set_float_tensor(new_k, keys);
  set_float_tensor(new_v, values);
  ggml_backend_tensor_set(position, &position_value, 0, sizeof(position_value));
  ggml_cgraph *graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, result);
  bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS;
  int32_t result_value = 0;
  ggml_backend_tensor_get(result, &result_value, 0, sizeof(result_value));
  passed &= result_value == position_value + tokens;
  const float tolerance = cache_type == GGML_TYPE_Q4_0   ? 0.12f
                          : cache_type == GGML_TYPE_Q8_0 ? 0.01f
                          : cache_type == GGML_TYPE_F16  ? 6e-4f
                                                         : 1e-6f;
  passed &= check_cache(read_cache(cache_k), keys, width, capacity, heads,
                        batches, tokens, position_value, tolerance);
  passed &= check_cache(read_cache(cache_v), values, width, capacity, heads,
                        batches, tokens, position_value, tolerance);
  std::cout << backend_name << " KVCache cache=" << ggml_type_name(cache_type)
            << " source=" << ggml_type_name(source_type) << " width=" << width
            << (passed ? " PASSED\n" : " FAILED\n");
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  return passed;
}

} // namespace

int main() {
  ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
  ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
  ggml_ops_ext_sycl_init();
#endif
  ggml_backend_load_all();
  ggml_ops_ext::acquire_ops_hook();
  bool passed = true;
  for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    const std::string name = ggml_backend_dev_name(device);
    if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0))
      continue;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend)
      continue;
    passed &= run_case(backend, name, GGML_TYPE_F32, GGML_TYPE_F16, 37);
    passed &= run_case(backend, name, GGML_TYPE_F16, GGML_TYPE_F32, 37);
    passed &= run_case(backend, name, GGML_TYPE_Q8_0, GGML_TYPE_F32, 64);
    passed &= run_case(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F16, 64);
    ggml_backend_free(backend);
  }
  ggml_ops_ext::release_ops_hook();
  return passed ? 0 : 1;
}
