#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
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

float round_to_type(float value, ggml_type type) {
  return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value))
                               : value;
}

void set_tensor(ggml_tensor *tensor, const std::vector<float> &values) {
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

std::vector<float> get_tensor(const ggml_tensor *tensor) {
  std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
  if (tensor->type == GGML_TYPE_F32) {
    ggml_backend_tensor_get(tensor, values.data(), 0,
                            values.size() * sizeof(float));
    return values;
  }
  std::vector<ggml_fp16_t> converted(values.size());
  ggml_backend_tensor_get(tensor, converted.data(), 0,
                          converted.size() * sizeof(ggml_fp16_t));
  std::transform(converted.begin(), converted.end(), values.begin(),
                 ggml_fp16_to_fp32);
  return values;
}

bool check_error(const std::string &label, const std::vector<float> &actual,
                 const std::vector<float> &expected, float tolerance) {
  float maximum_error = 0.0f;
  for (size_t index = 0; index < actual.size(); ++index) {
    maximum_error =
        std::max(maximum_error, std::abs(actual[index] - expected[index]));
  }
  const bool passed = maximum_error <= tolerance;
  std::cout << label << " error=" << maximum_error
            << (passed ? " PASSED\n" : " FAILED\n");
  return passed;
}

bool run_keys(ggml_backend_t backend, const std::string &backend_name,
              ggml_type query_type, ggml_type relative_type, int32_t window) {
  constexpr int64_t width = 31;
  constexpr int64_t tokens = 11;
  constexpr int64_t heads = 2;
  const int64_t relative_length = 2 * window + 1;
  const float scale = 1.0f / std::sqrt(static_cast<float>(width));
  std::vector<float> query(static_cast<size_t>(width * tokens * heads));
  std::vector<float> relative(
      static_cast<size_t>(width * relative_length * heads));
  for (size_t index = 0; index < query.size(); ++index) {
    query[index] =
        round_to_type(std::sin(static_cast<float>(index) * 0.071f), query_type);
  }
  for (size_t index = 0; index < relative.size(); ++index) {
    relative[index] = round_to_type(
        std::cos(static_cast<float>(index) * 0.053f), relative_type);
  }

  ggml_context *context = ggml_init({4 * 1024 * 1024, nullptr, true});
  ggml_tensor *query_tensor =
      ggml_new_tensor_3d(context, query_type, width, tokens, heads);
  ggml_tensor *relative_tensor =
      ggml_new_tensor_3d(context, relative_type, width, relative_length, heads);
  ggml_tensor *output = ggml_ops_relative_pe_keys(
      context, query_tensor, relative_tensor, scale, window, backend);
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(context, backend);
  set_tensor(query_tensor, query);
  set_tensor(relative_tensor, relative);
  ggml_cgraph *graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output);
  bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS;

  std::vector<float> expected(static_cast<size_t>(tokens * tokens * heads),
                              0.0f);
  for (int64_t head = 0; head < heads; ++head) {
    for (int64_t token = 0; token < tokens; ++token) {
      const int64_t begin = std::max<int64_t>(0, token - window);
      const int64_t end = std::min<int64_t>(tokens, token + window + 1);
      for (int64_t key = begin; key < end; ++key) {
        const int64_t relative_index = key - token + window;
        float sum = 0.0f;
        for (int64_t channel = 0; channel < width; ++channel) {
          sum += query[(head * tokens + token) * width + channel] *
                 relative[(head * relative_length + relative_index) * width +
                          channel];
        }
        expected[(head * tokens + token) * tokens + key] =
            round_to_type(sum * scale, query_type);
      }
    }
  }
  const std::string label =
      backend_name + " RelativePEKeys " + ggml_type_name(query_type) + "/" +
      ggml_type_name(relative_type) + " W=" + std::to_string(window);
  passed &= check_error(label, get_tensor(output), expected,
                        query_type == GGML_TYPE_F16 ? 1e-2f : 3e-5f);
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  return passed;
}

bool run_values(ggml_backend_t backend, const std::string &backend_name,
                ggml_type weight_type, ggml_type relative_type,
                int32_t window) {
  constexpr int64_t width = 31;
  constexpr int64_t tokens = 11;
  constexpr int64_t heads = 2;
  const int64_t relative_length = 2 * window + 1;
  std::vector<float> weights(static_cast<size_t>(tokens * tokens * heads));
  std::vector<float> relative(
      static_cast<size_t>(width * relative_length * heads));
  for (size_t index = 0; index < weights.size(); ++index) {
    weights[index] = round_to_type(
        0.5f + 0.4f * std::sin(static_cast<float>(index) * 0.037f),
        weight_type);
  }
  for (size_t index = 0; index < relative.size(); ++index) {
    relative[index] = round_to_type(
        std::cos(static_cast<float>(index) * 0.061f), relative_type);
  }

  ggml_context *context = ggml_init({4 * 1024 * 1024, nullptr, true});
  ggml_tensor *weight_tensor =
      ggml_new_tensor_3d(context, weight_type, tokens, tokens, heads);
  ggml_tensor *relative_tensor =
      ggml_new_tensor_3d(context, relative_type, width, relative_length, heads);
  ggml_tensor *output = ggml_ops_relative_pe_values(
      context, weight_tensor, relative_tensor, nullptr, window, backend);
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(context, backend);
  set_tensor(weight_tensor, weights);
  set_tensor(relative_tensor, relative);
  ggml_cgraph *graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output);
  bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS;

  std::vector<float> expected(static_cast<size_t>(width * heads * tokens));
  for (int64_t token = 0; token < tokens; ++token) {
    for (int64_t head = 0; head < heads; ++head) {
      const int64_t begin = std::max<int64_t>(0, token - window);
      const int64_t end = std::min<int64_t>(tokens, token + window + 1);
      for (int64_t channel = 0; channel < width; ++channel) {
        float sum = 0.0f;
        for (int64_t key = begin; key < end; ++key) {
          const int64_t relative_index = key - token + window;
          sum += weights[(head * tokens + token) * tokens + key] *
                 relative[(head * relative_length + relative_index) * width +
                          channel];
        }
        expected[(token * heads + head) * width + channel] =
            round_to_type(sum, weight_type);
      }
    }
  }
  const std::string label =
      backend_name + " RelativePEValues " + ggml_type_name(weight_type) + "/" +
      ggml_type_name(relative_type) + " W=" + std::to_string(window);
  passed &= check_error(label, get_tensor(output), expected,
                        weight_type == GGML_TYPE_F16 ? 1e-2f : 3e-5f);
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
    for (const auto types : {std::pair{GGML_TYPE_F32, GGML_TYPE_F32},
                             std::pair{GGML_TYPE_F16, GGML_TYPE_F16},
                             std::pair{GGML_TYPE_F16, GGML_TYPE_F32}}) {
      passed &= run_keys(backend, name, types.first, types.second, 3);
      passed &= run_values(backend, name, types.first, types.second, 3);
    }
    passed &= run_keys(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, 0);
    passed &= run_values(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, 0);
    ggml_backend_free(backend);
  }
  ggml_ops_ext::release_ops_hook();
  return passed ? 0 : 1;
}
