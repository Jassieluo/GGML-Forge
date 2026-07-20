#include "ops/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

static float activate(float value, ggml_ops_gate_activation activation) {
  switch (activation) {
  case ggml_ops_gate_activation::silu:
    return value / (1.0f + std::exp(-value));
  case ggml_ops_gate_activation::gelu:
    return 0.5f * value *
           (1.0f + std::tanh(0.7978845608028654f * value *
                             (1.0f + 0.044715f * value * value)));
  case ggml_ops_gate_activation::relu:
    return std::max(value, 0.0f);
  case ggml_ops_gate_activation::identity:
    return value;
  }
  return value;
}

static void set_tensor(ggml_tensor *tensor, const std::vector<float> &values) {
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

static std::vector<float> get_tensor(const ggml_tensor *tensor) {
  std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
  if (tensor->type == GGML_TYPE_F32) {
    ggml_backend_tensor_get(tensor, result.data(), 0,
                            result.size() * sizeof(float));
    return result;
  }
  std::vector<ggml_fp16_t> converted(result.size());
  ggml_backend_tensor_get(tensor, converted.data(), 0,
                          converted.size() * sizeof(ggml_fp16_t));
  std::transform(converted.begin(), converted.end(), result.begin(),
                 ggml_fp16_to_fp32);
  return result;
}

static bool run_case(ggml_backend_t backend, const std::string &name,
                     ggml_type type, ggml_ops_gate_activation activation,
                     int axis, const int64_t shape[4], bool benchmark,
                     bool fused) {
  int64_t output_shape[4] = {shape[0], shape[1], shape[2], shape[3]};
  output_shape[axis] /= 2;
  const size_t input_count =
      static_cast<size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
  std::vector<float> input(input_count);
  for (size_t index = 0; index < input_count; ++index) {
    float value = std::sin(static_cast<float>(index) * 0.011f) * 3.0f;
    input[index] = type == GGML_TYPE_F16
                       ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value))
                       : value;
  }

  ggml_context *context = ggml_init({4 * 1024 * 1024, nullptr, true});
  ggml_tensor *x =
      ggml_new_tensor_4d(context, type, shape[0], shape[1], shape[2], shape[3]);
  ggml_tensor *output = ggml_ops_gated_activation(context, x, activation, axis,
                                                  fused ? backend : nullptr);
  if (!output)
    return false;
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(context, backend);
  if (!buffer)
    return false;
  set_tensor(x, input);
  ggml_cgraph *graph = ggml_new_graph(context);
  ggml_build_forward_expand(graph, output);
  bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS;
  if (benchmark && passed) {
    constexpr int iterations = 50;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      passed &= ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS;
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - start)
                                    .count() /
                                iterations;
    std::cout << "BENCH GatedActivation " << name
              << (fused ? " fused" : " fallback")
              << " type=" << ggml_type_name(type)
              << " activation=" << static_cast<int>(activation)
              << " axis=" << axis << " " << milliseconds << " ms\n";
  } else if (passed) {
    const std::vector<float> actual = get_tensor(output);
    int64_t inner = 1;
    for (int dimension = 0; dimension < axis; ++dimension)
      inner *= shape[dimension];
    const int64_t half_axis = shape[axis] / 2;
    const int64_t elements =
        output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
    float maximum_error = 0.0f;
    for (int64_t index = 0; index < elements; ++index) {
      const int64_t inner_index = index % inner;
      const int64_t quotient = index / inner;
      const int64_t axis_index = quotient % half_axis;
      const int64_t outer_index = quotient / half_axis;
      const int64_t gate_offset =
          (outer_index * 2 * half_axis + axis_index) * inner + inner_index;
      const int64_t linear_offset = gate_offset + half_axis * inner;
      float expected =
          activate(input[gate_offset], activation) * input[linear_offset];
      if (type == GGML_TYPE_F16)
        expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
      maximum_error =
          std::max(maximum_error, std::abs(actual[index] - expected));
    }
    passed = maximum_error <= (type == GGML_TYPE_F32 ? 3e-5f : 3e-3f);
    std::cout << name << " GatedActivation type=" << ggml_type_name(type)
              << " activation=" << static_cast<int>(activation)
              << " axis=" << axis << " error=" << maximum_error
              << (passed ? " PASSED\n" : " FAILED\n");
  }
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  return passed;
}

int main(int argc, char **argv) {
  ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
  ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
  ggml_ops_ext_sycl_init();
#endif
  ggml_backend_load_all();
  ggml_ops_ext::acquire_ops_hook();
  const bool benchmark = argc > 1 && std::string(argv[1]) == "--benchmark";
  const std::string filter = benchmark && argc > 2 ? argv[2] : "";
  bool passed = true;
  for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    const std::string name = ggml_backend_dev_name(device);
    if (name.rfind("BLAS", 0) == 0 ||
        (!filter.empty() && name.rfind(filter, 0) != 0))
      continue;
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend)
      continue;
    if (benchmark) {
      const int64_t shape[4] = {2048, 1024, 1, 1};
      for (auto activation :
           {ggml_ops_gate_activation::silu, ggml_ops_gate_activation::gelu}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
          passed &=
              run_case(backend, name, type, activation, 0, shape, true, false);
          passed &=
              run_case(backend, name, type, activation, 0, shape, true, true);
        }
      }
    } else {
      const int64_t shapes[3][4] = {{62, 7, 3, 1}, {5, 14, 3, 1}, {5, 7, 6, 1}};
      for (int axis = 0; axis < 3; ++axis) {
        for (auto activation :
             {ggml_ops_gate_activation::silu, ggml_ops_gate_activation::gelu,
              ggml_ops_gate_activation::relu,
              ggml_ops_gate_activation::identity}) {
          passed &= run_case(backend, name, GGML_TYPE_F32, activation, axis,
                             shapes[axis], false, true);
          passed &= run_case(backend, name, GGML_TYPE_F16, activation, axis,
                             shapes[axis], false, true);
        }
      }
    }
    ggml_backend_free(backend);
  }
  ggml_ops_ext::release_ops_hook();
  return passed ? 0 : 1;
}
