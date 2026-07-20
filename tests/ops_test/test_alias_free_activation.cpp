#include "ops/ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

constexpr int64_t kFilterSize = 12;
constexpr int64_t kPadding = 5;

bool require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

float round_to_type(float value, ggml_type type) {
  return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value))
                               : value;
}

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

std::vector<float> get_float_tensor(const ggml_tensor *tensor) {
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

std::vector<float>
reference(const std::vector<float> &input, const std::vector<float> &up,
          const std::vector<float> &down, const std::vector<float> &alpha,
          const std::vector<float> &beta, int64_t length, int64_t channels,
          int64_t batch, ggml_type output_type) {
  std::vector<float> output(static_cast<size_t>(length * channels * batch));
  std::vector<float> upsampled(static_cast<size_t>(2 * length));
  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    for (int64_t channel = 0; channel < channels; ++channel) {
      std::fill(upsampled.begin(), upsampled.end(), 0.0f);
      for (int64_t position = 0; position < 2 * length; ++position) {
        float value = 0.0f;
        for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
          const int64_t numerator = position + kPadding - kernel;
          if (numerator < 0 || (numerator & 1) != 0) {
            continue;
          }
          const int64_t input_index = numerator / 2;
          if (input_index >= length) {
            continue;
          }
          value +=
              input[input_index + length * (channel + channels * batch_index)] *
              up[kernel + kFilterSize * channel];
        }
        value *= 2.0f;
        const float sine = std::sin(value * alpha[channel]);
        upsampled[position] = value + sine * sine / beta[channel];
      }
      for (int64_t position = 0; position < length; ++position) {
        float value = 0.0f;
        for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
          const int64_t source = 2 * position - kPadding + kernel;
          if (source >= 0 && source < 2 * length) {
            value += upsampled[source] * down[kernel + kFilterSize * channel];
          }
        }
        output[position + length * (channel + channels * batch_index)] =
            round_to_type(value, output_type);
      }
    }
  }
  return output;
}

bool run_case(ggml_backend_dev_t device, ggml_type type) {
  constexpr int64_t length = 137;
  constexpr int64_t channels = 3;
  constexpr int64_t batch = 2;
  std::vector<float> input(static_cast<size_t>(length * channels * batch));
  std::vector<float> up(static_cast<size_t>(kFilterSize * channels));
  std::vector<float> down(static_cast<size_t>(kFilterSize * channels));
  std::vector<float> alpha(channels);
  std::vector<float> beta(channels);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index) * 0.037f) * 0.4f;
  }
  for (size_t index = 0; index < up.size(); ++index) {
    up[index] = std::cos(static_cast<float>(index) * 0.19f) * 0.12f;
    down[index] = std::sin(static_cast<float>(index) * 0.13f + 0.2f) * 0.1f;
  }
  for (int64_t channel = 0; channel < channels; ++channel) {
    alpha[channel] = 0.7f + 0.1f * static_cast<float>(channel);
    beta[channel] = 1.1f + 0.2f * static_cast<float>(channel);
  }
  for (float &value : input)
    value = round_to_type(value, type);
  for (float &value : up)
    value = round_to_type(value, type);
  for (float &value : down)
    value = round_to_type(value, type);
  for (float &value : alpha)
    value = round_to_type(value, type);
  for (float &value : beta)
    value = round_to_type(value, type);
  const std::vector<float> expected =
      reference(input, up, down, alpha, beta, length, channels, batch, type);

  ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
  if (!require(backend != nullptr, "failed to initialize alias-free backend")) {
    return false;
  }
  ggml_context *context = ggml_init({4 * 1024 * 1024, nullptr, true});
  ggml_tensor *x = ggml_new_tensor_3d(context, type, length, channels, batch);
  ggml_tensor *up_tensor =
      ggml_new_tensor_3d(context, type, kFilterSize, 1, channels);
  ggml_tensor *down_tensor =
      ggml_new_tensor_3d(context, type, kFilterSize, 1, channels);
  ggml_tensor *alpha_tensor = ggml_new_tensor_1d(context, type, channels);
  ggml_tensor *beta_tensor = ggml_new_tensor_1d(context, type, channels);
  ggml_tensor *output = ggml_ops_alias_free_activation(
      context, x, up_tensor, down_tensor, alpha_tensor, beta_tensor, backend);
  ggml_tensor *fallback = nullptr;
  if (type == GGML_TYPE_F32) {
    ggml_tensor *upsampled = ggml_ops_conv_transpose_1d(
        context, up_tensor, x, 2, 5, 1, channels, backend);
    fallback =
        upsampled
            ? ggml_ops_snake_beta(context, ggml_scale(context, upsampled, 2.0f),
                                  alpha_tensor, beta_tensor, backend)
            : nullptr;
    fallback = fallback ? ggml_ops_conv_1d(context, down_tensor, fallback, 2, 5,
                                           1, channels, backend)
                        : nullptr;
  }
  bool passed = require(output != nullptr,
                        std::string(ggml_backend_dev_name(device)) +
                            " did not expose fused AliasFreeActivation");
  passed &= require(type != GGML_TYPE_F32 || fallback != nullptr,
                    "failed to build AliasFreeActivation composition");
  ggml_backend_buffer_t buffer = nullptr;
  if (passed) {
    buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    passed &=
        require(buffer != nullptr, "failed to allocate alias-free tensors");
  }
  if (passed) {
    set_float_tensor(x, input);
    set_float_tensor(up_tensor, up);
    set_float_tensor(down_tensor, down);
    set_float_tensor(alpha_tensor, alpha);
    set_float_tensor(beta_tensor, beta);
    ggml_cgraph *graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (fallback) {
      ggml_build_forward_expand(graph, fallback);
    }
    passed &= require(ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                          GGML_STATUS_SUCCESS,
                      "AliasFreeActivation execution failed");
  }
  if (passed) {
    const std::vector<float> actual = get_float_tensor(output);
    const std::vector<float> composed =
        fallback ? get_float_tensor(fallback) : std::vector<float>{};
    float maximum_error = 0.0f;
    float composition_error = 0.0f;
    for (size_t index = 0; index < actual.size(); ++index) {
      maximum_error =
          std::max(maximum_error, std::abs(actual[index] - expected[index]));
      if (fallback) {
        composition_error = std::max(composition_error,
                                     std::abs(actual[index] - composed[index]));
      }
    }
    const float tolerance = type == GGML_TYPE_F16 ? 8e-4f : 2e-5f;
    passed &=
        require(maximum_error <= tolerance,
                "AliasFreeActivation error=" + std::to_string(maximum_error));
    passed &= require(composition_error <= tolerance,
                      "AliasFreeActivation composition error=" +
                          std::to_string(composition_error));
  }
  std::cout << ggml_backend_dev_name(device) << " AliasFreeActivation "
            << ggml_type_name(type) << (passed ? " PASSED\n" : " FAILED\n");
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  ggml_backend_free(backend);
  return passed;
}

bool run_benchmark(ggml_backend_dev_t device) {
  constexpr int64_t length = 2048;
  constexpr int64_t channels = 64;
  constexpr int iterations = 20;
  ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
  if (!backend) {
    return false;
  }
  ggml_context *context = ggml_init({32 * 1024 * 1024, nullptr, true});
  ggml_tensor *x =
      ggml_new_tensor_3d(context, GGML_TYPE_F32, length, channels, 1);
  ggml_tensor *up =
      ggml_new_tensor_3d(context, GGML_TYPE_F32, kFilterSize, 1, channels);
  ggml_tensor *down =
      ggml_new_tensor_3d(context, GGML_TYPE_F32, kFilterSize, 1, channels);
  ggml_tensor *alpha = ggml_new_tensor_1d(context, GGML_TYPE_F32, channels);
  ggml_tensor *beta = ggml_new_tensor_1d(context, GGML_TYPE_F32, channels);
  ggml_tensor *fused = ggml_ops_alias_free_activation(context, x, up, down,
                                                      alpha, beta, backend);
  ggml_tensor *upsampled =
      ggml_ops_conv_transpose_1d(context, up, x, 2, 5, 1, channels, backend);
  ggml_tensor *activated =
      upsampled
          ? ggml_ops_snake_beta(context, ggml_scale(context, upsampled, 2.0f),
                                alpha, beta, backend)
          : nullptr;
  ggml_tensor *composed = activated
                              ? ggml_ops_conv_1d(context, down, activated, 2, 5,
                                                 1, channels, backend)
                              : nullptr;
  if (!fused || !composed) {
    ggml_free(context);
    ggml_backend_free(backend);
    return false;
  }
  ggml_backend_buffer_t buffer =
      ggml_backend_alloc_ctx_tensors(context, backend);
  std::vector<float> input(static_cast<size_t>(length * channels), 0.05f);
  std::vector<float> filter(static_cast<size_t>(kFilterSize * channels), 0.02f);
  std::vector<float> parameter(channels, 1.0f);
  set_float_tensor(x, input);
  set_float_tensor(up, filter);
  set_float_tensor(down, filter);
  set_float_tensor(alpha, parameter);
  set_float_tensor(beta, parameter);
  ggml_cgraph *fused_graph = ggml_new_graph_custom(context, 64, false);
  ggml_build_forward_expand(fused_graph, fused);
  ggml_cgraph *composed_graph = ggml_new_graph_custom(context, 64, false);
  ggml_build_forward_expand(composed_graph, composed);

  auto measure = [&](ggml_cgraph *graph) {
    for (int warmup = 0; warmup < 3; ++warmup) {
      ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    }
    ggml_backend_synchronize(backend);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           iterations;
  };
  const double fused_ms = measure(fused_graph);
  const double composed_ms = measure(composed_graph);
  std::cout << ggml_backend_dev_name(device)
            << " AliasFreeActivation benchmark fused=" << fused_ms
            << " ms composed=" << composed_ms
            << " ms speedup=" << composed_ms / fused_ms << "x\n";
  ggml_backend_buffer_free(buffer);
  ggml_free(context);
  ggml_backend_free(backend);
  return fused_ms <= composed_ms * 1.05;
}

} // namespace

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
  bool passed = true;
  const bool benchmark = argc > 1 && std::string(argv[1]) == "--benchmark";
  size_t tested_devices = 0;
  for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
    ggml_backend_dev_t device = ggml_backend_dev_get(index);
    const std::string name = ggml_backend_dev_name(device);
    if (name.rfind("CPU", 0) != 0 && name.rfind("CUDA", 0) != 0 &&
        name.rfind("SYCL", 0) != 0) {
      continue;
    }
    ++tested_devices;
    if (benchmark) {
      passed &= run_benchmark(device);
    } else {
      passed &= run_case(device, GGML_TYPE_F32);
      passed &= run_case(device, GGML_TYPE_F16);
    }
  }
  passed &=
      require(tested_devices > 0, "no CPU, CUDA, or SYCL backend was tested");
  ggml_ops_ext::release_ops_hook();
  return passed ? 0 : 1;
}
