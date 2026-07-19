#include "ops/ops.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
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

static int64_t volume(const int32_t s[3]) { return int64_t(s[0]) * s[1] * s[2]; }

static bool run_case(ggml_backend_t backend, const std::string& name, int dims, ggml_ops_ext::ops_pool_mode mode,
                     bool global = false) {
    ggml_ops_ext::ops_adaptive_pool_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.input_size[0] = 7;
    config.input_size[1] = dims >= 2 ? 5 : 1;
    config.input_size[2] = dims >= 3 ? 4 : 1;
    config.output_size[0] = global ? 1 : 3;
    config.output_size[1] = global ? 1 : (dims >= 2 ? 4 : 1);
    config.output_size[2] = global ? 1 : (dims >= 3 ? 2 : 1);
    const int64_t input_volume = volume(config.input_size);
    const int64_t output_volume = volume(config.output_size);
    const int64_t channels = 2;
    const int64_t batch = 2;
    std::vector<float> input(static_cast<size_t>(input_volume * channels * batch));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i + 1) * 0.137f);
    }
    std::vector<float> reference(static_cast<size_t>(output_volume * channels * batch));
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t output_spatial = 0; output_spatial < output_volume; ++output_spatial) {
                const int64_t output_x = output_spatial % config.output_size[0];
                const int64_t output_remainder = output_spatial / config.output_size[0];
                const int64_t output_y = output_remainder % config.output_size[1];
                const int64_t output_z = output_remainder / config.output_size[1];
                const int64_t input_x_begin = output_x * config.input_size[0] / config.output_size[0];
                const int64_t input_x_end =
                    ((output_x + 1) * config.input_size[0] + config.output_size[0] - 1) / config.output_size[0];
                const int64_t input_y_begin = output_y * config.input_size[1] / config.output_size[1];
                const int64_t input_y_end =
                    ((output_y + 1) * config.input_size[1] + config.output_size[1] - 1) / config.output_size[1];
                const int64_t input_z_begin = output_z * config.input_size[2] / config.output_size[2];
                const int64_t input_z_end =
                    ((output_z + 1) * config.input_size[2] + config.output_size[2] - 1) / config.output_size[2];
                float value =
                    mode == ggml_ops_ext::ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                for (int64_t z = input_z_begin; z < input_z_end; ++z) {
                    for (int64_t y = input_y_begin; y < input_y_end; ++y) {
                        for (int64_t x = input_x_begin; x < input_x_end; ++x) {
                            const float sample = input[x + config.input_size[0] * (y + config.input_size[1] * z) +
                                                       input_volume * (channel + channels * batch_index)];
                            value =
                                mode == ggml_ops_ext::ops_pool_mode::maximum ? std::max(value, sample) : value + sample;
                        }
                    }
                }
                if (mode == ggml_ops_ext::ops_pool_mode::average) {
                    value /= float((input_x_end - input_x_begin) * (input_y_end - input_y_begin) *
                                   (input_z_end - input_z_begin));
                }
                reference[output_spatial + output_volume * (channel + channels * batch_index)] = value;
            }
        }
    }
    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor =
        dims == 1 ? ggml_new_tensor_3d(context, GGML_TYPE_F32, config.input_size[0], channels, batch)
        : dims == 2
            ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0], config.input_size[1], channels, batch)
            : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, batch);
    ggml_tensor* output_tensor = dims == 1   ? ggml_ops_adaptive_pool_1d(context, input_tensor, config, backend)
                                 : dims == 2 ? ggml_ops_adaptive_pool_2d(context, input_tensor, config, backend)
                                             : ggml_ops_adaptive_pool_3d(context, input_tensor, config, backend);
    if (!output_tensor) {
        ggml_free(context);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output_tensor, actual.data(), 0, actual.size() * sizeof(float));
    }
    float maximum_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum_error = std::max(maximum_error, std::abs(actual[i] - reference[i]));
    }
    const bool passed = status == GGML_STATUS_SUCCESS && maximum_error < 2e-5f;
    std::cout << name << " adaptive_pool" << dims << "d "
              << (mode == ggml_ops_ext::ops_pool_mode::maximum ? "max" : "avg") << (global ? " global" : "")
              << " error=" << maximum_error << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
}

static bool run_benchmark(ggml_backend_t backend, const std::string& name, int dims, ggml_ops_ext::ops_pool_mode mode,
                          bool global) {
    ggml_ops_ext::ops_adaptive_pool_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.input_size[0] = dims == 2 ? 256 : 32;
    config.input_size[1] = dims == 2 ? 256 : 32;
    config.input_size[2] = dims == 3 ? 32 : 1;
    config.output_size[0] = global ? 1 : (dims == 2 ? 37 : 11);
    config.output_size[1] = global ? 1 : (dims == 2 ? 29 : 9);
    config.output_size[2] = global ? 1 : (dims == 3 ? 7 : 1);
    const int64_t channels = dims == 2 ? 64 : 16;
    const int64_t input_volume = volume(config.input_size);
    std::vector<float> input(static_cast<size_t>(input_volume * channels));
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = std::sin(float(i + 1) * 0.013f);

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor =
        dims == 2 ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0], config.input_size[1], channels, 1)
                  : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, 1);
    ggml_tensor* output_tensor = dims == 2 ? ggml_ops_adaptive_pool_2d(context, input_tensor, config, backend)
                                           : ggml_ops_adaptive_pool_3d(context, input_tensor, config, backend);
    if (!output_tensor) {
        ggml_free(context);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    constexpr int iterations = 50;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
    std::cout << "BENCH AdaptivePoolND " << name << " dims=" << dims << " "
              << (mode == ggml_ops_ext::ops_pool_mode::maximum ? "max" : "avg") << (global ? " global " : " ") << ms
              << " ms\n";
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return true;
}

int main(int argc, char** argv) {
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
    const std::string benchmark_device = benchmark && argc > 2 ? argv[2] : "";
    bool all_passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = device ? ggml_backend_dev_name(device) : "";
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        if (benchmark && !benchmark_device.empty() && name.rfind(benchmark_device, 0) != 0) {
            ggml_backend_free(backend);
            continue;
        }
        if (benchmark) {
            for (int dims = 2; dims <= 3; ++dims) {
                all_passed &= run_benchmark(backend, name, dims, ggml_ops_ext::ops_pool_mode::maximum, false);
                all_passed &= run_benchmark(backend, name, dims, ggml_ops_ext::ops_pool_mode::average, false);
                all_passed &= run_benchmark(backend, name, dims, ggml_ops_ext::ops_pool_mode::average, true);
            }
            ggml_backend_free(backend);
            continue;
        }
        for (int dims = 1; dims <= 3; ++dims) {
            all_passed &= run_case(backend, name, dims, ggml_ops_ext::ops_pool_mode::maximum);
            all_passed &= run_case(backend, name, dims, ggml_ops_ext::ops_pool_mode::average);
            all_passed &= run_case(backend, name, dims, ggml_ops_ext::ops_pool_mode::average, true);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return all_passed ? 0 : 1;
}
