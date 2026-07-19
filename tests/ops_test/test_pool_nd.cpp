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

static int64_t volume(const int32_t size[3]) { return int64_t(size[0]) * size[1] * size[2]; }

static bool run_case(ggml_backend_t backend, const std::string& backend_name, int dims,
                     ggml_ops_ext::ops_pool_mode mode, bool include_pad, bool ceil_mode,
                     ggml_type type = GGML_TYPE_F32) {
    ggml_ops_ext::ops_pool_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.input_size[0] = 7;
    config.input_size[1] = dims >= 2 ? 5 : 1;
    config.input_size[2] = dims >= 3 ? 4 : 1;
    config.kernel_size[0] = 3;
    config.kernel_size[1] = dims >= 2 ? 2 : 1;
    config.kernel_size[2] = dims >= 3 ? 2 : 1;
    config.stride[0] = 2;
    config.stride[1] = dims >= 2 ? 2 : 1;
    config.stride[2] = dims >= 3 ? 2 : 1;
    config.padding_before[0] = 1;
    config.padding_after[0] = 2;
    config.padding_before[1] = dims >= 2 ? 1 : 0;
    config.padding_after[1] = 0;
    config.dilation[0] = 2;
    config.ceil_mode = ceil_mode;
    config.count_include_pad = include_pad;
    ggml_ops_ext::ops_pool_nd_encoded_params encoded;
    if (!ggml_ops_ext::ops_encode_pool_nd_params(config, encoded)) {
        return false;
    }

    const int64_t channels = 3;
    const int64_t batch = 2;
    const int64_t input_volume = volume(config.input_size);
    std::vector<float> input(static_cast<size_t>(input_volume * channels * batch));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i + 3) * 0.17f);
    }
    std::vector<ggml_fp16_t> input_f16;
    std::vector<ggml_bf16_t> input_bf16;
    if (type == GGML_TYPE_F16) {
        input_f16.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input_f16[i] = ggml_fp32_to_fp16(input[i]);
            input[i] = ggml_fp16_to_fp32(input_f16[i]);
        }
    } else if (type == GGML_TYPE_BF16) {
        input_bf16.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input_bf16[i] = ggml_fp32_to_bf16(input[i]);
            input[i] = ggml_bf16_to_fp32(input_bf16[i]);
        }
    }

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor =
        dims == 1   ? ggml_new_tensor_3d(context, type, config.input_size[0], channels, batch)
        : dims == 2 ? ggml_new_tensor_4d(context, type, config.input_size[0], config.input_size[1], channels, batch)
                    : ggml_new_tensor_3d(context, type, input_volume, channels, batch);
    ggml_tensor* output_tensor = dims == 1   ? ggml_ops_pool_1d(context, input_tensor, config, backend)
                                 : dims == 2 ? ggml_ops_pool_2d(context, input_tensor, config, backend)
                                             : ggml_ops_pool_3d(context, input_tensor, config, backend);
    if (!output_tensor) {
        ggml_free(context);
        return false;
    }
    int32_t output_size[3] = {static_cast<int32_t>(output_tensor->ne[0]), 1, 1};
    if (dims == 2) {
        output_size[1] = static_cast<int32_t>(output_tensor->ne[1]);
    }
    if (dims == 3) {
        for (int axis = 0; axis < 3; ++axis) {
            const int64_t effective = config.dilation[axis] * int64_t(config.kernel_size[axis] - 1) + 1;
            const int64_t numerator =
                config.input_size[axis] + config.padding_before[axis] + config.padding_after[axis] - effective;
            int64_t size = (numerator + (ceil_mode ? config.stride[axis] - 1 : 0)) / config.stride[axis] + 1;
            if (ceil_mode && size > 1 &&
                (size - 1) * config.stride[axis] >= config.input_size[axis] + config.padding_before[axis]) {
                --size;
            }
            output_size[axis] = static_cast<int32_t>(size);
        }
    }
    const int64_t output_volume = volume(output_size);
    std::vector<float> reference(static_cast<size_t>(output_volume * channels * batch));
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t output_spatial = 0; output_spatial < output_volume; ++output_spatial) {
                const int64_t output_x = output_spatial % output_size[0];
                const int64_t output_remainder = output_spatial / output_size[0];
                const int64_t output_y = output_remainder % output_size[1];
                const int64_t output_z = output_remainder / output_size[1];
                float value =
                    mode == ggml_ops_ext::ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                int64_t valid_elements = 0;
                int64_t padded_elements = 0;
                for (int64_t kernel_z = 0; kernel_z < config.kernel_size[2]; ++kernel_z) {
                    for (int64_t kernel_y = 0; kernel_y < config.kernel_size[1]; ++kernel_y) {
                        for (int64_t kernel_x = 0; kernel_x < config.kernel_size[0]; ++kernel_x) {
                            const int64_t input_x =
                                output_x * config.stride[0] - config.padding_before[0] + kernel_x * config.dilation[0];
                            const int64_t input_y =
                                output_y * config.stride[1] - config.padding_before[1] + kernel_y * config.dilation[1];
                            const int64_t input_z =
                                output_z * config.stride[2] - config.padding_before[2] + kernel_z * config.dilation[2];
                            if (input_x >= -config.padding_before[0] &&
                                input_x < config.input_size[0] + config.padding_after[0] &&
                                input_y >= -config.padding_before[1] &&
                                input_y < config.input_size[1] + config.padding_after[1] &&
                                input_z >= -config.padding_before[2] &&
                                input_z < config.input_size[2] + config.padding_after[2]) {
                                ++padded_elements;
                            }
                            if (input_x < 0 || input_x >= config.input_size[0] || input_y < 0 ||
                                input_y >= config.input_size[1] || input_z < 0 || input_z >= config.input_size[2]) {
                                continue;
                            }
                            const float sample =
                                input[input_x + config.input_size[0] * (input_y + config.input_size[1] * input_z) +
                                      input_volume * (channel + channels * batch_index)];
                            value =
                                mode == ggml_ops_ext::ops_pool_mode::maximum ? std::max(value, sample) : value + sample;
                            ++valid_elements;
                        }
                    }
                }
                if (mode == ggml_ops_ext::ops_pool_mode::average) {
                    value /= float(include_pad ? padded_elements : valid_elements);
                }
                reference[output_spatial + output_volume * (channel + channels * batch_index)] = value;
            }
        }
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(input_tensor, input_f16.data(), 0, input_f16.size() * sizeof(ggml_fp16_t));
    } else if (type == GGML_TYPE_BF16) {
        ggml_backend_tensor_set(input_tensor, input_bf16.data(), 0, input_bf16.size() * sizeof(ggml_bf16_t));
    } else {
        ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    }
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS) {
        if (type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> values(actual.size());
            ggml_backend_tensor_get(output_tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < actual.size(); ++i)
                actual[i] = ggml_fp16_to_fp32(values[i]);
        } else if (type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> values(actual.size());
            ggml_backend_tensor_get(output_tensor, values.data(), 0, values.size() * sizeof(ggml_bf16_t));
            for (size_t i = 0; i < actual.size(); ++i)
                actual[i] = ggml_bf16_to_fp32(values[i]);
        } else {
            ggml_backend_tensor_get(output_tensor, actual.data(), 0, actual.size() * sizeof(float));
        }
    }
    float maximum_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum_error = std::max(maximum_error, std::abs(actual[i] - reference[i]));
    }
    const float tolerance = type == GGML_TYPE_F32 ? 2e-5f : type == GGML_TYPE_F16 ? 2e-3f : 8e-3f;
    const bool passed = status == GGML_STATUS_SUCCESS && maximum_error < tolerance;
    std::cout << backend_name << " " << ggml_type_name(type) << " pool" << dims << "d "
              << (mode == ggml_ops_ext::ops_pool_mode::maximum ? "max" : "avg") << " error=" << maximum_error
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
}

static bool run_benchmark(ggml_backend_t backend, const std::string& backend_name, int dims,
                          ggml_ops_ext::ops_pool_mode mode) {
    ggml_ops_ext::ops_pool_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.input_size[0] = dims == 2 ? 256 : 32;
    config.input_size[1] = dims >= 2 ? (dims == 2 ? 256 : 32) : 1;
    config.input_size[2] = dims == 3 ? 32 : 1;
    for (int axis = 0; axis < dims; ++axis) {
        config.kernel_size[axis] = 3;
        config.stride[axis] = 2;
        config.padding_before[axis] = 1;
        config.padding_after[axis] = 1;
    }
    config.count_include_pad = false;

    const int64_t channels = dims == 2 ? 64 : 16;
    const int64_t batch = 1;
    const int64_t input_volume = volume(config.input_size);
    std::vector<float> input(static_cast<size_t>(input_volume * channels * batch));
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = std::sin(float(i + 1) * 0.013f);

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor = dims == 2 ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0],
                                                               config.input_size[1], channels, batch)
                                          : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, batch);
    ggml_tensor* output_tensor = dims == 2 ? ggml_ops_pool_2d(context, input_tensor, config, backend)
                                           : ggml_ops_pool_3d(context, input_tensor, config, backend);
    if (!output_tensor) {
        ggml_free(context);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    constexpr int iterations = 50;
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(context);
            return false;
        }
    }
    const double milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
    std::cout << "BENCH PoolND " << backend_name << " dims=" << dims << " "
              << (mode == ggml_ops_ext::ops_pool_mode::maximum ? "max" : "avg") << " " << milliseconds << " ms\n";

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
            all_passed &= run_benchmark(backend, name, 2, ggml_ops_ext::ops_pool_mode::maximum);
            all_passed &= run_benchmark(backend, name, 2, ggml_ops_ext::ops_pool_mode::average);
            all_passed &= run_benchmark(backend, name, 3, ggml_ops_ext::ops_pool_mode::maximum);
            all_passed &= run_benchmark(backend, name, 3, ggml_ops_ext::ops_pool_mode::average);
            ggml_backend_free(backend);
            continue;
        }
        all_passed &= run_case(backend, name, 1, ggml_ops_ext::ops_pool_mode::maximum, true, true);
        all_passed &= run_case(backend, name, 2, ggml_ops_ext::ops_pool_mode::average, false, false);
        all_passed &= run_case(backend, name, 3, ggml_ops_ext::ops_pool_mode::average, true, true);
        all_passed &= run_case(backend, name, 2, ggml_ops_ext::ops_pool_mode::maximum, true, false, GGML_TYPE_F16);
        all_passed &= run_case(backend, name, 3, ggml_ops_ext::ops_pool_mode::average, false, false, GGML_TYPE_F16);
        if (name.rfind("SYCL", 0) != 0) {
            all_passed &= run_case(backend, name, 2, ggml_ops_ext::ops_pool_mode::maximum, true, false, GGML_TYPE_BF16);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return all_passed ? 0 : 1;
}
