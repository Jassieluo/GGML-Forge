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

static float round_to_type(float value, ggml_type type) {
    return type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : value;
}

static void set_tensor(ggml_tensor* tensor, const std::vector<float>& values) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
        return;
    }
    std::vector<ggml_fp16_t> converted(values.size());
    std::transform(values.begin(), values.end(), converted.begin(), ggml_fp32_to_fp16);
    ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(ggml_fp16_t));
}

static std::vector<float> get_tensor(const ggml_tensor* tensor) {
    std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
    std::vector<ggml_fp16_t> converted(result.size());
    ggml_backend_tensor_get(tensor, converted.data(), 0, converted.size() * sizeof(ggml_fp16_t));
    std::transform(converted.begin(), converted.end(), result.begin(), ggml_fp16_to_fp32);
    return result;
}

static std::vector<float> reference(const std::vector<float>& input, const std::vector<float>& gamma,
                                    const std::vector<float>& beta, int64_t length, int64_t channels, int64_t batches,
                                    float eps) {
    std::vector<float> output(input.size());
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            const size_t offset = static_cast<size_t>((batch * channels + channel) * length);
            float sum = 0.0f;
            for (int64_t index = 0; index < length; ++index)
                sum += input[offset + index];
            const float mean = sum / static_cast<float>(length);
            float squared_sum = 0.0f;
            for (int64_t index = 0; index < length; ++index) {
                const float difference = input[offset + index] - mean;
                squared_sum += difference * difference;
            }
            const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(length) + eps);
            for (int64_t index = 0; index < length; ++index) {
                output[offset + index] = (input[offset + index] - mean) * inverse_std * gamma[channel] + beta[channel];
            }
        }
    }
    return output;
}

static bool run_case(ggml_backend_t backend, const std::string& name, ggml_type type, ggml_type parameter_type,
                     int64_t length, int64_t channels, int64_t batches, bool benchmark) {
    constexpr float eps = 1e-5f;
    const size_t count = static_cast<size_t>(length * channels * batches);
    std::vector<float> input(count);
    std::vector<float> gamma(static_cast<size_t>(channels));
    std::vector<float> beta(static_cast<size_t>(channels));
    for (size_t index = 0; index < count; ++index) {
        input[index] = round_to_type(std::sin(static_cast<float>(index) * 0.017f) * 1.3f + 0.2f, type);
    }
    for (int64_t channel = 0; channel < channels; ++channel) {
        gamma[channel] = round_to_type(0.8f + 0.2f * std::sin(static_cast<float>(channel) * 0.03f), parameter_type);
        beta[channel] = round_to_type(0.1f * std::cos(static_cast<float>(channel) * 0.02f), parameter_type);
    }

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_3d(context, type, length, channels, batches);
    ggml_tensor* scale = ggml_new_tensor_1d(context, parameter_type, channels);
    ggml_tensor* shift = ggml_new_tensor_1d(context, parameter_type, channels);
    ggml_tensor* output = ggml_ops_instance_norm(context, x, scale, shift, eps, backend);
    if (!output)
        return false;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer)
        return false;
    set_tensor(x, input);
    set_tensor(scale, gamma);
    set_tensor(shift, beta);
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (benchmark && passed) {
        constexpr int iterations = 100;
        const auto start = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            passed &= ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
        }
        const double milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
        std::cout << "BENCH InstanceNorm " << name << " type=" << ggml_type_name(type)
                  << " param=" << ggml_type_name(parameter_type) << " length=" << length << " channels=" << channels
                  << " batches=" << batches << " " << milliseconds << " ms\n";
    } else if (passed) {
        const std::vector<float> actual = get_tensor(output);
        const std::vector<float> expected = reference(input, gamma, beta, length, channels, batches, eps);
        float maximum_error = 0.0f;
        for (size_t index = 0; index < count; ++index) {
            maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
        }
        passed = maximum_error <= (type == GGML_TYPE_F32 ? 5e-4f : 3e-3f);
        std::cout << name << " InstanceNorm type=" << ggml_type_name(type)
                  << " param=" << ggml_type_name(parameter_type) << " length=" << length << " channels=" << channels
                  << " batches=" << batches << " error=" << maximum_error << (passed ? " PASSED\n" : " FAILED\n");
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
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
    const std::string filter = benchmark && argc > 2 ? argv[2] : "";
    bool passed = true;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("BLAS", 0) == 0 || (!filter.empty() && name.rfind(filter, 0) != 0))
            continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend)
            continue;
        if (benchmark) {
            for (int64_t length : {128, 250, 1024, 4096}) {
                const int64_t channels = std::max<int64_t>(64, 1024 * 1024 / length);
                passed &= run_case(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, length, channels, 1, true);
                passed &= run_case(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, length, channels, 1, true);
            }
        } else {
            passed &= run_case(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, 31, 7, 3, false);
            passed &= run_case(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, 250, 13, 2, false);
            passed &= run_case(backend, name, GGML_TYPE_F16, GGML_TYPE_F32, 1024, 5, 3, false);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
