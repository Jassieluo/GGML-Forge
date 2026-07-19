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

static std::vector<float> reference_layer_norm(const std::vector<float>& input, const std::vector<float>& gamma,
                                               const std::vector<float>& beta, int64_t width, int64_t rows,
                                               float eps) {
    std::vector<float> output(input.size());
    for (int64_t row = 0; row < rows; ++row) {
        const float* source = input.data() + row * width;
        float sum = 0.0f;
        for (int64_t column = 0; column < width; ++column) {
            sum += source[column];
        }
        const float mean = sum / static_cast<float>(width);
        float squared_sum = 0.0f;
        for (int64_t column = 0; column < width; ++column) {
            const float difference = source[column] - mean;
            squared_sum += difference * difference;
        }
        const float inverse_std = 1.0f / std::sqrt(squared_sum / static_cast<float>(width) + eps);
        for (int64_t column = 0; column < width; ++column) {
            output[static_cast<size_t>(row * width + column)] =
                (source[column] - mean) * inverse_std * gamma[static_cast<size_t>(column)] +
                beta[static_cast<size_t>(column)];
        }
    }
    return output;
}

static bool run_case(ggml_backend_t backend, const std::string& backend_name, ggml_type type, int64_t width,
                     int64_t rows, bool benchmark) {
    constexpr float eps = 1e-5f;
    const size_t count = static_cast<size_t>(width * rows);
    std::vector<float> input(count);
    std::vector<float> gamma(static_cast<size_t>(width));
    std::vector<float> beta(static_cast<size_t>(width));
    for (size_t index = 0; index < count; ++index) {
        input[index] = std::sin(static_cast<float>(index) * 0.013f) * 1.7f +
                       std::cos(static_cast<float>(index) * 0.003f) * 0.2f;
    }
    for (int64_t column = 0; column < width; ++column) {
        gamma[static_cast<size_t>(column)] = 0.75f + 0.25f * std::sin(static_cast<float>(column) * 0.01f);
        beta[static_cast<size_t>(column)] = 0.1f * std::cos(static_cast<float>(column) * 0.02f);
    }

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_2d(context, type, width, rows);
    ggml_tensor* scale = ggml_new_tensor_1d(context, type, width);
    ggml_tensor* shift = ggml_new_tensor_1d(context, type, width);
    ggml_tensor* output = ggml_ops_layer_norm(context, x, scale, shift, eps, backend);
    if (!output) {
        ggml_free(context);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) {
        ggml_free(context);
        return false;
    }
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
        std::cout << "BENCH LayerNorm " << backend_name << " type=" << ggml_type_name(type) << " width=" << width
                  << " rows=" << rows << " " << milliseconds << " ms\n";
    } else if (passed) {
        const std::vector<float> actual = get_tensor(output);
        const std::vector<float> expected = reference_layer_norm(input, gamma, beta, width, rows, eps);
        float maximum_error = 0.0f;
        for (size_t index = 0; index < count; ++index) {
            maximum_error = std::max(maximum_error, std::abs(actual[index] - expected[index]));
        }
        const float tolerance = type == GGML_TYPE_F32 ? 2e-4f : 5e-3f;
        passed = maximum_error <= tolerance;
        std::cout << backend_name << " LayerNorm type=" << ggml_type_name(type) << " width=" << width
                  << " rows=" << rows << " error=" << maximum_error << (passed ? " PASSED\n" : " FAILED\n");
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
        if (name.rfind("BLAS", 0) == 0 || (!filter.empty() && name.rfind(filter, 0) != 0)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        if (benchmark) {
            for (int64_t width : {128, 768, 1024, 4096}) {
                const int64_t rows = std::max<int64_t>(32, 1024 * 1024 / width);
                passed &= run_case(backend, name, GGML_TYPE_F32, width, rows, true);
                passed &= run_case(backend, name, GGML_TYPE_F16, width, rows, true);
            }
        } else {
            for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                passed &= run_case(backend, name, type, 31, 7, false);
                passed &= run_case(backend, name, type, 768, 5, false);
                passed &= run_case(backend, name, type, 4096, 3, false);
            }
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
