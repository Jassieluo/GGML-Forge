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

static bool run_case(ggml_backend_t backend, const std::string& name, ggml_type type, int64_t width, int64_t rows,
                     int64_t batches, bool benchmark, bool fallback = false) {
    const int64_t output_width = width / 2;
    const size_t input_count = static_cast<size_t>(width * rows * batches);
    std::vector<float> input(input_count);
    for (size_t index = 0; index < input_count; ++index) {
        float value = std::sin(static_cast<float>(index) * 0.013f) * 3.0f;
        input[index] = type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)) : value;
    }

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_3d(context, type, width, rows, batches);
    ggml_tensor* output = ggml_ops_glu(context, x, fallback ? nullptr : backend);
    if (!output)
        return false;
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer)
        return false;
    set_tensor(x, input);
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
        std::cout << "BENCH GLU " << name << " type=" << ggml_type_name(type) << " width=" << width << " rows=" << rows
                  << " batches=" << batches << " " << milliseconds << " ms\n";
    } else if (passed) {
        const std::vector<float> actual = get_tensor(output);
        float maximum_error = 0.0f;
        for (int64_t row = 0; row < rows * batches; ++row) {
            const size_t input_offset = static_cast<size_t>(row * width);
            const size_t output_offset = static_cast<size_t>(row * output_width);
            for (int64_t column = 0; column < output_width; ++column) {
                float expected =
                    input[input_offset + column] / (1.0f + std::exp(-input[input_offset + output_width + column]));
                if (type == GGML_TYPE_F16)
                    expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
                maximum_error = std::max(maximum_error, std::abs(actual[output_offset + column] - expected));
            }
        }
        passed = maximum_error <= (type == GGML_TYPE_F32 ? 2e-5f : 2e-3f);
        std::cout << name << " GLU type=" << ggml_type_name(type) << " width=" << width << " rows=" << rows
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
            for (int64_t width : {128, 512, 2048}) {
                const int64_t rows = std::max<int64_t>(32, 2 * 1024 * 1024 / width);
                passed &= run_case(backend, name, GGML_TYPE_F32, width, rows, 1, true);
                passed &= run_case(backend, name, GGML_TYPE_F16, width, rows, 1, true);
            }
        } else {
            passed &= run_case(backend, name, GGML_TYPE_F32, 62, 7, 3, false);
            passed &= run_case(backend, name, GGML_TYPE_F16, 1024, 5, 2, false);
            if (name.rfind("CPU", 0) == 0) {
                passed &= run_case(backend, name + " fallback", GGML_TYPE_F32, 62, 7, 3, false, true);
            }
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
