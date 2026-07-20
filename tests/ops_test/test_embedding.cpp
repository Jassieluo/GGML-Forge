#include "ops/ops.h"

#include <algorithm>
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

bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

std::vector<uint8_t> encode_rows(ggml_type type,
                                 const std::vector<float>& values,
                                 int64_t rows, int64_t width) {
    if (type == GGML_TYPE_F32) {
        const auto* begin = reinterpret_cast<const uint8_t*>(values.data());
        return {begin, begin + values.size() * sizeof(float)};
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        std::transform(values.begin(), values.end(), converted.begin(),
                       ggml_fp32_to_fp16);
        const auto* begin = reinterpret_cast<const uint8_t*>(converted.data());
        return {begin, begin + converted.size() * sizeof(ggml_fp16_t)};
    }

    std::vector<uint8_t> encoded(ggml_row_size(type, width) * rows);
    const size_t written = ggml_quantize_chunk(
        type, values.data(), encoded.data(), 0, rows, width, nullptr);
    return written == encoded.size() ? encoded : std::vector<uint8_t>{};
}

std::vector<float> decode_rows(ggml_type type,
                               const std::vector<uint8_t>& encoded,
                               int64_t rows, int64_t width) {
    std::vector<float> decoded(static_cast<size_t>(rows * width));
    if (type == GGML_TYPE_F32) {
        std::copy_n(reinterpret_cast<const float*>(encoded.data()),
                    decoded.size(), decoded.begin());
        return decoded;
    }
    if (type == GGML_TYPE_F16) {
        const auto* source = reinterpret_cast<const ggml_fp16_t*>(encoded.data());
        std::transform(source, source + decoded.size(), decoded.begin(),
                       ggml_fp16_to_fp32);
        return decoded;
    }

    const ggml_type_traits* traits = ggml_get_type_traits(type);
    const size_t encoded_row_size = ggml_row_size(type, width);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(encoded.data() + row * encoded_row_size,
                         decoded.data() + row * width, width);
    }
    return decoded;
}

bool run_case(ggml_backend_dev_t device, ggml_type weight_type) {
    constexpr int64_t width = 64;
    constexpr int64_t rows = 19;
    constexpr int64_t batches = 2;
    constexpr int64_t tokens = 7;
    const std::vector<int32_t> indices = {
        3, 1, 3, 18, 0, 7, 12,
        5, 5, 2, 9, 16, 4, 1,
    };

    std::vector<float> weights(static_cast<size_t>(width * rows));
    for (size_t index = 0; index < weights.size(); ++index) {
        weights[index] = std::sin(static_cast<float>(index) * 0.031f) * 0.75f;
    }
    const std::vector<uint8_t> encoded =
        encode_rows(weight_type, weights, rows, width);
    if (!require(!encoded.empty(), "failed to encode embedding weights")) {
        return false;
    }
    const std::vector<float> decoded =
        decode_rows(weight_type, encoded, rows, width);

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!require(backend != nullptr, "failed to initialize embedding backend")) {
        return false;
    }

    ggml_context* context =
        ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* weight =
        ggml_new_tensor_2d(context, weight_type, width, rows);
    ggml_tensor* index =
        ggml_new_tensor_2d(context, GGML_TYPE_I32, tokens, batches);
    ggml_tensor* output = ggml_ops_embedding(context, weight, index);
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(context, backend);

    bool passed = require(buffer != nullptr, "failed to allocate embedding tensors");
    if (passed) {
        ggml_backend_tensor_set(weight, encoded.data(), 0, encoded.size());
        ggml_backend_tensor_set(index, indices.data(), 0,
                                indices.size() * sizeof(int32_t));
        ggml_cgraph* graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        passed = require(
            ggml_ops_ext::ops_backend_graph_compute(backend, graph) ==
                GGML_STATUS_SUCCESS,
            "embedding graph execution failed");
    }

    std::vector<float> actual(static_cast<size_t>(width * tokens * batches));
    if (passed) {
        ggml_backend_tensor_get(output, actual.data(), 0,
                                actual.size() * sizeof(float));
        float maximum_error = 0.0f;
        for (int64_t batch = 0; batch < batches; ++batch) {
            for (int64_t token = 0; token < tokens; ++token) {
                const int32_t row = indices[token + tokens * batch];
                for (int64_t channel = 0; channel < width; ++channel) {
                    const size_t output_index = static_cast<size_t>(
                        channel + width * (token + tokens * batch));
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(actual[output_index] -
                                 decoded[channel + width * row]));
                }
            }
        }
        const float tolerance = ggml_is_quantized(weight_type) ? 5e-4f : 1e-6f;
        passed = require(maximum_error <= tolerance,
                         "embedding result exceeded tolerance: " +
                             std::to_string(maximum_error));
    }

    std::cout << ggml_backend_dev_name(device) << " Embedding "
              << ggml_type_name(weight_type) << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
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
    size_t tested_devices = 0;
    const size_t device_count = ggml_backend_dev_count();
    for (size_t index = 0; index < device_count; ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("CPU", 0) != 0 && name.rfind("CUDA", 0) != 0 &&
            name.rfind("SYCL", 0) != 0) {
            continue;
        }
        ++tested_devices;
        passed &= run_case(device, GGML_TYPE_F32);
        passed &= run_case(device, GGML_TYPE_F16);
        passed &= run_case(device, GGML_TYPE_Q4_0);
        passed &= run_case(device, GGML_TYPE_Q8_0);
    }

    passed &= require(tested_devices > 0, "no CPU, CUDA, or SYCL backend was tested");
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
