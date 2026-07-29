#include "ops/ops.h"

#include <algorithm>
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

namespace {

void decode_index(int64_t flat, const int64_t shape[4], int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
}

std::vector<float> reference_reduce(const std::vector<float>& input,
                                    const ggml_ops_ext::ops_reduce_nd_desc& desc) {
    std::vector<float> output(static_cast<size_t>(desc.output_count));
    for (int64_t output_index = 0; output_index < desc.output_count; ++output_index) {
        int64_t output_coord[4];
        int64_t input_coord[4] = {};
        decode_index(output_index, desc.output_shape, output_coord);
        int compact_axis = 0;
        for (int axis = 0; axis < 4; ++axis) {
            if (!(desc.axis_mask & (1u << axis))) {
                input_coord[axis] = output_coord[desc.keep_dims ? axis : compact_axis++];
            }
        }
        float value = desc.mode == ggml_ops_ext::ops_reduce_mode::maximum ||
                              desc.mode == ggml_ops_ext::ops_reduce_mode::logsumexp
                          ? -std::numeric_limits<float>::infinity()
                      : desc.mode == ggml_ops_ext::ops_reduce_mode::minimum
                          ? std::numeric_limits<float>::infinity()
                      : desc.mode == ggml_ops_ext::ops_reduce_mode::product
                          ? 1.0f
                          : 0.0f;
        float auxiliary = 0.0f;
        for (int64_t reduction_index = 0; reduction_index < desc.reduction_count; ++reduction_index) {
            int64_t cursor = reduction_index;
            for (int axis = 0; axis < 4; ++axis) {
                if (desc.axis_mask & (1u << axis)) {
                    input_coord[axis] = cursor % desc.input_shape[axis];
                    cursor /= desc.input_shape[axis];
                }
            }
            const int64_t index = input_coord[0] + desc.input_shape[0] *
                (input_coord[1] + desc.input_shape[1] *
                (input_coord[2] + desc.input_shape[2] * input_coord[3]));
            const float sample = input[static_cast<size_t>(index)];
            if (desc.mode == ggml_ops_ext::ops_reduce_mode::maximum) value = std::max(value, sample);
            else if (desc.mode == ggml_ops_ext::ops_reduce_mode::minimum) value = std::min(value, sample);
            else if (desc.mode == ggml_ops_ext::ops_reduce_mode::product) value *= sample;
            else if (desc.mode == ggml_ops_ext::ops_reduce_mode::variance ||
                     desc.mode == ggml_ops_ext::ops_reduce_mode::standard_deviation) {
                const float delta = sample - value;
                value += delta / static_cast<float>(reduction_index + 1);
                auxiliary += delta * (sample - value);
            } else if (desc.mode == ggml_ops_ext::ops_reduce_mode::logsumexp) {
                if (sample <= value) auxiliary += std::exp(sample - value);
                else {
                    auxiliary = auxiliary * std::exp(value - sample) + 1.0f;
                    value = sample;
                }
            } else value += sample;
        }
        if (desc.mode == ggml_ops_ext::ops_reduce_mode::mean) {
            value /= static_cast<float>(desc.reduction_count);
        }
        else if (desc.mode == ggml_ops_ext::ops_reduce_mode::variance ||
                 desc.mode == ggml_ops_ext::ops_reduce_mode::standard_deviation) {
            value = auxiliary / static_cast<float>(desc.reduction_count - desc.correction);
            if (desc.mode == ggml_ops_ext::ops_reduce_mode::standard_deviation) value = std::sqrt(value);
        } else if (desc.mode == ggml_ops_ext::ops_reduce_mode::logsumexp) value += std::log(auxiliary);
        output[static_cast<size_t>(output_index)] = value;
    }
    return output;
}

const char* mode_name(ggml_ops_ext::ops_reduce_mode mode) {
    switch (mode) {
    case ggml_ops_ext::ops_reduce_mode::sum: return "sum";
    case ggml_ops_ext::ops_reduce_mode::mean: return "mean";
    case ggml_ops_ext::ops_reduce_mode::maximum: return "max";
    case ggml_ops_ext::ops_reduce_mode::minimum: return "min";
    case ggml_ops_ext::ops_reduce_mode::product: return "prod";
    case ggml_ops_ext::ops_reduce_mode::variance: return "var";
    case ggml_ops_ext::ops_reduce_mode::standard_deviation: return "std";
    case ggml_ops_ext::ops_reduce_mode::logsumexp: return "logsumexp";
    }
    return "unknown";
}

bool run_case(ggml_backend_t backend, const std::string& backend_name,
              ggml_ops_ext::ops_reduce_mode mode, uint8_t axis_mask, bool keep_dims) {
    constexpr int64_t shape[4] = {3, 4, 2, 2};
    std::vector<float> input(48);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(static_cast<float>(i + 1) * 0.31f) + static_cast<float>(i % 7) * 0.1f;
    }
    ggml_context* context = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32,
                                                    shape[0], shape[1], shape[2], shape[3]);
    ggml_ops_ext::ops_reduce_nd_config config;
    config.mode = mode;
    config.axis_mask = axis_mask;
    config.keep_dims = keep_dims;
    ggml_tensor* output_tensor = ggml_ops_reduce_nd(context, input_tensor, config, backend);
    if (!output_tensor) {
        ggml_free(context);
        std::cout << backend_name << " reduce_nd builder FAILED\n";
        return false;
    }
    ggml_ops_ext::ops_reduce_nd_encoded_params encoded{};
    ggml_ops_ext::ops_encode_reduce_nd_params(config, encoded);
    ggml_tensor* sources[] = {input_tensor};
    ggml_ops_ext::ops_request request = {ggml_backend_get_device(backend),
        ggml_ops_ext::GGML_OP_OPS_VIRT_REDUCE_ND, sources, 1,
        &encoded, sizeof(encoded), output_tensor};
    ggml_ops_ext::ops_reduce_nd_desc desc;
    if (!ggml_ops_ext::ops_validate_reduce_nd_contract(request, &desc)) {
        ggml_free(context);
        return false;
    }
    const std::vector<float> reference = reference_reduce(input, desc);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(input_tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output_tensor);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(output_tensor, actual.data(), 0, actual.size() * sizeof(float));
    }
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
    }
    const bool passed = status == GGML_STATUS_SUCCESS && max_error < 2e-5f;
    std::cout << backend_name << " reduce_nd " << mode_name(mode)
              << " axes=0x" << std::hex << static_cast<int>(axis_mask) << std::dec
              << (keep_dims ? " keep" : " compact") << " error=" << max_error
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
}

bool run_arg_case(ggml_backend_t backend, const std::string& backend_name,
                  ggml_ops_ext::ops_arg_reduce_mode mode, uint8_t axis_mask) {
    constexpr int64_t shape[4] = {3, 4, 2, 2};
    std::vector<float> input(48);
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i);
    ggml_context* context = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_4d(context, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
    ggml_ops_ext::ops_arg_reduce_nd_config config;
    config.mode = mode; config.axis_mask = axis_mask; config.keep_dims = false;
    ggml_tensor* output = ggml_ops_arg_reduce_nd(context, x, config, backend);
    if (!output) { ggml_free(context); return false; }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context); ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<int32_t> actual(static_cast<size_t>(ggml_nelements(output)));
    if (status == GGML_STATUS_SUCCESS) ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(int32_t));
    const int32_t expected = mode == ggml_ops_ext::ops_arg_reduce_mode::maximum
                                 ? (axis_mask == 0x01 ? 2 : 11) : 0;
    const bool passed = status == GGML_STATUS_SUCCESS &&
                        std::all_of(actual.begin(), actual.end(), [=](int32_t value) { return value == expected; });
    std::cout << backend_name << " arg_reduce "
              << (mode == ggml_ops_ext::ops_arg_reduce_mode::maximum ? "max" : "min")
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer); ggml_free(context); return passed;
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
    bool all_passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = device ? ggml_backend_dev_name(device) : "";
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::sum, 0x01, true);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::mean, 0x06, false);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::maximum, 0x03, true);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::minimum, 0x0f, false);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::product, 0x01, true);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::variance, 0x03, false);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::standard_deviation, 0x06, true);
        all_passed &= run_case(backend, name, ggml_ops_ext::ops_reduce_mode::logsumexp, 0x03, false);
        all_passed &= run_arg_case(backend, name, ggml_ops_ext::ops_arg_reduce_mode::maximum, 0x01);
        all_passed &= run_arg_case(backend, name, ggml_ops_ext::ops_arg_reduce_mode::minimum, 0x03);
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return all_passed ? 0 : 1;
}
