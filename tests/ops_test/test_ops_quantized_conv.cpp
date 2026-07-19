#include "ops/ops.h"
#include "ops/cpu.h"

#include <cmath>
#include <chrono>
#include <cstdint>
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

static std::vector<uint8_t> quantize_weights(
    ggml_type type, const std::vector<float>& src, int64_t rows, int64_t row_size
) {
    std::vector<uint8_t> data(ggml_row_size(type, row_size) * rows);
    const size_t written = ggml_quantize_chunk(type, src.data(), data.data(), 0, rows, row_size, nullptr);
    if (written != data.size()) return {};
    return data;
}

static std::vector<float> dequantize_weights(
    ggml_type type, const std::vector<uint8_t>& src, int64_t rows, int64_t row_size
) {
    const ggml_type_traits* traits = ggml_get_type_traits(type);
    std::vector<float> result(rows * row_size);
    const size_t quant_row_size = ggml_row_size(type, row_size);
    for (int64_t row = 0; row < rows; ++row) {
        traits->to_float(src.data() + row * quant_row_size, result.data() + row * row_size, row_size);
    }
    return result;
}

static bool run_case(
    ggml_backend_t backend,
    const std::string& backend_name,
    ggml_type qtype,
    bool transpose,
    ggml_type activation_type = GGML_TYPE_F32
) {
    constexpr int64_t kW = 3;
    const int64_t c_in = qtype == GGML_TYPE_Q4_K ? 256 : 32;
    const int64_t c_out = qtype == GGML_TYPE_Q4_K ? 256 : 32;
    constexpr int64_t batch = 1;
    const int64_t l_in = 8;
    const int stride = 1;
    const int padding = 0;
    const int dilation = 1;
    const int64_t l_out = transpose
        ? (l_in - 1) * stride - 2 * padding + dilation * (kW - 1) + 1
        : (l_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    const int64_t quant_channels = transpose ? c_out : c_in;
    const int64_t weight_rows = kW * (transpose ? c_in : c_out);
    std::vector<float> weights(weight_rows * quant_channels);
    std::vector<float> input(l_in * c_in * batch);
    std::vector<float> bias(c_out);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * 0.17f) * 0.5f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.11f) * 0.25f;
    for (int64_t i = 0; i < c_out; ++i) bias[i] = float(i - 1) * 0.125f;
    std::vector<float> reference_input = input;
    if (activation_type == GGML_TYPE_F16) {
        for (float& value : reference_input) {
            value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
        }
    }

    const std::vector<uint8_t> quantized = quantize_weights(qtype, weights, weight_rows, quant_channels);
    if (quantized.empty()) return false;
    const std::vector<float> decoded = dequantize_weights(qtype, quantized, weight_rows, quant_channels);
    std::vector<float> reference(l_out * c_out * batch, 0.0f);

    if (!transpose) {
        for (int64_t oc = 0; oc < c_out; ++oc) {
            for (int64_t ow = 0; ow < l_out; ++ow) {
                float sum = bias[oc];
                for (int64_t ic = 0; ic < c_in; ++ic) {
                    for (int64_t k = 0; k < kW; ++k) {
                        sum += decoded[ic + c_in * (k + kW * oc)] * reference_input[ow + k + l_in * ic];
                    }
                }
                reference[ow + l_out * oc] = sum;
            }
        }
    } else {
        for (int64_t oc = 0; oc < c_out; ++oc) {
            for (int64_t ow = 0; ow < l_out; ++ow) reference[ow + l_out * oc] = bias[oc];
        }
        for (int64_t ic = 0; ic < c_in; ++ic) {
            for (int64_t iw = 0; iw < l_in; ++iw) {
                for (int64_t oc = 0; oc < c_out; ++oc) {
                    for (int64_t k = 0; k < kW; ++k) {
                        const int64_t ow = iw * stride - padding + k * dilation;
                        reference[ow + l_out * oc] +=
                            reference_input[iw + l_in * ic] * decoded[oc + c_out * (k + kW * ic)];
                    }
                }
            }
        }
    }

    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* w = transpose
        ? ggml_new_tensor_3d(ctx, qtype, c_out, kW, c_in)
        : ggml_new_tensor_3d(ctx, qtype, c_in, kW, c_out);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, activation_type, l_in, c_in, batch);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c_out);
    ggml_tensor* dst = transpose
        ? ggml_ops_conv_transpose_1d(ctx, w, x, stride, padding, dilation, 1, backend, b)
        : ggml_ops_conv_1d(ctx, w, x, stride, padding, dilation, 1, backend, b);
    if (!dst) {
        std::cerr << backend_name << " failed to build quantized custom convolution." << std::endl;
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    std::vector<ggml_fp16_t> input_f16;
    if (activation_type == GGML_TYPE_F16) {
        input_f16.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) input_f16[i] = ggml_fp32_to_fp16(input[i]);
        ggml_backend_tensor_set(x, input_f16.data(), 0, input_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    }
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS && activation_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> actual_f16(actual.size());
        ggml_backend_tensor_get(dst, actual_f16.data(), 0, actual_f16.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < actual.size(); ++i) actual[i] = ggml_fp16_to_fp32(actual_f16[i]);
    } else if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(dst, actual.data(), 0, actual.size() * sizeof(float));
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) max_diff = std::max(max_diff, std::abs(actual[i] - reference[i]));
    const float tolerance = activation_type == GGML_TYPE_F16 && qtype == GGML_TYPE_Q4_K
        ? 1.5e-2f
        : (activation_type == GGML_TYPE_F16 ? 4e-3f : 2e-3f);
    const bool passed = status == GGML_STATUS_SUCCESS && max_diff < tolerance;
    std::cout << backend_name << " " << ggml_type_name(qtype) << " "
              << (transpose ? "ConvT" : "Conv") << " " << ggml_type_name(activation_type)
              << ": max diff " << max_diff
              << (passed ? " PASSED" : " FAILED") << std::endl;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static bool run_small_group_convt_case(
    ggml_backend_t backend,
    const std::string& backend_name,
    ggml_type qtype,
    bool benchmark
) {
    constexpr int64_t kernel = 3;
    constexpr int64_t groups = 4;
    constexpr int64_t input_channels_per_group = 2;
    constexpr int64_t input_channels = groups * input_channels_per_group;
    const int64_t output_channels_per_group = qtype == GGML_TYPE_Q4_K ? 256 : 32;
    const int64_t output_channels = groups * output_channels_per_group;
    constexpr int64_t input_length = 32;
    constexpr int stride = 2;
    constexpr int padding = 1;
    constexpr int dilation = 1;
    constexpr int64_t output_length =
        (input_length - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1;
    constexpr int64_t weight_rows = kernel * input_channels;

    std::vector<float> weights(weight_rows * output_channels_per_group);
    std::vector<float> input(input_length * input_channels);
    std::vector<float> bias(output_channels);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * 0.019f) * 0.2f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.023f) * 0.15f;
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 7) - 3) * 0.01f;

    const std::vector<uint8_t> quantized =
        quantize_weights(qtype, weights, weight_rows, output_channels_per_group);
    if (quantized.empty()) return false;
    const std::vector<float> decoded =
        dequantize_weights(qtype, quantized, weight_rows, output_channels_per_group);
    std::vector<float> reference(output_length * output_channels);
    for (int64_t group = 0; group < groups; ++group) {
        for (int64_t local_oc = 0; local_oc < output_channels_per_group; ++local_oc) {
            const int64_t oc = group * output_channels_per_group + local_oc;
            for (int64_t ow = 0; ow < output_length; ++ow) {
                float sum = bias[oc];
                for (int64_t local_ic = 0; local_ic < input_channels_per_group; ++local_ic) {
                    const int64_t ic = group * input_channels_per_group + local_ic;
                    for (int64_t kw = 0; kw < kernel; ++kw) {
                        const int64_t numerator = ow + padding - kw * dilation;
                        if (numerator < 0 || numerator % stride != 0) continue;
                        const int64_t iw = numerator / stride;
                        if (iw < input_length) {
                            sum += input[iw + input_length * ic] *
                                decoded[local_oc + output_channels_per_group * (kw + kernel * ic)];
                        }
                    }
                }
                reference[ow + output_length * oc] = sum;
            }
        }
    }

    ggml_context* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    ggml_tensor* w = ggml_new_tensor_3d(
        ctx, qtype, output_channels_per_group, kernel, input_channels);
    ggml_tensor* x = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, input_length, input_channels, 1);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_channels);
    ggml_tensor* dst = ggml_ops_conv_transpose_1d(
        ctx, w, x, stride, padding, dilation, groups, backend, b);
    if (!dst) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);

    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(dst, actual.data(), 0, actual.size() * sizeof(float));
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(actual[i] - reference[i]));
    }
    const bool passed = status == GGML_STATUS_SUCCESS && max_diff < 2e-3f;
    std::cout << backend_name << " " << ggml_type_name(qtype)
              << " SmallGrouped ConvT f32: max diff " << max_diff
              << (passed ? " PASSED" : " FAILED") << std::endl;

    if (benchmark && passed) {
        constexpr int iterations = 30;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) break;
        }
        const auto end = std::chrono::steady_clock::now();
        const double milliseconds =
            std::chrono::duration<double, std::milli>(end - start).count() / iterations;
        std::cout << "BENCH " << backend_name << " " << ggml_type_name(qtype)
                  << " SmallGrouped ConvT " << milliseconds << " ms" << std::endl;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static bool run_benchmark_case(
    ggml_backend_t backend,
    const std::string& backend_name,
    ggml_type weight_type,
    bool transpose,
    ggml_type activation_type = GGML_TYPE_F32
) {
    constexpr int64_t kernel = 3;
    constexpr int64_t channels = 256;
    constexpr int64_t batch = 1;
    constexpr int64_t input_length = 64;
    const int stride = transpose ? 2 : 1;
    const int padding = 1;
    const int64_t output_length = transpose
        ? (input_length - 1) * stride - 2 * padding + kernel
        : input_length;
    const int64_t weight_rows = kernel * channels;

    std::vector<float> weights(weight_rows * channels);
    std::vector<float> input(input_length * channels * batch);
    std::vector<float> bias(channels);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * 0.013f) * 0.1f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.017f) * 0.1f;
    std::vector<uint8_t> quantized;
    std::vector<ggml_fp16_t> f16_weights;
    if (ggml_is_quantized(weight_type)) {
        quantized = quantize_weights(weight_type, weights, weight_rows, channels);
        if (quantized.empty()) return false;
    } else if (weight_type == GGML_TYPE_F16) {
        f16_weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) f16_weights[i] = ggml_fp32_to_fp16(weights[i]);
    }
    std::vector<ggml_fp16_t> f16_input;
    if (activation_type == GGML_TYPE_F16) {
        f16_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) f16_input[i] = ggml_fp32_to_fp16(input[i]);
    }

    ggml_context* ctx = ggml_init({16 * 1024 * 1024, nullptr, true});
    ggml_tensor* w = ggml_is_quantized(weight_type)
        ? ggml_new_tensor_3d(ctx, weight_type, channels, kernel, channels)
        : ggml_new_tensor_3d(ctx, weight_type, kernel, channels, channels);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, activation_type, input_length, channels, batch);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    ggml_tensor* dst = transpose
        ? ggml_ops_conv_transpose_1d(ctx, w, x, stride, padding, 1, 1, backend, b)
        : ggml_ops_conv_1d(ctx, w, x, stride, padding, 1, 1, backend, b);
    if (!dst || dst->ne[0] != output_length) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!quantized.empty()) ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    else if (!f16_weights.empty()) ggml_backend_tensor_set(w, f16_weights.data(), 0, f16_weights.size() * sizeof(ggml_fp16_t));
    else ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
    if (!f16_input.empty()) ggml_backend_tensor_set(x, f16_input.data(), 0, f16_input.size() * sizeof(ggml_fp16_t));
    else ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);

    if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    constexpr int iterations = 30;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    const auto end = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    std::cout << "BENCH " << backend_name << " " << ggml_type_name(weight_type)
              << "/" << ggml_type_name(activation_type) << " "
              << (transpose ? "ConvT" : "Conv") << " " << milliseconds << " ms" << std::endl;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
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
    bool passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = device ? ggml_backend_dev_name(device) : "";
        if (name.rfind("CPU", 0) != 0 && name.rfind("CUDA", 0) != 0 && name.rfind("SYCL", 0) != 0) continue;
        if (benchmark && !benchmark_device.empty() && name.rfind(benchmark_device, 0) != 0) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        if (name.rfind("CPU", 0) == 0) ggml_ops_ext_cpu_set_n_threads(backend, 4);
        if (benchmark) {
            for (ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K }) {
                passed &= run_benchmark_case(backend, name, type, false);
                passed &= run_benchmark_case(backend, name, type, true);
                passed &= run_small_group_convt_case(backend, name, type, true);
            }
            for (const auto& types : { std::pair{ GGML_TYPE_F32, GGML_TYPE_F32 },
                                       std::pair{ GGML_TYPE_F16, GGML_TYPE_F16 } }) {
                passed &= run_benchmark_case(backend, name, types.first, false, types.second);
                passed &= run_benchmark_case(backend, name, types.first, true, types.second);
            }
            ggml_backend_free(backend);
            continue;
        }
        for (ggml_type type : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 }) {
            passed &= run_case(backend, name, type, false);
            passed &= run_case(backend, name, type, true);
            passed &= run_case(backend, name, type, false, GGML_TYPE_F16);
            passed &= run_case(backend, name, type, true, GGML_TYPE_F16);
            passed &= run_small_group_convt_case(backend, name, type, false);
        }
        ggml_backend_free(backend);
    }

    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
