#include "ops/ops.h"
#include "ops/cpu.h"

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

static int64_t volume(const int32_t size[3]) {
    return int64_t(size[0]) * size[1] * size[2];
}

static bool run_case(ggml_backend_t backend, const std::string& name, int dims, bool transposed,
                     ggml_type weight_type, int groups,
                     ggml_ops_ext::ops_weight_layout layout = ggml_ops_ext::ops_weight_layout::channel_rows,
                     bool implicit_geometry = false,
                     ggml_type activation_type = GGML_TYPE_F32,
                     int64_t kernel_edge = 2,
                     int transposed_stride = 2,
                     int64_t channels_per_group = 0,
                     int32_t spatial_edge = 0) {
    ggml_ops_ext::ops_conv_nd_config config;
    config.spatial_dims = dims;
    config.input_size[0] = spatial_edge > 0 ? spatial_edge : implicit_geometry ? 9 : 4;
    config.input_size[1] = spatial_edge > 0 ? spatial_edge : implicit_geometry ? 9 : 3;
    config.input_size[2] = dims == 3 ? (spatial_edge > 0 ? spatial_edge : 2) : 1;
    config.kernel_size[0] = kernel_edge;
    config.kernel_size[1] = kernel_edge;
    config.kernel_size[2] = dims == 3 ? kernel_edge : 1;
    config.stride[0] = transposed ? transposed_stride : 1;
    config.stride[1] = transposed ? transposed_stride : 1;
    config.padding_before[0] = kernel_edge > 1 ? 1 : 0;
    config.padding_after[0] = 0;
    config.output_padding[0] = transposed && transposed_stride > 1 ? 1 : 0;
    config.groups = groups;
    config.weight_layout = layout;
    const int64_t cin_group = channels_per_group > 0 ? channels_per_group
        : implicit_geometry ? (ggml_is_quantized(weight_type) ? 32 : 16)
                            : (ggml_is_quantized(weight_type) ? ggml_blck_size(weight_type) : 2);
    const int64_t cout_group = channels_per_group > 0 ? channels_per_group
        : implicit_geometry ? (ggml_is_quantized(weight_type) ? 32 : 16)
                            : (ggml_is_quantized(weight_type) && transposed ? ggml_blck_size(weight_type) : 3);
    const int64_t cin = cin_group * groups;
    const int64_t cout = cout_group * groups;
    const int64_t kernel_volume = volume(config.kernel_size);
    int32_t output_size[3];
    for (int axis = 0; axis < 3; ++axis) {
        const int64_t effective = config.dilation[axis] * (config.kernel_size[axis] - 1) + 1;
        output_size[axis] = static_cast<int32_t>(transposed
            ? (config.input_size[axis] - 1) * config.stride[axis] - config.padding_before[axis] -
                  config.padding_after[axis] + effective + config.output_padding[axis]
            : (config.input_size[axis] + config.padding_before[axis] + config.padding_after[axis] - effective) /
                  config.stride[axis] + 1);
    }
    const int64_t input_volume = volume(config.input_size);
    const int64_t output_volume = volume(output_size);
    const int64_t batch = 2;
    const int64_t row_channels = transposed ? cout_group : cin_group;
    const int64_t row_elements = layout == ggml_ops_ext::ops_weight_layout::flattened_rows
        ? row_channels * kernel_volume : row_channels;
    const int64_t outer = transposed ? cin : cout;
    const int64_t rows = layout == ggml_ops_ext::ops_weight_layout::flattened_rows
        ? outer : kernel_volume * outer;
    std::vector<float> weights(static_cast<size_t>(rows * row_elements));
    std::vector<float> input(static_cast<size_t>(input_volume * cin * batch));
    std::vector<float> bias(static_cast<size_t>(cout));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i + 1) * 0.071f) * 0.15f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i + 1) * 0.053f) * 0.2f;
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = float(int(i % 5) - 2) * 0.01f;

    std::vector<uint8_t> stored_weights;
    std::vector<ggml_fp16_t> stored_f16_weights;
    std::vector<ggml_bf16_t> stored_bf16_weights;
    std::vector<float> decoded = weights;
    if (ggml_is_quantized(weight_type)) {
        stored_weights.resize(ggml_row_size(weight_type, row_elements) * rows);
        if (ggml_quantize_chunk(weight_type, weights.data(), stored_weights.data(), 0, rows, row_elements, nullptr) != stored_weights.size()) return false;
        const ggml_type_traits* traits = ggml_get_type_traits(weight_type);
        for (int64_t row = 0; row < rows; ++row) {
            traits->to_float(stored_weights.data() + row * ggml_row_size(weight_type, row_elements),
                             decoded.data() + row * row_elements, row_elements);
        }
    } else if (weight_type == GGML_TYPE_F16) {
        stored_f16_weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            stored_f16_weights[i] = ggml_fp32_to_fp16(weights[i]);
            decoded[i] = ggml_fp16_to_fp32(stored_f16_weights[i]);
        }
    } else if (weight_type == GGML_TYPE_BF16) {
        stored_bf16_weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            stored_bf16_weights[i] = ggml_fp32_to_bf16(weights[i]);
            decoded[i] = ggml_bf16_to_fp32(stored_bf16_weights[i]);
        }
    }
    std::vector<ggml_fp16_t> stored_f16_input;
    std::vector<ggml_bf16_t> stored_bf16_input;
    if (activation_type == GGML_TYPE_F16) {
        stored_f16_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) stored_f16_input[i] = ggml_fp32_to_fp16(input[i]);
    } else if (activation_type == GGML_TYPE_BF16) {
        stored_bf16_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) stored_bf16_input[i] = ggml_fp32_to_bf16(input[i]);
    }

    std::vector<float> reference(static_cast<size_t>(output_volume * cout * batch));
    for (int64_t n = 0; n < batch; ++n) for (int64_t oc = 0; oc < cout; ++oc) {
        const int64_t group = oc / cout_group;
        const int64_t local_oc = oc % cout_group;
        for (int64_t out = 0; out < output_volume; ++out) {
            const int64_t ox = out % output_size[0];
            const int64_t or1 = out / output_size[0];
            const int64_t oy = or1 % output_size[1];
            const int64_t oz = or1 / output_size[1];
            float sum = bias[oc];
            for (int64_t k = 0; k < kernel_volume; ++k) {
                const int64_t kx = k % config.kernel_size[0];
                const int64_t kr1 = k / config.kernel_size[0];
                const int64_t ky = kr1 % config.kernel_size[1];
                const int64_t kz = kr1 / config.kernel_size[1];
                int64_t ix, iy, iz;
                if (!transposed) {
                    ix = ox * config.stride[0] - config.padding_before[0] + kx * config.dilation[0];
                    iy = oy * config.stride[1] - config.padding_before[1] + ky * config.dilation[1];
                    iz = oz * config.stride[2] - config.padding_before[2] + kz * config.dilation[2];
                } else {
                    const int64_t sx = ox + config.padding_before[0] - kx * config.dilation[0];
                    const int64_t sy = oy + config.padding_before[1] - ky * config.dilation[1];
                    const int64_t sz = oz + config.padding_before[2] - kz * config.dilation[2];
                    if (sx < 0 || sy < 0 || sz < 0 || sx % config.stride[0] || sy % config.stride[1] || sz % config.stride[2]) continue;
                    ix = sx / config.stride[0]; iy = sy / config.stride[1]; iz = sz / config.stride[2];
                }
                if (ix < 0 || ix >= config.input_size[0] || iy < 0 || iy >= config.input_size[1] ||
                    iz < 0 || iz >= config.input_size[2]) continue;
                const int64_t in_spatial = ix + config.input_size[0] * (iy + config.input_size[1] * iz);
                for (int64_t local_ic = 0; local_ic < cin_group; ++local_ic) {
                    const int64_t ic = group * cin_group + local_ic;
                    const int64_t row = layout == ggml_ops_ext::ops_weight_layout::flattened_rows
                        ? (transposed ? ic : oc)
                        : (transposed ? ic : oc) * kernel_volume + k;
                    const int64_t channel_inner = transposed ? local_oc : local_ic;
                    const int64_t inner = layout == ggml_ops_ext::ops_weight_layout::flattened_rows
                        ? k + kernel_volume * channel_inner : channel_inner;
                    sum += input[in_spatial + input_volume * (ic + cin * n)] * decoded[row * row_elements + inner];
                }
            }
            reference[out + output_volume * (oc + cout * n)] = sum;
        }
    }

    ggml_context* ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true });
    ggml_tensor* weight = layout == ggml_ops_ext::ops_weight_layout::flattened_rows
        ? ggml_new_tensor_2d(ctx, weight_type, row_elements, outer)
        : ggml_new_tensor_3d(ctx, weight_type, row_elements, kernel_volume, outer);
    ggml_tensor* x = dims == 2 ? ggml_new_tensor_4d(ctx, activation_type, config.input_size[0], config.input_size[1], cin, batch)
                               : ggml_new_tensor_3d(ctx, activation_type, input_volume, cin, batch);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cout);
    ggml_tensor* result = dims == 2
        ? (transposed ? ggml_ops_conv_transpose_2d(ctx, weight, x, config, backend, b)
                      : ggml_ops_conv_2d(ctx, weight, x, config, backend, b))
        : (transposed ? ggml_ops_conv_transpose_3d(ctx, weight, x, config, backend, b)
                      : ggml_ops_conv_3d(ctx, weight, x, config, backend, b));
    if (!result) {
        std::cerr << name << " Conv" << (transposed ? "Transpose" : "") << dims << "D "
                  << ggml_type_name(weight_type) << " input=" << ggml_type_name(activation_type)
                  << " graph creation FAILED\n";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!stored_weights.empty()) ggml_backend_tensor_set(weight, stored_weights.data(), 0, stored_weights.size());
    else if (!stored_f16_weights.empty()) ggml_backend_tensor_set(weight, stored_f16_weights.data(), 0, stored_f16_weights.size() * sizeof(ggml_fp16_t));
    else if (!stored_bf16_weights.empty()) ggml_backend_tensor_set(weight, stored_bf16_weights.data(), 0, stored_bf16_weights.size() * sizeof(ggml_bf16_t));
    else ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    if (!stored_f16_input.empty()) ggml_backend_tensor_set(x, stored_f16_input.data(), 0, stored_f16_input.size() * sizeof(ggml_fp16_t));
    else if (!stored_bf16_input.empty()) ggml_backend_tensor_set(x, stored_bf16_input.data(), 0, stored_bf16_input.size() * sizeof(ggml_bf16_t));
    else ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(reference.size());
    if (status == GGML_STATUS_SUCCESS && activation_type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    } else if (status == GGML_STATUS_SUCCESS && activation_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> actual_f16(actual.size());
        ggml_backend_tensor_get(result, actual_f16.data(), 0, actual_f16.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < actual.size(); ++i) actual[i] = ggml_fp16_to_fp32(actual_f16[i]);
    } else if (status == GGML_STATUS_SUCCESS) {
        std::vector<ggml_bf16_t> actual_bf16(actual.size());
        ggml_backend_tensor_get(result, actual_bf16.data(), 0, actual_bf16.size() * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < actual.size(); ++i) actual[i] = ggml_bf16_to_fp32(actual_bf16[i]);
    }
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) max_error = std::max(max_error, std::abs(actual[i] - reference[i]));
    const float tolerance = activation_type == GGML_TYPE_F16 || activation_type == GGML_TYPE_BF16 ? 2e-2f
        : (ggml_is_quantized(weight_type) ? 2e-3f : 1e-5f);
    const bool passed = status == GGML_STATUS_SUCCESS && max_error <= tolerance;
    std::cout << name << " Conv" << (transposed ? "Transpose" : "") << dims << "D "
              << ggml_type_name(weight_type) << " groups=" << groups << " error=" << max_error
              << " input=" << ggml_type_name(activation_type)
              << (layout == ggml_ops_ext::ops_weight_layout::flattened_rows ? " flattened" : " channel")
              << " status=" << static_cast<int>(status)
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

static bool run_benchmark(
    ggml_backend_t backend, const std::string& name, ggml_type weight_type,
    ggml_type activation_type, bool transposed,
    int64_t width = 32, int64_t height = 32, int64_t channels = 64,
    int iterations = 20, int64_t kernel = 3, int transposed_stride = 2,
    int dims = 2, int64_t depth = 1
) {
    const int64_t kernel_volume = kernel * kernel * (dims == 3 ? kernel : 1);
    const int64_t input_volume = width * height * (dims == 3 ? depth : 1);
    const int64_t row_elements = kernel_volume * channels;
    const int64_t weight_rows = channels;

    std::vector<float> weights(static_cast<size_t>(row_elements * weight_rows));
    std::vector<float> input(static_cast<size_t>(input_volume * channels));
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i + 1) * 0.013f) * 0.1f;
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i + 1) * 0.017f) * 0.1f;

    std::vector<uint8_t> stored_weights;
    std::vector<ggml_fp16_t> stored_f16_weights;
    std::vector<ggml_bf16_t> stored_bf16_weights;
    if (ggml_is_quantized(weight_type)) {
        stored_weights.resize(ggml_row_size(weight_type, row_elements) * weight_rows);
        if (ggml_quantize_chunk(weight_type, weights.data(), stored_weights.data(), 0,
                                weight_rows, row_elements, nullptr) != stored_weights.size()) return false;
    } else if (weight_type == GGML_TYPE_F16) {
        stored_f16_weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) stored_f16_weights[i] = ggml_fp32_to_fp16(weights[i]);
    } else if (weight_type == GGML_TYPE_BF16) {
        stored_bf16_weights.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) stored_bf16_weights[i] = ggml_fp32_to_bf16(weights[i]);
    }
    std::vector<ggml_fp16_t> stored_f16_input;
    std::vector<ggml_bf16_t> stored_bf16_input;
    if (activation_type == GGML_TYPE_F16) {
        stored_f16_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) stored_f16_input[i] = ggml_fp32_to_fp16(input[i]);
    } else if (activation_type == GGML_TYPE_BF16) {
        stored_bf16_input.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) stored_bf16_input[i] = ggml_fp32_to_bf16(input[i]);
    }

    ggml_context* ctx = ggml_init({ 8 * 1024 * 1024, nullptr, true });
    ggml_tensor* weight = ggml_new_tensor_2d(ctx, weight_type, row_elements, weight_rows);
    ggml_tensor* x = dims == 2
        ? ggml_new_tensor_4d(ctx, activation_type, width, height, channels, 1)
        : ggml_new_tensor_3d(ctx, activation_type, input_volume, channels, 1);
    ggml_ops_ext::ops_conv_nd_config config;
    config.spatial_dims = dims;
    config.input_size[0] = width;
    config.input_size[1] = height;
    config.input_size[2] = dims == 3 ? depth : 1;
    config.kernel_size[0] = kernel;
    config.kernel_size[1] = kernel;
    config.kernel_size[2] = dims == 3 ? kernel : 1;
    for (int axis = 0; axis < dims; ++axis) {
        config.stride[axis] = transposed ? transposed_stride : 1;
        config.padding_before[axis] = config.padding_after[axis] = kernel / 2;
    }
    config.weight_layout = ggml_ops_ext::ops_weight_layout::flattened_rows;
    ggml_tensor* result = dims == 2
        ? (transposed ? ggml_ops_conv_transpose_2d(ctx, weight, x, config, backend)
                      : ggml_ops_conv_2d(ctx, weight, x, config, backend))
        : (transposed ? ggml_ops_conv_transpose_3d(ctx, weight, x, config, backend)
                      : ggml_ops_conv_3d(ctx, weight, x, config, backend));
    if (!result) { ggml_free(ctx); return false; }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!stored_weights.empty()) ggml_backend_tensor_set(weight, stored_weights.data(), 0, stored_weights.size());
    else if (!stored_f16_weights.empty()) ggml_backend_tensor_set(weight, stored_f16_weights.data(), 0, stored_f16_weights.size() * sizeof(ggml_fp16_t));
    else if (!stored_bf16_weights.empty()) ggml_backend_tensor_set(weight, stored_bf16_weights.data(), 0, stored_bf16_weights.size() * sizeof(ggml_bf16_t));
    else ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    if (!stored_f16_input.empty()) ggml_backend_tensor_set(x, stored_f16_input.data(), 0, stored_f16_input.size() * sizeof(ggml_fp16_t));
    else if (!stored_bf16_input.empty()) ggml_backend_tensor_set(x, stored_bf16_input.data(), 0, stored_bf16_input.size() * sizeof(ggml_bf16_t));
    else ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    for (int i = 0; i < 3; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    ggml_backend_synchronize(backend);
    const double milliseconds = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / iterations;
    // A transposed convolution performs one kernel contribution per input
    // position; do not count stride-created output holes as dense MACs.
    const double operations = 2.0 * input_volume * channels * channels * kernel_volume;
    std::cout << "BENCH ConvND " << name << " weight=" << ggml_type_name(weight_type)
              << " input=" << ggml_type_name(activation_type)
              << " dims=" << dims << " shape=" << width << "x" << height;
    if (dims == 3) std::cout << "x" << depth;
    std::cout << "x" << channels << " k=" << kernel
              << (transposed ? " transpose " : " forward ")
              << milliseconds << " ms " << operations / (milliseconds * 1.0e6) << " GFLOP/s\n";
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
    bool ok = true;
    {
        ggml_context* ctx = ggml_init({ 1024 * 1024, nullptr, true });
        ggml_tensor* weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 8, 3);
        ggml_tensor* bad_volume = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 2, 1);
        ggml_ops_ext::ops_conv_nd_config config;
        config.spatial_dims = 3;
        config.input_size[0] = config.input_size[1] = config.input_size[2] = 2;
        config.kernel_size[0] = config.kernel_size[1] = config.kernel_size[2] = 2;
        ok = ggml_ops_conv_3d(ctx, weight, bad_volume, config, nullptr) == nullptr && ok;
        ggml_tensor* valid_volume = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 1);
        config.output_padding[0] = 1;
        ok = ggml_ops_conv_transpose_3d(ctx, weight, valid_volume, config, nullptr) == nullptr && ok;
        ggml_free(ctx);
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        if (benchmark && !benchmark_device.empty() && name.rfind(benchmark_device, 0) != 0) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        if (benchmark) {
            if (name.rfind("CPU", 0) == 0) ggml_ops_ext_cpu_set_n_threads(backend, 4);
            for (bool transposed : { false, true }) {
                ok = run_benchmark(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, transposed) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, transposed) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_1, GGML_TYPE_F32, transposed) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q5_0, GGML_TYPE_F32, transposed) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, transposed) && ok;
                if (name.rfind("CPU", 0) == 0 || name.rfind("CUDA", 0) == 0) {
                    ok = run_benchmark(backend, name, GGML_TYPE_BF16, GGML_TYPE_BF16, transposed) && ok;
                }
            }
            if (name.rfind("CUDA", 0) == 0 || name.rfind("SYCL", 0) == 0) {
                ok = run_benchmark(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, false,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, false,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, true,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_K, GGML_TYPE_F32, false,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_K, GGML_TYPE_F32, true,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q6_K, GGML_TYPE_F32, false,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q6_K, GGML_TYPE_F32, true,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, false,
                                   16, 16, 256, 10) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, true,
                                   16, 16, 256, 10) && ok;
            }
            ok = run_benchmark(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, false,
                               32, 32, 256, 10, 1) && ok;
            ok = run_benchmark(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, true,
                               32, 32, 256, 10, 1, 1) && ok;
            ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, false,
                               32, 32, 256, 10, 1) && ok;
            ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, true,
                               32, 32, 256, 10, 1, 1) && ok;
            if (name.rfind("CPU", 0) != 0) {
                ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, false,
                                   32, 32, 256, 10, 1) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, true,
                                   32, 32, 256, 10, 1, 1) && ok;
            }
            for (bool transposed : { false, true }) {
                ok = run_benchmark(backend, name, GGML_TYPE_F32, GGML_TYPE_F32, transposed,
                                   8, 8, 64, 10, 3, 2, 3, 8) && ok;
                ok = run_benchmark(backend, name, GGML_TYPE_Q4_0, GGML_TYPE_F32, transposed,
                                   8, 8, 64, 10, 3, 2, 3, 8) && ok;
                if (name.rfind("CPU", 0) != 0) {
                    ok = run_benchmark(backend, name, GGML_TYPE_F16, GGML_TYPE_F16, transposed,
                                       8, 8, 64, 10, 3, 2, 3, 8) && ok;
                }
                if (name.rfind("CPU", 0) == 0 || name.rfind("CUDA", 0) == 0) {
                    ok = run_benchmark(backend, name, GGML_TYPE_BF16, GGML_TYPE_BF16, transposed,
                                       8, 8, 64, 10, 3, 2, 3, 8) && ok;
                }
            }
            ggml_backend_free(backend);
            continue;
        }
        for (int dims : { 2, 3 }) for (bool transposed : { false, true }) {
            ok = run_case(backend, name, dims, transposed, GGML_TYPE_F32, 2) && ok;
            ok = run_case(backend, name, dims, transposed, GGML_TYPE_Q4_0, 2) && ok;
        }
        for (int dims : { 2, 3 }) for (bool transposed : { false, true }) {
            ok = run_case(backend, name, dims, transposed, GGML_TYPE_F32, 2,
                          ggml_ops_ext::ops_weight_layout::flattened_rows) && ok;
            ok = run_case(backend, name, dims, transposed, GGML_TYPE_Q4_0, 2,
                          ggml_ops_ext::ops_weight_layout::flattened_rows) && ok;
        }
        for (ggml_type type : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                                GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
            ok = run_case(backend, name, 2, false, type, 1) && ok;
        }
        for (ggml_type type : { GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4 }) {
            ok = run_case(backend, name, 2, false, type, 1) && ok;
        }
        for (ggml_type type : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                                GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 }) {
            ok = run_case(backend, name, 2, false, type, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_F32, 3) && ok;
        }
        for (ggml_type type : { GGML_TYPE_Q4_1, GGML_TYPE_Q5_0 }) {
            ok = run_case(backend, name, 2, true, type, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_F32, 3, 2) && ok;
        }
        ok = run_case(backend, name, 2, false, GGML_TYPE_Q4_K, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F32, 3, 2, 256, 10) && ok;
        ok = run_case(backend, name, 2, true, GGML_TYPE_Q6_K, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F32, 3, 2, 256, 10) && ok;
        for (ggml_type type : { GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                                GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS }) {
            ok = run_case(backend, name, 2, true, type, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, false,
                          GGML_TYPE_F32, 3, 2, 256) && ok;
        }
        ok = run_case(backend, name, 2, false, GGML_TYPE_F32, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_F32, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 1) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_F32, 2,
                      ggml_ops_ext::ops_weight_layout::channel_rows, true, GGML_TYPE_F32, 1) && ok;
        ok = run_case(backend, name, 3, false, GGML_TYPE_F32, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 1) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_Q4_0, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_F16, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F16) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_F16, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F16, 1) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_F16, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F16, 3, 2, 64) && ok;
        ok = run_case(backend, name, 2, true, GGML_TYPE_F16, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F16, 3, 2, 64) && ok;
        ok = run_case(backend, name, 3, true, GGML_TYPE_F32, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F32, 3, 2, 64, 4) && ok;
        ok = run_case(backend, name, 3, true, GGML_TYPE_Q4_0, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F32, 3, 2, 64, 4) && ok;
        ok = run_case(backend, name, 3, true, GGML_TYPE_F16, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                      GGML_TYPE_F16, 3, 2, 64, 4) && ok;
        if (name.rfind("CPU", 0) == 0 || name.rfind("CUDA", 0) == 0) {
            ok = run_case(backend, name, 2, false, GGML_TYPE_BF16, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_BF16, 3, 2, 64) && ok;
            ok = run_case(backend, name, 2, true, GGML_TYPE_BF16, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_BF16, 3, 2, 64) && ok;
            ok = run_case(backend, name, 3, false, GGML_TYPE_BF16, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_BF16, 3, 2, 64, 4) && ok;
            ok = run_case(backend, name, 3, true, GGML_TYPE_BF16, 1,
                          ggml_ops_ext::ops_weight_layout::flattened_rows, true,
                          GGML_TYPE_BF16, 3, 2, 64, 4) && ok;
        }
        ok = run_case(backend, name, 2, true, GGML_TYPE_F32, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 1, 1) && ok;
        ok = run_case(backend, name, 2, false, GGML_TYPE_Q4_0, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 1) && ok;
        ok = run_case(backend, name, 2, true, GGML_TYPE_Q4_0, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 1, 1) && ok;
        ok = run_case(backend, name, 2, true, GGML_TYPE_Q4_0, 1,
                      ggml_ops_ext::ops_weight_layout::flattened_rows, true, GGML_TYPE_F32, 3, 3) && ok;
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return ok ? 0 : 1;
}
