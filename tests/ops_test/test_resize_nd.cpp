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

namespace {

struct linear_axis_sample {
    int64_t lower_index;
    int64_t upper_index;
    float upper_weight;
};

int64_t spatial_volume(const int32_t size[3]) { return int64_t(size[0]) * size[1] * size[2]; }

linear_axis_sample make_linear_axis_sample(int64_t output_index, int64_t input_size, int64_t output_size,
                                           bool align_corners) {
    double source_position;
    if (align_corners && output_size > 1) {
        source_position = static_cast<double>(output_index) * (input_size - 1) / (output_size - 1);
    } else {
        source_position = (output_index + 0.5) * static_cast<double>(input_size) / output_size - 0.5;
    }
    source_position = std::clamp(source_position, 0.0, static_cast<double>(input_size - 1));
    const int64_t lower_index = static_cast<int64_t>(std::floor(source_position));
    return {
        lower_index,
        std::min(input_size - 1, lower_index + 1),
        static_cast<float>(source_position - lower_index),
    };
}

std::vector<float> make_reference(const std::vector<float>& input, const ggml_ops_ext::ops_resize_nd_config& config,
                                  int64_t channels, int64_t batch) {
    const int64_t input_volume = spatial_volume(config.input_size);
    const int64_t output_volume = spatial_volume(config.output_size);
    std::vector<float> output(static_cast<size_t>(output_volume * channels * batch));

    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t output_spatial = 0; output_spatial < output_volume; ++output_spatial) {
                const int64_t output_x = output_spatial % config.output_size[0];
                const int64_t output_remainder = output_spatial / config.output_size[0];
                const int64_t output_y = output_remainder % config.output_size[1];
                const int64_t output_z = output_remainder / config.output_size[1];
                float result = 0.0f;

                if (config.mode == ggml_ops_ext::ops_resize_mode::nearest) {
                    const int64_t input_x = std::min<int64_t>(config.input_size[0] - 1,
                                                              output_x * config.input_size[0] / config.output_size[0]);
                    const int64_t input_y = std::min<int64_t>(config.input_size[1] - 1,
                                                              output_y * config.input_size[1] / config.output_size[1]);
                    const int64_t input_z = std::min<int64_t>(config.input_size[2] - 1,
                                                              output_z * config.input_size[2] / config.output_size[2]);
                    const int64_t input_spatial =
                        input_x + config.input_size[0] * (input_y + config.input_size[1] * input_z);
                    result = input[input_spatial + input_volume * (channel + channels * batch_index)];
                } else {
                    const linear_axis_sample samples[3] = {
                        make_linear_axis_sample(output_x, config.input_size[0], config.output_size[0],
                                                config.align_corners),
                        make_linear_axis_sample(output_y, config.input_size[1], config.output_size[1],
                                                config.align_corners),
                        make_linear_axis_sample(output_z, config.input_size[2], config.output_size[2],
                                                config.align_corners),
                    };
                    for (int z_side = 0; z_side < 2; ++z_side) {
                        for (int y_side = 0; y_side < 2; ++y_side) {
                            for (int x_side = 0; x_side < 2; ++x_side) {
                                const int sides[3] = {x_side, y_side, z_side};
                                int64_t input_coordinate[3];
                                float weight = 1.0f;
                                for (int axis = 0; axis < 3; ++axis) {
                                    input_coordinate[axis] =
                                        sides[axis] ? samples[axis].upper_index : samples[axis].lower_index;
                                    weight *=
                                        sides[axis] ? samples[axis].upper_weight : 1.0f - samples[axis].upper_weight;
                                }
                                const int64_t input_spatial =
                                    input_coordinate[0] +
                                    config.input_size[0] *
                                        (input_coordinate[1] + config.input_size[1] * input_coordinate[2]);
                                result +=
                                    weight * input[input_spatial + input_volume * (channel + channels * batch_index)];
                            }
                        }
                    }
                }
                output[output_spatial + output_volume * (channel + channels * batch_index)] = result;
            }
        }
    }
    return output;
}

bool run_case(ggml_backend_t backend, const std::string& backend_name, int spatial_dims,
              ggml_ops_ext::ops_resize_mode mode, bool align_corners) {
    ggml_ops_ext::ops_resize_nd_config config;
    config.spatial_dims = spatial_dims;
    config.mode = mode;
    config.align_corners = align_corners;
    config.input_size[0] = 5;
    config.input_size[1] = spatial_dims >= 2 ? 4 : 1;
    config.input_size[2] = spatial_dims >= 3 ? 3 : 1;
    config.output_size[0] = 8;
    config.output_size[1] = spatial_dims >= 2 ? 3 : 1;
    config.output_size[2] = spatial_dims >= 3 ? 5 : 1;

    const int64_t input_volume = spatial_volume(config.input_size);
    const int64_t channels = 2;
    const int64_t batch = 2;
    std::vector<float> input(static_cast<size_t>(input_volume * channels * batch));
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = std::sin(float(index + 2) * 0.113f);
    }
    const std::vector<float> reference = make_reference(input, config, channels, batch);

    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor =
        spatial_dims == 1 ? ggml_new_tensor_3d(context, GGML_TYPE_F32, config.input_size[0], channels, batch)
        : spatial_dims == 2
            ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0], config.input_size[1], channels, batch)
            : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, batch);
    ggml_tensor* output_tensor = spatial_dims == 1   ? ggml_ops_resize_1d(context, input_tensor, config, backend)
                                 : spatial_dims == 2 ? ggml_ops_resize_2d(context, input_tensor, config, backend)
                                                     : ggml_ops_resize_3d(context, input_tensor, config, backend);
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
    for (size_t index = 0; index < actual.size(); ++index) {
        maximum_error = std::max(maximum_error, std::abs(actual[index] - reference[index]));
    }
    const bool passed = status == GGML_STATUS_SUCCESS && maximum_error < 3e-5f;
    std::cout << backend_name << " resize" << spatial_dims << "d "
              << (mode == ggml_ops_ext::ops_resize_mode::nearest ? "nearest" : "linear")
              << " align_corners=" << align_corners << " error=" << maximum_error
              << (passed ? " PASSED\n" : " FAILED\n");

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
}

bool run_benchmark(ggml_backend_t backend, const std::string& name, int dims, ggml_ops_ext::ops_resize_mode mode,
                   bool equal_size = false) {
    ggml_ops_ext::ops_resize_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.input_size[0] = dims == 2 ? 128 : 24;
    config.input_size[1] = dims == 2 ? 128 : 24;
    config.input_size[2] = dims == 3 ? 24 : 1;
    config.output_size[0] = equal_size ? config.input_size[0] : (dims == 2 ? 256 : 36);
    config.output_size[1] = equal_size ? config.input_size[1] : (dims == 2 ? 256 : 36);
    config.output_size[2] = equal_size ? config.input_size[2] : (dims == 3 ? 36 : 1);
    const int64_t channels = dims == 2 ? 32 : 8;
    const int64_t input_volume = spatial_volume(config.input_size);
    std::vector<float> input(static_cast<size_t>(input_volume * channels), 0.25f);
    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* x =
        dims == 2 ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0], config.input_size[1], channels, 1)
                  : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, 1);
    ggml_tensor* y =
        dims == 2 ? ggml_ops_resize_2d(context, x, config, backend) : ggml_ops_resize_3d(context, x, config, backend);
    if (!y) {
        ggml_free(context);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, y);
    if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    constexpr int iterations = 50;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / iterations;
    std::cout << "BENCH ResizeND " << name << " dims=" << dims << " "
              << (mode == ggml_ops_ext::ops_resize_mode::nearest ? "nearest" : "linear")
              << (equal_size ? " equal " : " ") << ms << " ms\n";
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return true;
}

} // namespace

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
    for (size_t device_index = 0; device_index < ggml_backend_dev_count(); ++device_index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(device_index);
        const std::string backend_name = device ? ggml_backend_dev_name(device) : "";
        if (backend_name.rfind("CPU", 0) && backend_name.rfind("CUDA", 0) && backend_name.rfind("SYCL", 0)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        if (benchmark && !filter.empty() && backend_name.rfind(filter, 0) != 0) {
            ggml_backend_free(backend);
            continue;
        }
        if (benchmark) {
            for (int dims = 2; dims <= 3; ++dims) {
                passed &= run_benchmark(backend, backend_name, dims, ggml_ops_ext::ops_resize_mode::nearest);
                passed &= run_benchmark(backend, backend_name, dims, ggml_ops_ext::ops_resize_mode::linear);
                passed &= run_benchmark(backend, backend_name, dims, ggml_ops_ext::ops_resize_mode::nearest, true);
            }
            ggml_backend_free(backend);
            continue;
        }
        for (int spatial_dims = 1; spatial_dims <= 3; ++spatial_dims) {
            passed &= run_case(backend, backend_name, spatial_dims, ggml_ops_ext::ops_resize_mode::nearest, false);
            passed &= run_case(backend, backend_name, spatial_dims, ggml_ops_ext::ops_resize_mode::linear, false);
            passed &= run_case(backend, backend_name, spatial_dims, ggml_ops_ext::ops_resize_mode::linear, true);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
