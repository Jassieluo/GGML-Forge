#include "ops/ops.h"

#include <algorithm>
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

static int64_t volume(const int32_t size[3]) { return int64_t(size[0]) * size[1] * size[2]; }

static int64_t map_coordinate(int64_t coordinate, int64_t size, ggml_ops_ext::ops_pad_mode mode,
                              bool& valid) {
    if (coordinate >= 0 && coordinate < size) {
        return coordinate;
    }
    if (mode == ggml_ops_ext::ops_pad_mode::constant) {
        valid = false;
        return 0;
    }
    if (mode == ggml_ops_ext::ops_pad_mode::replicate) {
        return coordinate < 0 ? 0 : size - 1;
    }
    if (mode == ggml_ops_ext::ops_pad_mode::circular) {
        return (coordinate % size + size) % size;
    }
    return coordinate < 0 ? -coordinate : 2 * size - 2 - coordinate;
}

static bool run_case(ggml_backend_t backend, const std::string& backend_name, int dims,
                     ggml_ops_ext::ops_pad_mode mode) {
    ggml_ops_ext::ops_pad_nd_config config;
    config.spatial_dims = dims;
    config.mode = mode;
    config.value = -0.375f;
    config.input_size[0] = 5;
    config.input_size[1] = dims >= 2 ? 4 : 1;
    config.input_size[2] = dims >= 3 ? 3 : 1;
    config.padding_before[0] = 2;
    config.padding_after[0] = 1;
    config.padding_before[1] = dims >= 2 ? 1 : 0;
    config.padding_after[1] = dims >= 2 ? 2 : 0;
    config.padding_before[2] = dims >= 3 ? 1 : 0;
    config.padding_after[2] = dims >= 3 ? 1 : 0;
    int32_t output_size[3];
    for (int axis = 0; axis < 3; ++axis) {
        output_size[axis] =
            config.input_size[axis] + config.padding_before[axis] + config.padding_after[axis];
    }
    const int64_t input_volume = volume(config.input_size);
    const int64_t output_volume = volume(output_size);
    const int64_t channels = 2;
    const int64_t batch = 2;
    std::vector<float> input(static_cast<size_t>(input_volume * channels * batch));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = std::sin(float(i + 1) * 0.19f);
    }
    std::vector<float> reference(static_cast<size_t>(output_volume * channels * batch));
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t output_spatial = 0; output_spatial < output_volume; ++output_spatial) {
                const int64_t output_x = output_spatial % output_size[0];
                const int64_t output_remainder = output_spatial / output_size[0];
                const int64_t output_y = output_remainder % output_size[1];
                const int64_t output_z = output_remainder / output_size[1];
                bool valid = true;
                const int64_t input_x = map_coordinate(output_x - config.padding_before[0],
                                                       config.input_size[0], mode, valid);
                const int64_t input_y = map_coordinate(output_y - config.padding_before[1],
                                                       config.input_size[1], mode, valid);
                const int64_t input_z = map_coordinate(output_z - config.padding_before[2],
                                                       config.input_size[2], mode, valid);
                reference[output_spatial + output_volume * (channel + channels * batch_index)] =
                    valid
                        ? input[input_x +
                                config.input_size[0] * (input_y + config.input_size[1] * input_z) +
                                input_volume * (channel + channels * batch_index)]
                        : config.value;
            }
        }
    }
    ggml_context* context = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_tensor* input_tensor =
        dims == 1
            ? ggml_new_tensor_3d(context, GGML_TYPE_F32, config.input_size[0], channels, batch)
        : dims == 2 ? ggml_new_tensor_4d(context, GGML_TYPE_F32, config.input_size[0],
                                         config.input_size[1], channels, batch)
                    : ggml_new_tensor_3d(context, GGML_TYPE_F32, input_volume, channels, batch);
    ggml_tensor* output_tensor = dims == 1 ? ggml_ops_pad_1d(context, input_tensor, config, backend)
                                 : dims == 2
                                     ? ggml_ops_pad_2d(context, input_tensor, config, backend)
                                     : ggml_ops_pad_3d(context, input_tensor, config, backend);
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
    const bool passed = status == GGML_STATUS_SUCCESS && maximum_error < 1e-6f;
    std::cout << backend_name << " pad" << dims << "d mode=" << static_cast<int>(mode)
              << " error=" << maximum_error << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return passed;
}

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
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) {
            continue;
        }
        for (int dims = 1; dims <= 3; ++dims) {
            for (int mode = 0; mode <= 3; ++mode) {
                all_passed &=
                    run_case(backend, name, dims, static_cast<ggml_ops_ext::ops_pad_mode>(mode));
            }
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return all_passed ? 0 : 1;
}
