#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

struct linear_axis_sample {
    int64_t lower_index;
    int64_t upper_index;
    float upper_weight;
};

template <int Dims>
size_t spatial_offset(const ggml_tensor* tensor, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                      const int64_t spatial_size[3]) {
    if constexpr (Dims == 1) {
        return batch * tensor->nb[2] + channel * tensor->nb[1] + x * tensor->nb[0];
    }
    if constexpr (Dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    const int64_t spatial_index = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * tensor->nb[2] + channel * tensor->nb[1] + spatial_index * tensor->nb[0];
}

template <ggml_type Type> float load_value(const ggml_tensor* tensor, size_t byte_offset) {
    const char* address = static_cast<const char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(address);
    }
    if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <ggml_type Type> void store_value(ggml_tensor* tensor, size_t byte_offset, float value) {
    char* address = static_cast<char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(address) = value;
    } else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(address) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(address) = ggml_fp32_to_bf16(value);
    }
}

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

template <int Dims, ggml_type Type, ops_resize_mode Mode>
bool execute_resize(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                    const ops_resize_nd_desc& desc) {
    bool equal_size = true;
    for (int axis = 0; axis < 3; ++axis)
        equal_size &= desc.input_size[axis] == desc.output_size[axis];
    if (equal_size && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        std::memcpy(output->data, input->data, ggml_nbytes(input));
        return true;
    }
    std::vector<int64_t> nearest[3];
    std::vector<linear_axis_sample> linear[3];
    for (int axis = 0; axis < 3; ++axis) {
        if constexpr (Mode == ops_resize_mode::nearest) {
            nearest[axis].resize(static_cast<size_t>(desc.output_size[axis]));
            for (int64_t output_index = 0; output_index < desc.output_size[axis]; ++output_index) {
                nearest[axis][static_cast<size_t>(output_index)] =
                    std::min(desc.input_size[axis] - 1, output_index * desc.input_size[axis] / desc.output_size[axis]);
            }
        } else {
            linear[axis].resize(static_cast<size_t>(desc.output_size[axis]));
            for (int64_t output_index = 0; output_index < desc.output_size[axis]; ++output_index) {
                linear[axis][static_cast<size_t>(output_index)] = make_linear_axis_sample(
                    output_index, desc.input_size[axis], desc.output_size[axis], desc.align_corners);
            }
        }
    }
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const int thread_count = backend_thread_count(backend);

#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int64_t element_index = 0; element_index < total_elements; ++element_index) {
        const int64_t output_spatial = element_index % output_volume;
        const int64_t channel = (element_index / output_volume) % desc.channels;
        const int64_t batch = element_index / (output_volume * desc.channels);
        const int64_t output_x = output_spatial % desc.output_size[0];
        const int64_t output_remainder = output_spatial / desc.output_size[0];
        const int64_t output_y = output_remainder % desc.output_size[1];
        const int64_t output_z = output_remainder / desc.output_size[1];

        float result = 0.0f;
        if constexpr (Mode == ops_resize_mode::nearest) {
            const int64_t input_x = nearest[0][static_cast<size_t>(output_x)];
            const int64_t input_y = nearest[1][static_cast<size_t>(output_y)];
            const int64_t input_z = nearest[2][static_cast<size_t>(output_z)];
            result = load_value<Type>(
                input, spatial_offset<Dims>(input, input_x, input_y, input_z, channel, batch, desc.input_size));
        } else {
            const linear_axis_sample x_sample = linear[0][static_cast<size_t>(output_x)];
            const linear_axis_sample y_sample = linear[1][static_cast<size_t>(output_y)];
            const linear_axis_sample z_sample = linear[2][static_cast<size_t>(output_z)];
            const linear_axis_sample samples[3] = {x_sample, y_sample, z_sample};

            for (int z_side = 0; z_side < (Dims == 3 ? 2 : 1); ++z_side) {
                for (int y_side = 0; y_side < (Dims >= 2 ? 2 : 1); ++y_side) {
                    for (int x_side = 0; x_side < 2; ++x_side) {
                        const int sides[3] = {x_side, y_side, z_side};
                        int64_t input_coordinate[3] = {0, 0, 0};
                        float weight = 1.0f;
                        for (int axis = 0; axis < Dims; ++axis) {
                            input_coordinate[axis] =
                                sides[axis] ? samples[axis].upper_index : samples[axis].lower_index;
                            weight *= sides[axis] ? samples[axis].upper_weight : 1.0f - samples[axis].upper_weight;
                        }
                        result += weight *
                                  load_value<Type>(input, spatial_offset<Dims>(input, input_coordinate[0],
                                                                               input_coordinate[1], input_coordinate[2],
                                                                               channel, batch, desc.input_size));
                    }
                }
            }
        }

        store_value<Type>(output,
                          spatial_offset<Dims>(output, output_x, output_y, output_z, channel, batch, desc.output_size),
                          result);
    }
    return true;
}

template <int Dims, ggml_type Type>
bool dispatch_mode(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                   const ops_resize_nd_desc& desc) {
    if (desc.mode == ops_resize_mode::nearest) {
        return execute_resize<Dims, Type, ops_resize_mode::nearest>(backend, output, input, desc);
    }
    return execute_resize<Dims, Type, ops_resize_mode::linear>(backend, output, input, desc);
}

template <ggml_type Type>
bool dispatch_dims(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                   const ops_resize_nd_desc& desc) {
    if (desc.spatial_dims == 1) return dispatch_mode<1, Type>(backend, output, input, desc);
    if (desc.spatial_dims == 2) return dispatch_mode<2, Type>(backend, output, input, desc);
    return dispatch_mode<3, Type>(backend, output, input, desc);
}

} // namespace

bool ops_cpu_op_resize_nd(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_RESIZE_1D ? 1 : op == GGML_OP_OPS_VIRT_RESIZE_2D ? 2 : 3;
    ops_resize_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {
        ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node,
    };
    ops_resize_nd_desc desc;
    if (!ops_validate_resize_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    switch (node->type) {
    case GGML_TYPE_F32:
        return dispatch_dims<GGML_TYPE_F32>(backend, node, sources[0], desc);
    case GGML_TYPE_F16:
        return dispatch_dims<GGML_TYPE_F16>(backend, node, sources[0], desc);
    case GGML_TYPE_BF16:
        return dispatch_dims<GGML_TYPE_BF16>(backend, node, sources[0], desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::cpu
