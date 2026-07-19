#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

namespace ggml_ops_ext::sycl {
namespace {

struct tensor_strides {
    size_t values[4];
};

struct linear_axis_sample {
    int64_t lower_index;
    int64_t upper_index;
    float upper_weight;
};

template <int Dims, ops_resize_mode Mode, typename T> class ResizeNDKernel;

template <int Dims>
inline size_t spatial_offset(int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                             const int64_t spatial_size[3], tensor_strides strides) {
    if constexpr (Dims == 1) {
        return batch * strides.values[2] + channel * strides.values[1] + x * strides.values[0];
    }
    if constexpr (Dims == 2) {
        return batch * strides.values[3] + channel * strides.values[2] + y * strides.values[1] + x * strides.values[0];
    }
    const int64_t spatial_index = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * strides.values[2] + channel * strides.values[1] + spatial_index * strides.values[0];
}

inline linear_axis_sample make_linear_axis_sample(int64_t output_index, int64_t input_size, int64_t output_size,
                                                  bool align_corners) {
    float source_position;
    if (align_corners && output_size > 1) {
        source_position =
            static_cast<float>(output_index) * static_cast<float>(input_size - 1) / static_cast<float>(output_size - 1);
    } else {
        source_position = (static_cast<float>(output_index) + 0.5f) * static_cast<float>(input_size) /
                              static_cast<float>(output_size) -
                          0.5f;
    }
    source_position = ::sycl::fmax(0.0f, ::sycl::fmin(source_position, static_cast<float>(input_size - 1)));
    const int64_t lower_index = static_cast<int64_t>(::sycl::floor(source_position));
    return {
        lower_index,
        ::sycl::min(input_size - 1, lower_index + 1),
        static_cast<float>(source_position - lower_index),
    };
}

template <int Dims, ops_resize_mode Mode, typename T>
bool launch_resize(::sycl::queue* queue, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    bool equal_size = true;
    for (int axis = 0; axis < 3; ++axis)
        equal_size &= desc.input_size[axis] == desc.output_size[axis];
    if (equal_size && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        queue->memcpy(output->data, input->data, ggml_nbytes(input));
        return true;
    }
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const size_t local_size = 256;
    const size_t global_size = static_cast<size_t>((total_elements + local_size - 1) / local_size) * local_size;
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);
    const tensor_strides input_strides = {{
        input->nb[0],
        input->nb[1],
        input->nb[2],
        input->nb[3],
    }};
    const tensor_strides output_strides = {{
        output->nb[0],
        output->nb[1],
        output->nb[2],
        output->nb[3],
    }};

    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<ResizeNDKernel<Dims, Mode, T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int64_t element_index = item.get_global_linear_id();
                if (element_index >= total_elements) {
                    return;
                }
                const int64_t output_spatial = element_index % output_volume;
                const int64_t channel = (element_index / output_volume) % desc.channels;
                const int64_t batch = element_index / (output_volume * desc.channels);
                const int64_t output_x = output_spatial % desc.output_size[0];
                const int64_t output_remainder = output_spatial / desc.output_size[0];
                const int64_t output_y = output_remainder % desc.output_size[1];
                const int64_t output_z = output_remainder / desc.output_size[1];

                float result = 0.0f;
                if constexpr (Mode == ops_resize_mode::nearest) {
                    const int64_t input_x =
                        ::sycl::min(desc.input_size[0] - 1, output_x * desc.input_size[0] / desc.output_size[0]);
                    const int64_t input_y =
                        ::sycl::min(desc.input_size[1] - 1, output_y * desc.input_size[1] / desc.output_size[1]);
                    const int64_t input_z =
                        ::sycl::min(desc.input_size[2] - 1, output_z * desc.input_size[2] / desc.output_size[2]);
                    const char* address =
                        reinterpret_cast<const char*>(source) +
                        spatial_offset<Dims>(input_x, input_y, input_z, channel, batch, desc.input_size, input_strides);
                    result = static_cast<float>(*reinterpret_cast<const T*>(address));
                } else {
                    const linear_axis_sample samples[3] = {
                        make_linear_axis_sample(output_x, desc.input_size[0], desc.output_size[0], desc.align_corners),
                        make_linear_axis_sample(output_y, desc.input_size[1], desc.output_size[1], desc.align_corners),
                        make_linear_axis_sample(output_z, desc.input_size[2], desc.output_size[2], desc.align_corners),
                    };
                    for (int z_side = 0; z_side < (Dims == 3 ? 2 : 1); ++z_side) {
                        for (int y_side = 0; y_side < (Dims >= 2 ? 2 : 1); ++y_side) {
                            for (int x_side = 0; x_side < 2; ++x_side) {
                                const int sides[3] = {x_side, y_side, z_side};
                                int64_t input_coordinate[3] = {0, 0, 0};
                                float weight = 1.0f;
                                for (int axis = 0; axis < Dims; ++axis) {
                                    input_coordinate[axis] =
                                        sides[axis] ? samples[axis].upper_index : samples[axis].lower_index;
                                    weight *=
                                        sides[axis] ? samples[axis].upper_weight : 1.0f - samples[axis].upper_weight;
                                }
                                const char* address =
                                    reinterpret_cast<const char*>(source) +
                                    spatial_offset<Dims>(input_coordinate[0], input_coordinate[1], input_coordinate[2],
                                                         channel, batch, desc.input_size, input_strides);
                                result += weight * static_cast<float>(*reinterpret_cast<const T*>(address));
                            }
                        }
                    }
                }

                char* output_address = reinterpret_cast<char*>(destination) +
                                       spatial_offset<Dims>(output_x, output_y, output_z, channel, batch,
                                                            desc.output_size, output_strides);
                *reinterpret_cast<T*>(output_address) = static_cast<T>(result);
            });
    });
    return true;
}

template <int Dims, typename T>
bool dispatch_mode(::sycl::queue* queue, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    if (desc.mode == ops_resize_mode::nearest) {
        return launch_resize<Dims, ops_resize_mode::nearest, T>(queue, input, output, desc);
    }
    return launch_resize<Dims, ops_resize_mode::linear, T>(queue, input, output, desc);
}

template <typename T>
bool dispatch_dims(::sycl::queue* queue, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    if (desc.spatial_dims == 1) return dispatch_mode<1, T>(queue, input, output, desc);
    if (desc.spatial_dims == 2) return dispatch_mode<2, T>(queue, input, output, desc);
    return dispatch_mode<3, T>(queue, input, output, desc);
}

} // namespace

bool ggml_sycl_op_resize_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
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
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }
    switch (node->type) {
    case GGML_TYPE_F32:
        return dispatch_dims<float>(queue, sources[0], node, desc);
    case GGML_TYPE_F16:
        return dispatch_dims<::sycl::half>(queue, sources[0], node, desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::sycl
