#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>

namespace ggml_ops_ext::cuda {
namespace {

struct tensor_strides {
    size_t values[4];
};

struct linear_axis_sample {
    int64_t lower_index;
    int64_t upper_index;
    float upper_weight;
};

template <typename T> __device__ float to_float(T value) { return static_cast<float>(value); }
template <> __device__ float to_float<half>(half value) { return __half2float(value); }
template <> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> __device__ T from_float(float value) { return static_cast<T>(value); }
template <> __device__ half from_float<half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <int Dims>
__device__ size_t spatial_offset(int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
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

__device__ linear_axis_sample make_linear_axis_sample(int64_t output_index, int64_t input_size, int64_t output_size,
                                                      bool align_corners) {
    double source_position;
    if (align_corners && output_size > 1) {
        source_position = static_cast<double>(output_index) * (input_size - 1) / (output_size - 1);
    } else {
        source_position = (output_index + 0.5) * static_cast<double>(input_size) / output_size - 0.5;
    }
    source_position = fmax(0.0, fmin(source_position, static_cast<double>(input_size - 1)));
    const int64_t lower_index = static_cast<int64_t>(floor(source_position));
    return {
        lower_index,
        min(input_size - 1, lower_index + 1),
        static_cast<float>(source_position - lower_index),
    };
}

template <int Dims, ops_resize_mode Mode, typename T>
__global__ void resize_kernel(const T* input, T* output, ops_resize_nd_desc desc, tensor_strides input_strides,
                              tensor_strides output_strides) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t element_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
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
        const int64_t input_x = min(desc.input_size[0] - 1, output_x * desc.input_size[0] / desc.output_size[0]);
        const int64_t input_y = min(desc.input_size[1] - 1, output_y * desc.input_size[1] / desc.output_size[1]);
        const int64_t input_z = min(desc.input_size[2] - 1, output_z * desc.input_size[2] / desc.output_size[2]);
        const char* address =
            reinterpret_cast<const char*>(input) +
            spatial_offset<Dims>(input_x, input_y, input_z, channel, batch, desc.input_size, input_strides);
        result = to_float(*reinterpret_cast<const T*>(address));
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
                        input_coordinate[axis] = sides[axis] ? samples[axis].upper_index : samples[axis].lower_index;
                        weight *= sides[axis] ? samples[axis].upper_weight : 1.0f - samples[axis].upper_weight;
                    }
                    const char* address =
                        reinterpret_cast<const char*>(input) +
                        spatial_offset<Dims>(input_coordinate[0], input_coordinate[1], input_coordinate[2], channel,
                                             batch, desc.input_size, input_strides);
                    result += weight * to_float(*reinterpret_cast<const T*>(address));
                }
            }
        }
    }

    char* output_address =
        reinterpret_cast<char*>(output) +
        spatial_offset<Dims>(output_x, output_y, output_z, channel, batch, desc.output_size, output_strides);
    *reinterpret_cast<T*>(output_address) = from_float<T>(result);
}

template <int Dims, ops_resize_mode Mode, typename T>
bool launch_resize(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    bool equal_size = true;
    for (int axis = 0; axis < 3; ++axis)
        equal_size &= desc.input_size[axis] == desc.output_size[axis];
    if (equal_size && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        return cudaMemcpyAsync(output->data, input->data, ggml_nbytes(input), cudaMemcpyDeviceToDevice, stream) ==
               cudaSuccess;
    }
    const int64_t total_elements =
        desc.batch * desc.channels * desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
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
    resize_kernel<Dims, Mode, T><<<static_cast<unsigned>((total_elements + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<T*>(output->data), desc, input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

template <int Dims, typename T>
bool dispatch_mode(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    if (desc.mode == ops_resize_mode::nearest) {
        return launch_resize<Dims, ops_resize_mode::nearest, T>(stream, input, output, desc);
    }
    return launch_resize<Dims, ops_resize_mode::linear, T>(stream, input, output, desc);
}

template <typename T>
bool dispatch_dims(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output, const ops_resize_nd_desc& desc) {
    if (desc.spatial_dims == 1) return dispatch_mode<1, T>(stream, input, output, desc);
    if (desc.spatial_dims == 2) return dispatch_mode<2, T>(stream, input, output, desc);
    return dispatch_mode<3, T>(stream, input, output, desc);
}

} // namespace

bool ggml_cuda_op_resize_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
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
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) {
        return false;
    }
    switch (node->type) {
    case GGML_TYPE_F32:
        return dispatch_dims<float>(stream, sources[0], node, desc);
    case GGML_TYPE_F16:
        return dispatch_dims<half>(stream, sources[0], node, desc);
    case GGML_TYPE_BF16:
        return dispatch_dims<__nv_bfloat16>(stream, sources[0], node, desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::cuda
