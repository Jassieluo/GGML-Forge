#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <type_traits>

namespace ggml_ops_ext::cuda {
namespace {

template <typename T> __device__ float pool_to_float(T value) { return static_cast<float>(value); }
template <> __device__ float pool_to_float<half>(half value) { return __half2float(value); }
template <> __device__ float pool_to_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ T pool_from_float(float value) { return static_cast<T>(value); }
template <> __device__ half pool_from_float<half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 pool_from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

struct tensor_strides {
    size_t x;
    size_t y;
    size_t channel_or_batch;
    size_t batch;
};

template <int Dims>
__device__ size_t spatial_offset(int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                                 const int64_t spatial_size[3], tensor_strides strides) {
    if constexpr (Dims == 1) {
        return batch * strides.channel_or_batch + channel * strides.y + x * strides.x;
    }
    if constexpr (Dims == 2) {
        return batch * strides.batch + channel * strides.channel_or_batch + y * strides.y + x * strides.x;
    }
    return batch * strides.channel_or_batch + channel * strides.y +
           (x + spatial_size[0] * (y + spatial_size[1] * z)) * strides.x;
}

struct axis_window {
    int first;
    int count;
};

__device__ axis_window clip_axis_window(int64_t base, int kernel_size, int dilation, int64_t lower, int64_t upper) {
    int first = 0;
    while (first < kernel_size && base + int64_t(first) * dilation < lower)
        ++first;
    int end = kernel_size;
    while (end > first && base + int64_t(end - 1) * dilation >= upper)
        --end;
    return {first, end - first};
}

template <int Dims, ops_pool_mode Mode, typename T>
__global__ void pool_nd_kernel(const T* input, T* output, ops_pool_nd_encoded_params params, ops_pool_nd_desc desc,
                               tensor_strides input_strides, tensor_strides output_strides) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    if (index >= total_elements) {
        return;
    }
    const int64_t output_spatial = index % output_volume;
    const int64_t channel = (index / output_volume) % desc.channels;
    const int64_t batch = index / (output_volume * desc.channels);
    const int64_t output_x = output_spatial % desc.output_size[0];
    const int64_t output_remainder = output_spatial / desc.output_size[0];
    const int64_t output_y = output_remainder % desc.output_size[1];
    const int64_t output_z = output_remainder / desc.output_size[1];
    const int64_t base_x = output_x * params.stride[0] - params.padding_before[0];
    const int64_t base_y = output_y * params.stride[1] - params.padding_before[1];
    const int64_t base_z = output_z * params.stride[2] - params.padding_before[2];
    const axis_window valid_x =
        clip_axis_window(base_x, desc.kernel_size[0], params.dilation[0], 0, desc.input_size[0]);
    const axis_window valid_y =
        clip_axis_window(base_y, desc.kernel_size[1], params.dilation[1], 0, desc.input_size[1]);
    const axis_window valid_z =
        clip_axis_window(base_z, desc.kernel_size[2], params.dilation[2], 0, desc.input_size[2]);
    float accumulator = Mode == ops_pool_mode::maximum ? -CUDART_INF_F : 0.0f;
    for (int kernel_z = 0; kernel_z < valid_z.count; ++kernel_z) {
        const int64_t input_z = base_z + int64_t(valid_z.first + kernel_z) * params.dilation[2];
        for (int kernel_y = 0; kernel_y < valid_y.count; ++kernel_y) {
            const int64_t input_y = base_y + int64_t(valid_y.first + kernel_y) * params.dilation[1];
            for (int kernel_x = 0; kernel_x < valid_x.count; ++kernel_x) {
                const int64_t input_x = base_x + int64_t(valid_x.first + kernel_x) * params.dilation[0];
                const char* address =
                    reinterpret_cast<const char*>(input) +
                    spatial_offset<Dims>(input_x, input_y, input_z, channel, batch, desc.input_size, input_strides);
                const float value = pool_to_float(*reinterpret_cast<const T*>(address));
                if constexpr (Mode == ops_pool_mode::maximum)
                    accumulator = fmaxf(accumulator, value);
                else
                    accumulator += value;
            }
        }
    }
    if constexpr (Mode == ops_pool_mode::average) {
        int64_t denominator = int64_t(valid_x.count) * valid_y.count * valid_z.count;
        if (desc.count_include_pad) {
            const axis_window padded_x =
                clip_axis_window(base_x, desc.kernel_size[0], params.dilation[0], -params.padding_before[0],
                                 desc.input_size[0] + params.padding_after[0]);
            const axis_window padded_y =
                clip_axis_window(base_y, desc.kernel_size[1], params.dilation[1], -params.padding_before[1],
                                 desc.input_size[1] + params.padding_after[1]);
            const axis_window padded_z =
                clip_axis_window(base_z, desc.kernel_size[2], params.dilation[2], -params.padding_before[2],
                                 desc.input_size[2] + params.padding_after[2]);
            denominator = int64_t(padded_x.count) * padded_y.count * padded_z.count;
        }
        accumulator = denominator > 0 ? accumulator / denominator : 0.0f;
    }
    char* address = reinterpret_cast<char*>(output) + spatial_offset<Dims>(output_x, output_y, output_z, channel, batch,
                                                                           desc.output_size, output_strides);
    *reinterpret_cast<T*>(address) = pool_from_float<T>(accumulator);
}

template <int Dims, ops_pool_mode Mode, typename T>
bool launch_pool(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output, const ops_pool_nd_encoded_params& params,
                 const ops_pool_nd_desc& desc) {
    const int64_t total_elements =
        desc.batch * desc.channels * desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const tensor_strides input_strides = {input->nb[0], input->nb[1], input->nb[2], input->nb[3]};
    const tensor_strides output_strides = {output->nb[0], output->nb[1], output->nb[2], output->nb[3]};
    pool_nd_kernel<Dims, Mode, T><<<static_cast<unsigned>((total_elements + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<T*>(output->data), params, desc, input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

template <int Dims, typename T>
bool dispatch_mode(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output,
                   const ops_pool_nd_encoded_params& params, const ops_pool_nd_desc& desc) {
    if (desc.mode == ops_pool_mode::maximum) {
        return launch_pool<Dims, ops_pool_mode::maximum, T>(stream, input, output, params, desc);
    }
    return launch_pool<Dims, ops_pool_mode::average, T>(stream, input, output, params, desc);
}

template <typename T>
bool dispatch_dims(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output,
                   const ops_pool_nd_encoded_params& params, const ops_pool_nd_desc& desc) {
    if (desc.spatial_dims == 1) return dispatch_mode<1, T>(stream, input, output, params, desc);
    if (desc.spatial_dims == 2) return dispatch_mode<2, T>(stream, input, output, params, desc);
    return dispatch_mode<3, T>(stream, input, output, params, desc);
}

} // namespace

bool ggml_cuda_op_pool_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_POOL_1D ? 1 : op == GGML_OP_OPS_VIRT_POOL_2D ? 2 : 3;
    ops_pool_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node};
    ops_pool_nd_desc desc;
    if (!ops_validate_pool_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) {
        return false;
    }
    if (node->type == GGML_TYPE_F32) {
        return dispatch_dims<float>(stream, node->src[0], node, params, desc);
    }
    if (node->type == GGML_TYPE_F16) {
        return dispatch_dims<half>(stream, node->src[0], node, params, desc);
    }
    if (node->type == GGML_TYPE_BF16) {
        return dispatch_dims<__nv_bfloat16>(stream, node->src[0], node, params, desc);
    }
    return false;
}

} // namespace ggml_ops_ext::cuda
