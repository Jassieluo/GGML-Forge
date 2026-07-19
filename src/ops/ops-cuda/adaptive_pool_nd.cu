#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ggml_ops_ext::cuda {
namespace {

struct tensor_strides {
    size_t values[4];
};

struct adaptive_window {
    int64_t begin;
    int64_t end;
};

template <typename T> __device__ float to_float(T value) { return static_cast<float>(value); }
template <> __device__ float to_float<half>(half value) { return __half2float(value); }
template <> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> __device__ T from_float(float value) { return static_cast<T>(value); }
template <> __device__ half from_float<half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

__device__ adaptive_window make_adaptive_window(int64_t output_index, int64_t input_size, int64_t output_size) {
    return {
        output_index * input_size / output_size,
        ((output_index + 1) * input_size + output_size - 1) / output_size,
    };
}

__device__ size_t spatial_offset(int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                                 const int64_t spatial_size[3], tensor_strides strides) {
    if (spatial_dims == 1) {
        return batch * strides.values[2] + channel * strides.values[1] + x * strides.values[0];
    }
    if (spatial_dims == 2) {
        return batch * strides.values[3] + channel * strides.values[2] + y * strides.values[1] + x * strides.values[0];
    }
    const int64_t spatial_index = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * strides.values[2] + channel * strides.values[1] + spatial_index * strides.values[0];
}

template <typename T>
__global__ void adaptive_pool_kernel(const T* input, T* output, ops_adaptive_pool_nd_desc desc,
                                     tensor_strides input_strides, tensor_strides output_strides) {
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
    const adaptive_window x_window = make_adaptive_window(output_x, desc.input_size[0], desc.output_size[0]);
    const adaptive_window y_window = make_adaptive_window(output_y, desc.input_size[1], desc.output_size[1]);
    const adaptive_window z_window = make_adaptive_window(output_z, desc.input_size[2], desc.output_size[2]);

    float result = desc.mode == ops_pool_mode::maximum ? -CUDART_INF_F : 0.0f;
    for (int64_t input_z = z_window.begin; input_z < z_window.end; ++input_z) {
        for (int64_t input_y = y_window.begin; input_y < y_window.end; ++input_y) {
            for (int64_t input_x = x_window.begin; input_x < x_window.end; ++input_x) {
                const char* address = reinterpret_cast<const char*>(input) +
                                      spatial_offset(desc.spatial_dims, input_x, input_y, input_z, channel, batch,
                                                     desc.input_size, input_strides);
                const float sample = to_float(*reinterpret_cast<const T*>(address));
                result = desc.mode == ops_pool_mode::maximum ? fmaxf(result, sample) : result + sample;
            }
        }
    }
    if (desc.mode == ops_pool_mode::average) {
        const int64_t sample_count =
            (x_window.end - x_window.begin) * (y_window.end - y_window.begin) * (z_window.end - z_window.begin);
        result /= static_cast<float>(sample_count);
    }
    char* output_address =
        reinterpret_cast<char*>(output) + spatial_offset(desc.spatial_dims, output_x, output_y, output_z, channel,
                                                         batch, desc.output_size, output_strides);
    *reinterpret_cast<T*>(output_address) = from_float<T>(result);
}

template <typename T, ops_pool_mode Mode>
__global__ void adaptive_global_pool_kernel(const T* input, T* output, int64_t input_volume, int64_t batch_channels) {
    const int64_t batch_channel = blockIdx.x;
    if (batch_channel >= batch_channels) return;
    float value = Mode == ops_pool_mode::maximum ? -CUDART_INF_F : 0.0f;
    const T* row = input + batch_channel * input_volume;
    for (int64_t index = threadIdx.x; index < input_volume; index += blockDim.x) {
        const float sample = to_float(row[index]);
        if constexpr (Mode == ops_pool_mode::maximum)
            value = fmaxf(value, sample);
        else
            value += sample;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if constexpr (Mode == ops_pool_mode::maximum) {
                partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + offset]);
            } else {
                partial[threadIdx.x] += partial[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float result = partial[0];
        if constexpr (Mode == ops_pool_mode::average) result /= static_cast<float>(input_volume);
        output[batch_channel] = from_float<T>(result);
    }
}

template <typename T>
bool launch_adaptive_pool(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output,
                          const ops_adaptive_pool_nd_desc& desc) {
    const int64_t input_volume = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    if (output_volume == 1 && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        const int64_t batch_channels = desc.batch * desc.channels;
        if (desc.mode == ops_pool_mode::maximum) {
            adaptive_global_pool_kernel<T, ops_pool_mode::maximum>
                <<<static_cast<unsigned>(batch_channels), 256, 0, stream>>>(
                    static_cast<const T*>(input->data), static_cast<T*>(output->data), input_volume, batch_channels);
        } else {
            adaptive_global_pool_kernel<T, ops_pool_mode::average>
                <<<static_cast<unsigned>(batch_channels), 256, 0, stream>>>(
                    static_cast<const T*>(input->data), static_cast<T*>(output->data), input_volume, batch_channels);
        }
        return cudaGetLastError() == cudaSuccess;
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
    adaptive_pool_kernel<T><<<static_cast<unsigned>((total_elements + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<T*>(output->data), desc, input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_adaptive_pool_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                             : op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                       : 3;
    ops_adaptive_pool_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {
        ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node,
    };
    ops_adaptive_pool_nd_desc desc;
    if (!ops_validate_adaptive_pool_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) {
        return false;
    }
    switch (node->type) {
    case GGML_TYPE_F32:
        return launch_adaptive_pool<float>(stream, sources[0], node, desc);
    case GGML_TYPE_F16:
        return launch_adaptive_pool<half>(stream, sources[0], node, desc);
    case GGML_TYPE_BF16:
        return launch_adaptive_pool<__nv_bfloat16>(stream, sources[0], node, desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::cuda
