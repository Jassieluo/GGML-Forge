#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ggml_ops_ext::cuda {
namespace {

struct tensor_strides { size_t values[4]; };

template <typename T> __device__ float to_float(T value) { return static_cast<float>(value); }
template <> __device__ float to_float<half>(half value) { return __half2float(value); }
template <> __device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ T from_float(float value) { return static_cast<T>(value); }
template <> __device__ half from_float<half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

__device__ void decode_index(int64_t flat, const int64_t shape[4], int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
}

template <typename T>
__global__ void reduce_nd_kernel(const T* input, T* output, ops_reduce_nd_desc desc,
                                 tensor_strides input_strides, tensor_strides output_strides) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= desc.output_count) return;
    int64_t output_coord[4];
    int64_t input_coord[4] = {};
    decode_index(output_index, desc.output_shape, output_coord);
    int compact_axis = 0;
    for (int axis = 0; axis < 4; ++axis) {
        if (!(desc.axis_mask & (1u << axis))) {
            input_coord[axis] = output_coord[desc.keep_dims ? axis : compact_axis++];
        }
    }
    float result = desc.mode == ops_reduce_mode::maximum || desc.mode == ops_reduce_mode::logsumexp
                     ? -CUDART_INF_F
                 : desc.mode == ops_reduce_mode::minimum ? CUDART_INF_F : 0.0f;
    if (desc.mode == ops_reduce_mode::product) result = 1.0f;
    float auxiliary = 0.0f;
    for (int64_t reduction_index = 0; reduction_index < desc.reduction_count; ++reduction_index) {
        int64_t cursor = reduction_index;
        for (int axis = 0; axis < 4; ++axis) {
            if (desc.axis_mask & (1u << axis)) {
                input_coord[axis] = cursor % desc.input_shape[axis];
                cursor /= desc.input_shape[axis];
            }
        }
        size_t offset = 0;
        for (int axis = 0; axis < 4; ++axis) offset += input_coord[axis] * input_strides.values[axis];
        const float value = to_float(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) + offset));
        if (desc.mode == ops_reduce_mode::maximum) result = fmaxf(result, value);
        else if (desc.mode == ops_reduce_mode::minimum) result = fminf(result, value);
        else if (desc.mode == ops_reduce_mode::product) result *= value;
        else if (desc.mode == ops_reduce_mode::variance ||
                 desc.mode == ops_reduce_mode::standard_deviation) {
            const float delta = value - result;
            result += delta / static_cast<float>(reduction_index + 1);
            auxiliary += delta * (value - result);
        } else if (desc.mode == ops_reduce_mode::logsumexp) {
            if (value <= result) auxiliary += expf(value - result);
            else {
                auxiliary = auxiliary * expf(result - value) + 1.0f;
                result = value;
            }
        } else result += value;
    }
    if (desc.mode == ops_reduce_mode::mean) result /= static_cast<float>(desc.reduction_count);
    else if (desc.mode == ops_reduce_mode::variance ||
             desc.mode == ops_reduce_mode::standard_deviation) {
        result = auxiliary / static_cast<float>(desc.reduction_count - desc.correction);
        if (desc.mode == ops_reduce_mode::standard_deviation) result = sqrtf(result);
    } else if (desc.mode == ops_reduce_mode::logsumexp) result += logf(auxiliary);
    size_t output_offset = 0;
    for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output_strides.values[axis];
    *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + output_offset) = from_float<T>(result);
}

template <typename T>
__global__ void arg_reduce_nd_kernel(const T* input, int32_t* output, ops_arg_reduce_nd_desc desc,
                                     tensor_strides input_strides, tensor_strides output_strides) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= desc.output_count) return;
    int64_t output_coord[4];
    int64_t input_coord[4] = {};
    decode_index(output_index, desc.output_shape, output_coord);
    int compact_axis = 0;
    for (int axis = 0; axis < 4; ++axis) {
        if (!(desc.axis_mask & (1u << axis))) {
            input_coord[axis] = output_coord[desc.keep_dims ? axis : compact_axis++];
        }
    }
    float best = desc.mode == ops_arg_reduce_mode::maximum ? -CUDART_INF_F : CUDART_INF_F;
    int32_t best_index = 0;
    for (int64_t reduction_index = 0; reduction_index < desc.reduction_count; ++reduction_index) {
        int64_t cursor = reduction_index;
        for (int axis = 0; axis < 4; ++axis) {
            if (desc.axis_mask & (1u << axis)) {
                input_coord[axis] = cursor % desc.input_shape[axis];
                cursor /= desc.input_shape[axis];
            }
        }
        size_t offset = 0;
        for (int axis = 0; axis < 4; ++axis) offset += input_coord[axis] * input_strides.values[axis];
        const float value = to_float(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) + offset));
        const bool better = desc.mode == ops_arg_reduce_mode::maximum ? value > best : value < best;
        if (better) {
            best = value;
            best_index = static_cast<int32_t>(reduction_index);
        }
    }
    size_t output_offset = 0;
    for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output_strides.values[axis];
    *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(output) + output_offset) = best_index;
}

template <typename T>
bool launch_reduce(cudaStream_t stream, const ggml_tensor* input, ggml_tensor* output,
                   const ops_reduce_nd_desc& desc) {
    const tensor_strides input_strides = {{input->nb[0], input->nb[1], input->nb[2], input->nb[3]}};
    const tensor_strides output_strides = {{output->nb[0], output->nb[1], output->nb[2], output->nb[3]}};
    reduce_nd_kernel<T><<<static_cast<unsigned>((desc.output_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<T*>(output->data), desc,
        input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
bool launch_arg_reduce(cudaStream_t stream, const ggml_tensor* input, ggml_tensor* output,
                       const ops_arg_reduce_nd_desc& desc) {
    const tensor_strides input_strides = {{input->nb[0], input->nb[1], input->nb[2], input->nb[3]}};
    const tensor_strides output_strides = {{output->nb[0], output->nb[1], output->nb[2], output->nb[3]}};
    arg_reduce_nd_kernel<T><<<static_cast<unsigned>((desc.output_count + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<int32_t*>(output->data), desc,
        input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_reduce_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources,
                           1, &params, sizeof(params), node};
    ops_reduce_nd_desc desc;
    if (!ops_validate_reduce_nd_contract(request, &desc)) return false;
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return launch_reduce<float>(stream, sources[0], node, desc);
    case GGML_TYPE_F16: return launch_reduce<half>(stream, sources[0], node, desc);
    case GGML_TYPE_BF16: return launch_reduce<__nv_bfloat16>(stream, sources[0], node, desc);
    default: return false;
    }
}

bool ggml_cuda_op_arg_reduce_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_arg_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources,
                           1, &params, sizeof(params), node};
    ops_arg_reduce_nd_desc desc;
    if (!ops_validate_arg_reduce_nd_contract(request, &desc)) return false;
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) return false;
    switch (sources[0]->type) {
    case GGML_TYPE_F32: return launch_arg_reduce<float>(stream, sources[0], node, desc);
    case GGML_TYPE_F16: return launch_arg_reduce<half>(stream, sources[0], node, desc);
    case GGML_TYPE_BF16: return launch_arg_reduce<__nv_bfloat16>(stream, sources[0], node, desc);
    default: return false;
    }
}

} // namespace ggml_ops_ext::cuda
