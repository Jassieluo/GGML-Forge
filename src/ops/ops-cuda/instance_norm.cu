#include "ops_cuda_common.cuh"

namespace ggml_ops_ext::cuda {
namespace {

__device__ float warp_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float load_parameter(const void* data, int type, int64_t channel, float fallback) {
    if (!data) {
        return fallback;
    }
    return type == 0 ? static_cast<const float*>(data)[channel]
                     : static_cast<float>(static_cast<const half*>(data)[channel]);
}

template <typename T>
__global__ void instance_norm_warp_kernel(const T* input, const void* gamma, const void* beta, T* output,
                                          int gamma_type, int beta_type, int64_t length, int64_t channels,
                                          int64_t groups, float eps) {
    constexpr int warps_per_block = 8;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t group = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
    if (group >= groups) {
        return;
    }
    const T* input_group = input + group * length;
    float local_sum = 0.0f;
    float local_squared_sum = 0.0f;
    for (int64_t index = lane; index < length; index += warpSize) {
        const float value = static_cast<float>(input_group[index]);
        local_sum += value;
        local_squared_sum += value * value;
    }
    const float sum = warp_sum(local_sum);
    const float squared_sum = warp_sum(local_squared_sum);
    const float mean = __shfl_sync(0xffffffff, sum, 0) / static_cast<float>(length);
    const float variance =
        fmaxf(__shfl_sync(0xffffffff, squared_sum, 0) / static_cast<float>(length) - mean * mean, 0.0f);
    const float inverse_std = rsqrtf(variance + eps);
    const int64_t channel = group % channels;
    const float scale = load_parameter(gamma, gamma_type, channel, 1.0f);
    const float shift = load_parameter(beta, beta_type, channel, 0.0f);
    T* output_group = output + group * length;
    for (int64_t index = lane; index < length; index += warpSize) {
        output_group[index] =
            static_cast<T>((static_cast<float>(input_group[index]) - mean) * inverse_std * scale + shift);
    }
}

template <typename T>
__global__ void instance_norm_block_kernel(const T* input, const void* gamma, const void* beta, T* output,
                                           int gamma_type, int beta_type, int64_t length, int64_t channels,
                                           int64_t groups, float eps) {
    constexpr int block_size = 256;
    const int64_t group = blockIdx.x;
    if (group >= groups) {
        return;
    }
    const T* input_group = input + group * length;
    float2 moments = make_float2(0.0f, 0.0f);
    for (int64_t index = threadIdx.x; index < length; index += block_size) {
        const float value = static_cast<float>(input_group[index]);
        moments.x += value;
        moments.y += value * value;
    }
    extern __shared__ float2 shared_moments[];
    moments = block_reduce<block_reduce_method::SUM, block_size>(moments, shared_moments);
    const float mean = moments.x / static_cast<float>(length);
    const float variance = fmaxf(moments.y / static_cast<float>(length) - mean * mean, 0.0f);
    const float inverse_std = rsqrtf(variance + eps);
    const int64_t channel = group % channels;
    const float scale = load_parameter(gamma, gamma_type, channel, 1.0f);
    const float shift = load_parameter(beta, beta_type, channel, 0.0f);
    T* output_group = output + group * length;
    for (int64_t index = threadIdx.x; index < length; index += block_size) {
        output_group[index] =
            static_cast<T>((static_cast<float>(input_group[index]) - mean) * inverse_std * scale + shift);
    }
}

template <typename T>
bool launch(cudaStream_t stream, const T* input, const void* gamma, const void* beta, T* output, int gamma_type,
            int beta_type, int64_t length, int64_t channels, int64_t groups, float eps) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    if (length > 2048) {
        instance_norm_block_kernel<<<static_cast<unsigned>(groups), threads, 32 * sizeof(float2), stream>>>(
            input, gamma, beta, output, gamma_type, beta_type, length, channels, groups, eps);
        return cudaGetLastError() == cudaSuccess;
    }
    const unsigned blocks = static_cast<unsigned>((groups + warps_per_block - 1) / warps_per_block);
    instance_norm_warp_kernel<<<blocks, threads, 0, stream>>>(input, gamma, beta, output, gamma_type, beta_type, length,
                                                              channels, groups, eps);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_instance_norm(ggml_backend_t backend, ggml_tensor* node) {
    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params)) {
        return false;
    }
    const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = static_cast<cudaStream_t>(ggml_ops_ext_bridge_cuda_get_stream(backend));
    CUDA_CHECK(cudaSetDevice(device));

    const int64_t length = params.x->ne[0];
    const int64_t channels = params.x->ne[1];
    const int64_t groups = ggml_nelements(params.x) / length;
    const int gamma_type = params.gamma && params.gamma->type == GGML_TYPE_F16 ? 1 : 0;
    const int beta_type = params.beta && params.beta->type == GGML_TYPE_F16 ? 1 : 0;
    if (params.x->type == GGML_TYPE_F32) {
        return launch(stream, static_cast<const float*>(params.x->data), params.gamma ? params.gamma->data : nullptr,
                      params.beta ? params.beta->data : nullptr, static_cast<float*>(node->data), gamma_type, beta_type,
                      length, channels, groups, params.eps);
    }
    if (params.x->type == GGML_TYPE_F16) {
        return launch(stream, static_cast<const half*>(params.x->data), params.gamma ? params.gamma->data : nullptr,
                      params.beta ? params.beta->data : nullptr, static_cast<half*>(node->data), gamma_type, beta_type,
                      length, channels, groups, params.eps);
    }
    return false;
}

bool ggml_cuda_op_instance_norm_entry(ggml_backend_t backend, ggml_tensor* node) {
    return ggml_cuda_op_instance_norm(backend, node);
}

} // namespace ggml_ops_ext::cuda
