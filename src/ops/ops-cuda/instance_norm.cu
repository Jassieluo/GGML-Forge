#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Multi-pass fallback kernel (when T is very large, exceeding shared memory limit)
template <int BLOCK_SIZE>
__global__ void instance_norm_kernel_f32_multipass(
    const float* x, const float* gamma, const float* beta, float* dst,
    int64_t T, int64_t C, float eps
) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float shared_sum[32];

    float local_sum = 0.0f;
    for (int64_t t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        local_sum += x[c * T + t];
    }

    float sum = warp_reduce_sum(local_sum);
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    sum = (threadIdx.x < BLOCK_SIZE / 32) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            s_mean = sum / T;
        }
    }
    __syncthreads();

    float mean = s_mean;

    float local_var_sum = 0.0f;
    for (int64_t t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        float diff = x[c * T + t] - mean;
        local_var_sum += diff * diff;
    }

    float var_sum = warp_reduce_sum(local_var_sum);
    if (lane == 0) {
        shared_sum[wid] = var_sum;
    }
    __syncthreads();

    var_sum = (threadIdx.x < BLOCK_SIZE / 32) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        var_sum = warp_reduce_sum(var_sum);
        if (threadIdx.x == 0) {
            float var = var_sum / T;
            s_inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    float inv_std = s_inv_std;
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    for (int64_t t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        dst[c * T + t] = (x[c * T + t] - mean) * inv_std * g + b;
    }
}

// Single-pass shared memory caching kernel (T is small, fits in dynamic shared memory)
template <int BLOCK_SIZE>
__global__ void instance_norm_kernel_f32_shared(
    const float* x, const float* gamma, const float* beta, float* dst,
    int64_t T, int64_t C, float eps
) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    extern __shared__ float s_data[];

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float shared_sum[32];

    // Load entire sequence to shared memory once
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        float val = x[c * T + t];
        s_data[t] = val;
        local_sum += val;
    }
    __syncthreads();

    // Sum reduction
    float sum = warp_reduce_sum(local_sum);
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        shared_sum[wid] = sum;
    }
    __syncthreads();

    sum = (threadIdx.x < BLOCK_SIZE / 32) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) {
            s_mean = sum / T;
        }
    }
    __syncthreads();

    float mean = s_mean;

    // Variance reduction from shared memory
    float local_var_sum = 0.0f;
    for (int t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        float diff = s_data[t] - mean;
        local_var_sum += diff * diff;
    }

    float var_sum = warp_reduce_sum(local_var_sum);
    if (lane == 0) {
        shared_sum[wid] = var_sum;
    }
    __syncthreads();

    var_sum = (threadIdx.x < BLOCK_SIZE / 32) ? shared_sum[lane] : 0.0f;
    if (wid == 0) {
        var_sum = warp_reduce_sum(var_sum);
        if (threadIdx.x == 0) {
            float var = var_sum / T;
            s_inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    float inv_std = s_inv_std;
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    // Write out directly from shared memory
    for (int t = threadIdx.x; t < T; t += BLOCK_SIZE) {
        dst[c * T + t] = (s_data[t] - mean) * inv_std * g + b;
    }
}

bool ggml_cuda_op_instance_norm(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params)) return false;

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    struct ggml_tensor* x = params.x;
    struct ggml_tensor* gamma = params.gamma;
    struct ggml_tensor* beta = params.beta;
    struct ggml_tensor* dst = node;
    float eps = params.eps;

    int64_t T = x->ne[0];
    int64_t C = x->ne[1];

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        const float* gamma_d = gamma ? (const float*)gamma->data : nullptr;
        const float* beta_d = beta ? (const float*)beta->data : nullptr;
        float* dst_d = (float*)dst->data;

        int block_size = 256;
        if (T < 256) {
            if (T <= 32) block_size = 32;
            else if (T <= 64) block_size = 64;
            else if (T <= 128) block_size = 128;
        }

        // Shared memory limit is typically 48 KB (12,288 float elements)
        size_t shmem_size = T * sizeof(float);
        bool use_shared = (shmem_size <= 48 * 1024);

        if (use_shared) {
            if (block_size == 32) {
                instance_norm_kernel_f32_shared<32><<<C, 32, shmem_size, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else if (block_size == 64) {
                instance_norm_kernel_f32_shared<64><<<C, 64, shmem_size, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else if (block_size == 128) {
                instance_norm_kernel_f32_shared<128><<<C, 128, shmem_size, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else {
                instance_norm_kernel_f32_shared<256><<<C, 256, shmem_size, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            }
        } else {
            // Multipass fallback for huge sequence lengths
            if (block_size == 32) {
                instance_norm_kernel_f32_multipass<32><<<C, 32, 0, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else if (block_size == 64) {
                instance_norm_kernel_f32_multipass<64><<<C, 64, 0, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else if (block_size == 128) {
                instance_norm_kernel_f32_multipass<128><<<C, 128, 0, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            } else {
                instance_norm_kernel_f32_multipass<256><<<C, 256, 0, stream>>>(x_d, gamma_d, beta_d, dst_d, T, C, eps);
            }
        }
    } else {
        fprintf(stderr, "Unsupported data type for CUDA InstanceNorm: %d\n", x->type);
        return false;
    }

    return true;
}

bool ggml_cuda_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_instance_norm(backend, node);
}

} // namespace cuda
} // namespace ggml_ops_ext
