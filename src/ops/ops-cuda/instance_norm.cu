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
__device__ inline float load_norm_param(const void* data, int type, int64_t index, float fallback) {
    if (!data) return fallback;
    return type == 0 ? ((const float*)data)[index] : (float)((const half*)data)[index];
}

template <typename T, int BLOCK_SIZE>
__global__ void instance_norm_kernel_f32_multipass(
    const T* x, const void* gamma, const void* beta, T* dst,
    int gamma_type, int beta_type, int64_t T_len, int64_t C, float eps
) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float shared_sum[32];

    float local_sum = 0.0f;
    for (int64_t t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
        local_sum += (float)x[c * T_len + t];
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
            s_mean = sum / T_len;
        }
    }
    __syncthreads();

    float mean = s_mean;

    float local_var_sum = 0.0f;
    for (int64_t t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
        float diff = (float)x[c * T_len + t] - mean;
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
            float var = var_sum / T_len;
            s_inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    float inv_std = s_inv_std;
    float g = load_norm_param(gamma, gamma_type, c, 1.0f);
    float b = load_norm_param(beta, beta_type, c, 0.0f);

    for (int64_t t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
        dst[c * T_len + t] = (T)(((float)x[c * T_len + t] - mean) * inv_std * g + b);
    }
}

// Single-pass shared memory caching kernel (T is small, fits in dynamic shared memory)
template <typename T, int BLOCK_SIZE>
__global__ void instance_norm_kernel_f32_shared(
    const T* x, const void* gamma, const void* beta, T* dst,
    int gamma_type, int beta_type, int64_t T_len, int64_t C, float eps
) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    extern __shared__ float s_data[];

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    __shared__ float shared_sum[32];

    // Load entire sequence to shared memory once
    float local_sum = 0.0f;
    for (int t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
        float val = (float)x[c * T_len + t];
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
            s_mean = sum / T_len;
        }
    }
    __syncthreads();

    float mean = s_mean;

    // Variance reduction from shared memory
    float local_var_sum = 0.0f;
    for (int t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
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
            float var = var_sum / T_len;
            s_inv_std = 1.0f / sqrtf(var + eps);
        }
    }
    __syncthreads();

    float inv_std = s_inv_std;
    float g = load_norm_param(gamma, gamma_type, c, 1.0f);
    float b = load_norm_param(beta, beta_type, c, 0.0f);

    // Write out directly from shared memory
    for (int t = threadIdx.x; t < T_len; t += BLOCK_SIZE) {
        dst[c * T_len + t] = (T)((s_data[t] - mean) * inv_std * g + b);
    }
}

template <typename T>
static void launch_instance_norm(
    cudaStream_t stream, const T* x, const void* gamma, const void* beta, T* dst,
    int gamma_type, int beta_type, int64_t T_len, int64_t C, float eps
) {
    int block_size = T_len <= 32 ? 32 : T_len <= 64 ? 64 : T_len <= 128 ? 128 : 256;
    size_t shmem_size = T_len * sizeof(float);
    bool use_shared = shmem_size <= 48 * 1024;
#define LAUNCH_INSTANCE_NORM(BS) \
    do { \
        if (use_shared) instance_norm_kernel_f32_shared<T, BS><<<C, BS, shmem_size, stream>>>( \
            x, gamma, beta, dst, gamma_type, beta_type, T_len, C, eps); \
        else instance_norm_kernel_f32_multipass<T, BS><<<C, BS, 0, stream>>>( \
            x, gamma, beta, dst, gamma_type, beta_type, T_len, C, eps); \
    } while (0)
    if (block_size == 32) LAUNCH_INSTANCE_NORM(32);
    else if (block_size == 64) LAUNCH_INSTANCE_NORM(64);
    else if (block_size == 128) LAUNCH_INSTANCE_NORM(128);
    else LAUNCH_INSTANCE_NORM(256);
#undef LAUNCH_INSTANCE_NORM
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

    int gamma_type = gamma && gamma->type == GGML_TYPE_F16 ? 1 : 0;
    int beta_type = beta && beta->type == GGML_TYPE_F16 ? 1 : 0;
    if (x->type == GGML_TYPE_F32) {
        launch_instance_norm(stream, (const float*)x->data,
            gamma ? gamma->data : nullptr, beta ? beta->data : nullptr, (float*)dst->data,
            gamma_type, beta_type, T, C, eps);
    } else if (x->type == GGML_TYPE_F16) {
        launch_instance_norm(stream, (const half*)x->data,
            gamma ? gamma->data : nullptr, beta ? beta->data : nullptr, (half*)dst->data,
            gamma_type, beta_type, T, C, eps);
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
