#include "ops_cuda_common.cuh"
#include "ggml-common.h"

namespace ggml_ops_ext::cuda {

template <typename T>
__device__ float cache_source_value(const char* row, int index, size_t stride) {
    return static_cast<float>(*reinterpret_cast<const T*>(row + index * stride));
}

template <typename Source, ggml_type CacheType>
__global__ void kv_cache_update_kernel(
    void* cache,
    const void* values,
    const int32_t* position,
    int64_t width,
    int64_t value_length,
    int64_t heads,
    int64_t batches,
    int64_t capacity,
    size_t cache_nb1, size_t cache_nb2, size_t cache_nb3,
    size_t value_nb0, size_t value_nb1, size_t value_nb2, size_t value_nb3
) {
    const int block_in_row = blockIdx.x % (width / 32);
    int64_t row = blockIdx.x / (width / 32);
    const int64_t token = row % value_length;
    row /= value_length;
    const int64_t head = row % heads;
    const int64_t batch = row / heads;
    const int32_t target = *position + static_cast<int32_t>(token);
    if (batch >= batches || target < 0 || target >= capacity) return;

    const char* source = static_cast<const char*>(values) + batch * value_nb3 +
        head * value_nb2 + token * value_nb1 + block_in_row * 32 * value_nb0;
    char* destination = static_cast<char*>(cache) + batch * cache_nb3 +
        head * cache_nb2 + target * cache_nb1;
    const int lane = threadIdx.x;
    const float value = cache_source_value<Source>(source, lane, value_nb0);

    if constexpr (CacheType == GGML_TYPE_F32) {
        reinterpret_cast<float*>(destination)[block_in_row * 32 + lane] = value;
    } else if constexpr (CacheType == GGML_TYPE_F16) {
        reinterpret_cast<half*>(destination)[block_in_row * 32 + lane] = __float2half(value);
    } else {
        float amax = fabsf(value);
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));
        }
        amax = __shfl_sync(0xffffffff, amax, 0);
        if constexpr (CacheType == GGML_TYPE_Q8_0) {
            auto* output = reinterpret_cast<block_q8_0*>(destination) + block_in_row;
            const float d = amax / 127.0f;
            if (lane == 0) output->d = __float2half(d);
            output->qs[lane] = static_cast<int8_t>(lrintf(d == 0.0f ? 0.0f : value / d));
        } else {
            auto* output = reinterpret_cast<block_q4_0*>(destination) + block_in_row;
            const float d = amax / 8.0f;
            if (lane == 0) output->d = __float2half(d);
            const int quant = max(0, min(15, static_cast<int>(lrintf(d == 0.0f ? 0.0f : value / d)) + 8));
            if (lane < 16) {
                const int other = __shfl_sync(0xffffffff, quant, lane + 16);
                output->qs[lane] = static_cast<uint8_t>(quant | (other << 4));
            }
        }
    }
}

__global__ void kv_cache_update_result(int32_t* result, const int32_t* position, int32_t count) {
    *result = *position + count;
}

template <typename Source>
static bool launch_cache_update(cudaStream_t stream, ggml_tensor* cache, const ggml_tensor* values, const ggml_tensor* position) {
    const int64_t blocks = values->ne[3] * values->ne[2] * values->ne[1] * (values->ne[0] / 32);
#define LAUNCH_CACHE(TYPE) kv_cache_update_kernel<Source, TYPE><<<blocks, 32, 0, stream>>>( \
    cache->data, values->data, static_cast<const int32_t*>(position->data), values->ne[0], values->ne[1], \
    values->ne[2], values->ne[3], cache->ne[1], cache->nb[1], cache->nb[2], cache->nb[3], \
    values->nb[0], values->nb[1], values->nb[2], values->nb[3])
    switch (cache->type) {
        case GGML_TYPE_F32: LAUNCH_CACHE(GGML_TYPE_F32); break;
        case GGML_TYPE_F16: LAUNCH_CACHE(GGML_TYPE_F16); break;
        case GGML_TYPE_Q8_0: LAUNCH_CACHE(GGML_TYPE_Q8_0); break;
        case GGML_TYPE_Q4_0: LAUNCH_CACHE(GGML_TYPE_Q4_0); break;
        default: return false;
    }
#undef LAUNCH_CACHE
    return true;
}

bool ggml_cuda_op_kv_cache_update_entry(ggml_backend_t backend, ggml_tensor* node) {
    ops_kv_cache_update_params params;
    if (!ops_extract_kv_cache_update_params(node, params)) return false;
    const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = static_cast<cudaStream_t>(ggml_ops_ext_bridge_cuda_get_stream(backend));
    CUDA_CHECK(cudaSetDevice(device));
    const bool k_ok = params.new_k->type == GGML_TYPE_F32
        ? launch_cache_update<float>(stream, params.cache_k, params.new_k, params.position)
        : launch_cache_update<half>(stream, params.cache_k, params.new_k, params.position);
    const bool v_ok = params.new_v->type == GGML_TYPE_F32
        ? launch_cache_update<float>(stream, params.cache_v, params.new_v, params.position)
        : launch_cache_update<half>(stream, params.cache_v, params.new_v, params.position);
    if (!k_ok || !v_ok) return false;
    kv_cache_update_result<<<1, 1, 0, stream>>>(
        static_cast<int32_t*>(node->data), static_cast<const int32_t*>(params.position->data),
        static_cast<int32_t>(params.new_k->ne[1]));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace ggml_ops_ext::cuda
