#include "ops_cuda_common.cuh"

#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {
namespace cuda {

// Deterministic fused temperature-softmax + top-k/top-p sampling.
//
// Mirrors src/ops/ops-cpu/sample.cpp bit-for-bit in structure: a full sort of
// (logit, index) pairs ordered by (logit desc, index asc), then a sequential
// single-threaded softmax/normalize/scan so float summation order matches the
// host reference exactly. Determinism and tie-breaking matter more than speed.
//
// Design: one block. The candidates are padded to a power of two and bitonic
// sorted in shared memory (padding entries are (-inf, INT32_MAX) so they sort
// strictly last and ties inside the padding still break by index). Thread 0
// then performs the sequential selection. Shared memory is
// padded * (sizeof(float) + sizeof(int32_t)) bytes, so the supported vocab is
// capped at SAMPLE_DIST_MAX_VOCAB (4096 -> 32 KiB, within the 48 KiB default
// static limit). The supports_ probe in ops_cuda.cu rejects larger vocabs and
// the graph builder falls back.
constexpr int64_t SAMPLE_DIST_MAX_VOCAB = 4096;

// Strict total order over (logit, index): logit descending, index ascending.
__device__ inline bool sample_comes_before(float value_a, int32_t index_a, float value_b,
                                           int32_t index_b) {
    return value_a > value_b || (value_a == value_b && index_a < index_b);
}

template <int block_size>
__global__ void sample_dist_kernel(const float* logits, const float* uniform, int32_t* out,
                                   int vocab, int padded, int32_t top_k, float top_p,
                                   float temperature) {
    extern __shared__ char smem[];
    float* vals = (float*)smem;
    int32_t* idxs = (int32_t*)(vals + padded);
    const int tid = threadIdx.x;

    for (int i = tid; i < padded; i += block_size) {
        vals[i] = i < vocab ? logits[i] : -INFINITY;
        idxs[i] = i < vocab ? i : INT32_MAX;
    }
    __syncthreads();

    // Bitonic sort; "ascending" means sample_comes_before order. Within one
    // (k, j) step every pair (i, i^j) is touched by exactly one loop index, so
    // steps only need a barrier between them.
    for (int k = 2; k <= padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < padded; i += block_size) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = (i & k) == 0;
                    const float vi = vals[i];
                    const int32_t ii = idxs[i];
                    const float vx = vals[ixj];
                    const int32_t ix = idxs[ixj];
                    const bool out_of_order = ascending
                                                  ? sample_comes_before(vx, ix, vi, ii)
                                                  : sample_comes_before(vi, ii, vx, ix);
                    if (out_of_order) {
                        vals[i] = vx;
                        idxs[i] = ix;
                        vals[ixj] = vi;
                        idxs[ixj] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid != 0) {
        return;
    }

    // Sequential selection replicating the host reference float-for-float:
    // same operations in the same order over the sorted prefix.
    float u = uniform[0];
    u = fminf(fmaxf(u, 0.0f), 0.999999f);

    const int keep = top_k > 0 ? (top_k < vocab ? top_k : vocab) : vocab;
    const float max_logit = vals[0];

    float total = 0.0f;
    for (int i = 0; i < keep; ++i) {
        const float p = expf((vals[i] - max_logit) / temperature);
        vals[i] = p;
        total += p;
    }
    for (int i = 0; i < keep; ++i) {
        vals[i] /= total;
    }

    int kept = keep;
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (int i = 0; i < keep; ++i) {
            cumulative += vals[i];
            if (cumulative >= top_p) {
                kept = i + 1;
                break;
            }
        }
    }

    float kept_total = 0.0f;
    for (int i = 0; i < kept; ++i) {
        kept_total += vals[i];
    }
    const float threshold = u * kept_total;

    float cumulative = 0.0f;
    int32_t selected = idxs[kept - 1];
    for (int i = 0; i < kept; ++i) {
        cumulative += vals[i];
        if (cumulative > threshold) {
            selected = idxs[i];
            break;
        }
    }
    out[0] = selected;
}

bool ggml_cuda_op_sample_dist(ggml_backend_t backend, struct ggml_tensor* logits,
                              struct ggml_tensor* uniform, struct ggml_tensor* dst, int32_t top_k,
                              float top_p, float temperature) {
    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    const int64_t vocab = logits->ne[0];
    if (vocab <= 0 || vocab > SAMPLE_DIST_MAX_VOCAB) {
        fprintf(stderr, "CUDA SampleDist: vocab out of range (%lld)\n", (long long)vocab);
        return false;
    }

    int64_t padded = 2;
    while (padded < vocab) {
        padded <<= 1;
    }
    const size_t shared_size = (size_t)padded * (sizeof(float) + sizeof(int32_t));

    // The uniform scalar lives in DEVICE memory; it is read inside the kernel.
    constexpr int block_size = 256;
    sample_dist_kernel<block_size><<<1, block_size, shared_size, stream>>>(
        (const float*)logits->data, (const float*)uniform->data, (int32_t*)dst->data, (int)vocab,
        (int)padded, top_k, top_p, temperature);

    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_sample_dist_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_sample_dist_params params;
    std::memcpy(&params, node->op_params, sizeof(params));
    return ggml_cuda_op_sample_dist(backend, node->src[0], node->src[1], node, params.top_k,
                                    params.top_p, params.temperature);
}

} // namespace cuda
} // namespace ggml_ops_ext
