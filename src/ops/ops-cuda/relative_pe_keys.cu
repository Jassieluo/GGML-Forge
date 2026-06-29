#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__global__ void relative_pe_keys_kernel_f32(
    const float* q, const float* r_k, float* dst,
    int d_k, int T, int n_head, int W, float scale
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // key index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // query index
    int h = blockIdx.z; // head index

    if (j >= T || i >= T || h >= n_head) return;

    int k_idx = j - i;
    float val = 0.0f;
    if (k_idx >= -W && k_idx <= W) {
        int r_idx = k_idx + W;
        int r_len = 2 * W + 1;
        const float* q_vec = q + h * T * d_k + i * d_k;
        const float* r_vec = r_k + h * r_len * d_k + r_idx * d_k;
        float sum = 0.0f;
        for (int d = 0; d < d_k; ++d) {
            sum += q_vec[d] * r_vec[d];
        }
        val = sum * scale;
    }
    dst[h * T * T + i * T + j] = val;
}

bool ggml_cuda_op_relative_pe_keys(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_relative_pe_keys_params params;
    if (!ops_extract_relative_pe_keys_params(node, params)) return false;

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    struct ggml_tensor* q = params.q;
    struct ggml_tensor* emb_rel_k = params.emb_rel_k;
    struct ggml_tensor* dst = node;

    float scale = params.scale;
    int32_t W = params.window_size;

    int64_t d_k = q->ne[0];
    int64_t T = q->ne[1];
    int64_t n_head = q->ne[2];

    if (q->type == GGML_TYPE_F32) {
        const float* q_d = (const float*)q->data;
        const float* r_d = (const float*)emb_rel_k->data;
        float* dst_d = (float*)dst->data;

        dim3 block_size(16, 16, 1);
        dim3 grid_size(
            (T + block_size.x - 1) / block_size.x,
            (T + block_size.y - 1) / block_size.y,
            n_head
        );

        relative_pe_keys_kernel_f32<<<grid_size, block_size, 0, stream>>>(
            q_d, r_d, dst_d, d_k, T, n_head, W, scale
        );
    } else {
        fprintf(stderr, "Unsupported data type for CUDA Relative PE Keys: %d\n", q->type);
        return false;
    }

    return true;
}

bool ggml_cuda_op_relative_pe_keys_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_relative_pe_keys(backend, node);
}

} // namespace cuda
} // namespace ggml_ops_ext
