#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

__global__ void relative_pe_values_kernel_f32(
    const float* w, const float* r_v, float* dst,
    int d_k, int T, int n_head, int W
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x; // channel component
    int h = blockIdx.y * blockDim.y + threadIdx.y; // head index
    int i = blockIdx.z; // query index

    if (d >= d_k || h >= n_head || i >= T) return;

    int r_len = 2 * W + 1;
    float sum = 0.0f;
    int j_start = i - W;
    if (j_start < 0) j_start = 0;
    int j_end = i + W;
    if (j_end >= T) j_end = T - 1;

    for (int j = j_start; j <= j_end; ++j) {
        int r_idx = (j - i) + W;
        float w_val = w[h * T * T + i * T + j];
        float r_val = r_v[h * r_len * d_k + r_idx * d_k + d];
        sum += w_val * r_val;
    }

    dst[i * n_head * d_k + h * d_k + d] = sum;
}

bool ggml_cuda_op_relative_pe_values(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_relative_pe_values_params params;
    if (!ops_extract_relative_pe_values_params(node, params)) return false;

    int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    cudaStream_t stream = (cudaStream_t)ggml_ops_ext_bridge_cuda_get_stream(backend);

    CUDA_CHECK(cudaSetDevice(device));

    struct ggml_tensor* attn_w = params.attn_w;
    struct ggml_tensor* emb_rel_v = params.emb_rel_v;
    struct ggml_tensor* dst = node;

    int32_t W = params.window_size;

    int64_t T = attn_w->ne[1];
    int64_t d_k = emb_rel_v->ne[0];
    int64_t n_head = attn_w->ne[2];

    if (attn_w->type == GGML_TYPE_F32) {
        const float* w_d = (const float*)attn_w->data;
        const float* r_d = (const float*)emb_rel_v->data;
        float* dst_d = (float*)dst->data;

        dim3 block_size(16, 16, 1);
        dim3 grid_size(
            (d_k + block_size.x - 1) / block_size.x,
            (n_head + block_size.y - 1) / block_size.y,
            T
        );

        relative_pe_values_kernel_f32<<<grid_size, block_size, 0, stream>>>(
            w_d, r_d, dst_d, d_k, T, n_head, W
        );
    } else {
        fprintf(stderr, "Unsupported data type for CUDA Relative PE Values: %d\n", attn_w->type);
        return false;
    }

    return true;
}

bool ggml_cuda_op_relative_pe_values_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_cuda_op_relative_pe_values(backend, node);
}

} // namespace cuda
} // namespace ggml_ops_ext
