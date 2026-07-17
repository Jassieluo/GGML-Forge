#include "ops_cuda_common.cuh"

namespace ggml_ops_ext {
namespace cuda {

template <typename TW, typename TR>
__global__ void relative_pe_values_kernel(
    const TW* w, const TR* r_v, TW* dst,
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
        float w_val = (float)w[h * T * T + i * T + j];
        float r_val = (float)r_v[h * r_len * d_k + r_idx * d_k + d];
        sum += w_val * r_val;
    }

    dst[i * n_head * d_k + h * d_k + d] = (TW)sum;
}

template <typename TW, typename TR>
static void launch_relative_pe_values(
    cudaStream_t stream, const void* w, const void* r, void* dst,
    int d_k, int T, int n_head, int W
) {
    dim3 block_size(16, 16, 1);
    dim3 grid_size((d_k + 15) / 16, (n_head + 15) / 16, T);
    relative_pe_values_kernel<<<grid_size, block_size, 0, stream>>>(
        (const TW*)w, (const TR*)r, (TW*)dst, d_k, T, n_head, W);
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

        if (emb_rel_v->type == GGML_TYPE_F32) {
            launch_relative_pe_values<float, float>(stream, w_d, r_d, dst_d, d_k, T, n_head, W);
        } else {
            launch_relative_pe_values<float, half>(stream, w_d, emb_rel_v->data, dst_d, d_k, T, n_head, W);
        }
    } else if (attn_w->type == GGML_TYPE_F16) {
        if (emb_rel_v->type == GGML_TYPE_F32) {
            launch_relative_pe_values<half, float>(stream, attn_w->data, emb_rel_v->data, dst->data, d_k, T, n_head, W);
        } else {
            launch_relative_pe_values<half, half>(stream, attn_w->data, emb_rel_v->data, dst->data, d_k, T, n_head, W);
        }
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
