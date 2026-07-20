#include "ops_cuda_common.cuh"

namespace ggml_ops_ext::cuda {
namespace {

template <typename TQ, typename TR>
__global__ void relative_pe_keys_kernel(const TQ *query, const TR *relative,
                                        TQ *output, int width, int tokens,
                                        int relative_length, int window,
                                        float scale) {
  const int lane = threadIdx.x;
  const int relative_index = blockIdx.x;
  const int token = blockIdx.y;
  const int head = blockIdx.z;
  const int key = token + relative_index - window;
  if (key < 0 || key >= tokens)
    return;

  const TQ *query_row = query + (head * tokens + token) * width;
  const TR *relative_row =
      relative + (head * relative_length + relative_index) * width;
  float sum = 0.0f;
  for (int channel = lane; channel < width; channel += warpSize) {
    sum += static_cast<float>(query_row[channel]) *
           static_cast<float>(relative_row[channel]);
  }
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0)
    output[(head * tokens + token) * tokens + key] =
        static_cast<TQ>(sum * scale);
}

template <typename TQ, typename TR>
bool launch(cudaStream_t stream, const TQ *query, const TR *relative,
            TQ *output, int width, int tokens, int heads, int relative_length,
            int window, float scale) {
  CUDA_CHECK(cudaMemsetAsync(
      output, 0, static_cast<size_t>(tokens) * tokens * heads * sizeof(TQ),
      stream));
  const int active_relative = std::min(relative_length, 2 * window + 1);
  const dim3 grid(active_relative, tokens, heads);
  relative_pe_keys_kernel<<<grid, 32, 0, stream>>>(
      query, relative, output, width, tokens, relative_length, window, scale);
  return cudaGetLastError() == cudaSuccess;
}

template <typename TQ>
bool dispatch_relative(cudaStream_t stream, const TQ *query,
                       const ggml_tensor *relative, TQ *output, int width,
                       int tokens, int heads, int relative_length, int window,
                       float scale) {
  if (relative->type == GGML_TYPE_F32) {
    return launch(stream, query, static_cast<const float *>(relative->data),
                  output, width, tokens, heads, relative_length, window, scale);
  }
  if (relative->type == GGML_TYPE_F16) {
    return launch(stream, query, static_cast<const half *>(relative->data),
                  output, width, tokens, heads, relative_length, window, scale);
  }
  return false;
}

} // namespace

bool ggml_cuda_op_relative_pe_keys_entry(ggml_backend_t backend,
                                         ggml_tensor *node) {
  ops_relative_pe_keys_params params;
  if (!ops_extract_relative_pe_keys_params(node, params))
    return false;
  const ggml_tensor *query = params.q;
  const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
  cudaStream_t stream =
      static_cast<cudaStream_t>(ggml_ops_ext_bridge_cuda_get_stream(backend));
  CUDA_CHECK(cudaSetDevice(device));
  const int width = static_cast<int>(query->ne[0]);
  const int tokens = static_cast<int>(query->ne[1]);
  const int heads = static_cast<int>(query->ne[2]);
  const int relative_length = static_cast<int>(params.emb_rel_k->ne[1]);
  if (query->type == GGML_TYPE_F32) {
    return dispatch_relative(stream, static_cast<const float *>(query->data),
                             params.emb_rel_k, static_cast<float *>(node->data),
                             width, tokens, heads, relative_length,
                             params.window_size, params.scale);
  }
  if (query->type == GGML_TYPE_F16) {
    return dispatch_relative(stream, static_cast<const half *>(query->data),
                             params.emb_rel_k, static_cast<half *>(node->data),
                             width, tokens, heads, relative_length,
                             params.window_size, params.scale);
  }
  return false;
}

} // namespace ggml_ops_ext::cuda
