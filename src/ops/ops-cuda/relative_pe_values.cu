#include "ops_cuda_common.cuh"

namespace ggml_ops_ext::cuda {
namespace {

template <typename TW, typename TR>
__global__ void relative_pe_values_kernel(const TW *weights, const TR *relative,
                                          TW *output, int width, int tokens,
                                          int heads, int relative_length,
                                          int window) {
  extern __shared__ float shared_weights[];
  const int token = blockIdx.y;
  const int head = blockIdx.z;
  const int key_begin = max(0, token - window);
  const int key_end = min(tokens, token + window + 1);
  const int key_count = key_end - key_begin;
  for (int index = threadIdx.x; index < key_count; index += blockDim.x) {
    shared_weights[index] = static_cast<float>(
        weights[(head * tokens + token) * tokens + key_begin + index]);
  }
  __syncthreads();

  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= width)
    return;
  float sum = 0.0f;
  for (int index = 0; index < key_count; ++index) {
    const int key = key_begin + index;
    const int relative_index = key - token + window;
    sum += shared_weights[index] *
           static_cast<float>(
               relative[(head * relative_length + relative_index) * width +
                        channel]);
  }
  output[(token * heads + head) * width + channel] = static_cast<TW>(sum);
}

template <typename TW, typename TR>
bool launch(cudaStream_t stream, const TW *weights, const TR *relative,
            TW *output, int width, int tokens, int heads, int relative_length,
            int window) {
  constexpr int threads = 256;
  const dim3 grid((width + threads - 1) / threads, tokens, heads);
  const size_t shared_bytes =
      static_cast<size_t>(std::min(tokens, 2 * window + 1)) * sizeof(float);
  relative_pe_values_kernel<<<grid, threads, shared_bytes, stream>>>(
      weights, relative, output, width, tokens, heads, relative_length, window);
  return cudaGetLastError() == cudaSuccess;
}

template <typename TW>
bool dispatch_relative(cudaStream_t stream, const TW *weights,
                       const ggml_tensor *relative, TW *output, int width,
                       int tokens, int heads, int relative_length, int window) {
  if (relative->type == GGML_TYPE_F32) {
    return launch(stream, weights, static_cast<const float *>(relative->data),
                  output, width, tokens, heads, relative_length, window);
  }
  if (relative->type == GGML_TYPE_F16) {
    return launch(stream, weights, static_cast<const half *>(relative->data),
                  output, width, tokens, heads, relative_length, window);
  }
  return false;
}

} // namespace

bool ggml_cuda_op_relative_pe_values_entry(ggml_backend_t backend,
                                           ggml_tensor *node) {
  ops_relative_pe_values_params params;
  if (!ops_extract_relative_pe_values_params(node, params))
    return false;
  const ggml_tensor *weights = params.attn_w;
  const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
  cudaStream_t stream =
      static_cast<cudaStream_t>(ggml_ops_ext_bridge_cuda_get_stream(backend));
  CUDA_CHECK(cudaSetDevice(device));
  const int width = static_cast<int>(params.emb_rel_v->ne[0]);
  const int tokens = static_cast<int>(weights->ne[1]);
  const int heads = static_cast<int>(weights->ne[2]);
  const int relative_length = static_cast<int>(params.emb_rel_v->ne[1]);
  if (weights->type == GGML_TYPE_F32) {
    return dispatch_relative(stream, static_cast<const float *>(weights->data),
                             params.emb_rel_v, static_cast<float *>(node->data),
                             width, tokens, heads, relative_length,
                             params.window_size);
  }
  if (weights->type == GGML_TYPE_F16) {
    return dispatch_relative(stream, static_cast<const half *>(weights->data),
                             params.emb_rel_v, static_cast<half *>(node->data),
                             width, tokens, heads, relative_length,
                             params.window_size);
  }
  return false;
}

} // namespace ggml_ops_ext::cuda
