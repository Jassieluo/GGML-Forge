#include "ops_cuda_common.cuh"

namespace ggml_ops_ext::cuda {
namespace {

constexpr int kFilterSize = 12;
constexpr int kPadding = 5;
constexpr int kTileSize = 128;
constexpr int kLocalSamples = 2 * kTileSize + kFilterSize - 2;

__device__ float load_float(const void *data, int type, int64_t byte_offset) {
  const char *address = static_cast<const char *>(data) + byte_offset;
  return type == GGML_TYPE_F32
             ? *reinterpret_cast<const float *>(address)
             : __half2float(*reinterpret_cast<const half *>(address));
}

template <typename T>
__global__ void alias_free_activation_kernel(
    const void *x_data, int x_type, size_t x_nb0, size_t x_nb1, size_t x_nb2,
    size_t x_nb3, const void *up_data, int up_type, size_t up_nb0,
    size_t up_nb2, const void *down_data, int down_type, size_t down_nb0,
    size_t down_nb2, const void *alpha_data, int alpha_type, size_t alpha_nb0,
    const void *beta_data, int beta_type, size_t beta_nb0, void *output_data,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3,
    int64_t length, int64_t channels, int64_t batch2, int64_t tiles) {
  __shared__ float activated[kLocalSamples];
  const int64_t lane = threadIdx.x;
  const int64_t group = blockIdx.x;
  const int64_t tile = group % tiles;
  const int64_t channel_batch = group / tiles;
  const int64_t channel = channel_batch % channels;
  const int64_t batch_index = channel_batch / channels;
  const int64_t i2 = batch_index % batch2;
  const int64_t i3 = batch_index / batch2;
  const int64_t output_start = tile * kTileSize;
  const int64_t upsample_start = 2 * output_start - kPadding;
  const float a = load_float(alpha_data, alpha_type, channel * alpha_nb0);
  const float b = load_float(beta_data, beta_type, channel * beta_nb0);

  for (int64_t local = lane; local < kLocalSamples; local += blockDim.x) {
    const int64_t upsample_index = upsample_start + local;
    float value = 0.0f;
    if (upsample_index >= 0 && upsample_index < 2 * length) {
      for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
        const int64_t numerator = upsample_index + kPadding - kernel;
        if (numerator < 0 || (numerator & 1) != 0) {
          continue;
        }
        const int64_t input_index = numerator / 2;
        if (input_index >= length) {
          continue;
        }
        const int64_t input_offset =
            i3 * x_nb3 + i2 * x_nb2 + channel * x_nb1 + input_index * x_nb0;
        const int64_t filter_offset = channel * up_nb2 + kernel * up_nb0;
        value += load_float(x_data, x_type, input_offset) *
                 load_float(up_data, up_type, filter_offset);
      }
      value *= 2.0f;
      const float sine = sinf(value * a);
      value += sine * sine / b;
    }
    activated[local] = value;
  }
  __syncthreads();

  const int64_t output_index = output_start + lane;
  if (output_index >= length) {
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int kernel = 0; kernel < kFilterSize; ++kernel) {
    const int64_t filter_offset = channel * down_nb2 + kernel * down_nb0;
    value += activated[2 * lane + kernel] *
             load_float(down_data, down_type, filter_offset);
  }
  const int64_t output_offset = i3 * output_nb3 + i2 * output_nb2 +
                                channel * output_nb1 +
                                output_index * output_nb0;
  *reinterpret_cast<T *>(static_cast<char *>(output_data) + output_offset) =
      static_cast<T>(value);
}

template <typename T>
bool launch(cudaStream_t stream, const ggml_tensor *x, const ggml_tensor *up,
            const ggml_tensor *down, const ggml_tensor *alpha,
            const ggml_tensor *beta, ggml_tensor *output) {
  const int64_t length = x->ne[0];
  const int64_t channels = x->ne[1];
  const int64_t batch2 = x->ne[2];
  const int64_t batch = batch2 * x->ne[3];
  const int64_t tiles = (length + kTileSize - 1) / kTileSize;
  const int64_t groups = batch * channels * tiles;
  if (groups <= 0 || groups > UINT32_MAX) {
    return false;
  }
  alias_free_activation_kernel<T>
      <<<static_cast<unsigned>(groups), kTileSize, 0, stream>>>(
          x->data, x->type, x->nb[0], x->nb[1], x->nb[2], x->nb[3], up->data,
          up->type, up->nb[0], up->nb[2], down->data, down->type, down->nb[0],
          down->nb[2], alpha->data, alpha->type, alpha->nb[0], beta->data,
          beta->type, beta->nb[0], output->data, output->nb[0], output->nb[1],
          output->nb[2], output->nb[3], length, channels, batch2, tiles);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_alias_free_activation_entry(ggml_backend_t backend,
                                              ggml_tensor *node) {
  if (!node) {
    return false;
  }
  ops_request request{ggml_backend_get_device(backend),
                      GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION,
                      node->src,
                      5,
                      nullptr,
                      0,
                      node};
  if (!ops_validate_alias_free_activation(request)) {
    return false;
  }
  const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
  cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
  CUDA_CHECK(cudaSetDevice(device));
  if (node->type == GGML_TYPE_F32) {
    return launch<float>(stream, node->src[0], node->src[1], node->src[2],
                         node->src[3], node->src[4], node);
  }
  if (node->type == GGML_TYPE_F16) {
    return launch<half>(stream, node->src[0], node->src[1], node->src[2],
                        node->src[3], node->src[4], node);
  }
  return false;
}

} // namespace ggml_ops_ext::cuda
