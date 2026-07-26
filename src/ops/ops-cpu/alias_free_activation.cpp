#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace ggml_ops_ext::cpu {
namespace {

constexpr int64_t kFilterSize = 12;
constexpr int64_t kPadding = 5;
constexpr int64_t kTileSize = 128;
constexpr int64_t kLocalSamples = 2 * kTileSize + kFilterSize - 2;

float load_float(const ggml_tensor *tensor, int64_t byte_offset) {
  const char *address = static_cast<const char *>(tensor->data) + byte_offset;
  return tensor->type == GGML_TYPE_F32
             ? *reinterpret_cast<const float *>(address)
             : ggml_fp16_to_fp32(
                   *reinterpret_cast<const ggml_fp16_t *>(address));
}

template <typename T>
void execute(const ggml_tensor *x, const ggml_tensor *up,
             const ggml_tensor *down, const ggml_tensor *alpha,
             const ggml_tensor *beta, ggml_tensor *output, int threads) {
  const int64_t length = x->ne[0];
  const int64_t channels = x->ne[1];
  const int64_t batch2 = x->ne[2];
  const int64_t batch = batch2 * x->ne[3];
  const int64_t tiles = (length + kTileSize - 1) / kTileSize;
  const int64_t rows = batch * channels;

#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t channel = row % channels;
    const int64_t batch_index = row / channels;
    const int64_t i2 = batch_index % batch2;
    const int64_t i3 = batch_index / batch2;
    const float a = load_float(alpha, channel * alpha->nb[0]);
    const float b = load_float(beta, channel * beta->nb[0]);
    std::array<float, kFilterSize> up_values{};
    std::array<float, kFilterSize> down_values{};
    for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
      up_values[kernel] =
          load_float(up, channel * up->nb[2] + kernel * up->nb[0]);
      down_values[kernel] =
          load_float(down, channel * down->nb[2] + kernel * down->nb[0]);
    }

    for (int64_t tile = 0; tile < tiles; ++tile) {
      std::array<float, kLocalSamples> activated{};
      const int64_t output_start = tile * kTileSize;
      const int64_t upsample_start = 2 * output_start - kPadding;
      for (int64_t local = 0; local < kLocalSamples; ++local) {
        const int64_t upsample_index = upsample_start + local;
        if (upsample_index < 0 || upsample_index >= 2 * length) {
          continue;
        }
        float value = 0.0f;
        for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
          const int64_t numerator = upsample_index + kPadding - kernel;
          if (numerator < 0 || (numerator & 1) != 0) {
            continue;
          }
          const int64_t input_index = numerator / 2;
          if (input_index >= length) {
            continue;
          }
          const int64_t input_offset = i3 * x->nb[3] + i2 * x->nb[2] +
                                       channel * x->nb[1] +
                                       input_index * x->nb[0];
          value += load_float(x, input_offset) * up_values[kernel];
        }
        value *= 2.0f;
        const float sine = std::sin(value * a);
        // Same near-zero beta convention as the snake kernels: identity.
        activated[local] = std::fabs(b) < 1e-6f ? value : value + sine * sine / b;
      }

      const int64_t tile_outputs =
          std::min<int64_t>(kTileSize, length - output_start);
      for (int64_t lane = 0; lane < tile_outputs; ++lane) {
        float value = 0.0f;
        for (int64_t kernel = 0; kernel < kFilterSize; ++kernel) {
          value += activated[2 * lane + kernel] * down_values[kernel];
        }
        const int64_t output_offset = i3 * output->nb[3] + i2 * output->nb[2] +
                                      channel * output->nb[1] +
                                      (output_start + lane) * output->nb[0];
        write_val(reinterpret_cast<T *>(static_cast<char *>(output->data) +
                                        output_offset),
                  value);
      }
    }
  }
}

} // namespace

bool ops_cpu_op_alias_free_activation(ggml_backend_t backend,
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
  const int threads = backend_thread_count(backend);
  if (node->type == GGML_TYPE_F32) {
    execute<float>(node->src[0], node->src[1], node->src[2], node->src[3],
                   node->src[4], node, threads);
    return true;
  }
  if (node->type == GGML_TYPE_F16) {
    execute<ggml_fp16_t>(node->src[0], node->src[1], node->src[2], node->src[3],
                         node->src[4], node, threads);
    return true;
  }
  return false;
}

} // namespace ggml_ops_ext::cpu
