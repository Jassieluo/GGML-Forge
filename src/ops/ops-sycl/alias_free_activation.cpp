#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"
#include <sycl/sycl.hpp>

namespace ggml_ops_ext {
namespace sycl {

template <typename T>
inline float load_float(const void *data, int type, int64_t index) {
  return type == GGML_TYPE_F32
             ? static_cast<const float *>(data)[index]
             : static_cast<float>(
                   static_cast<const ::sycl::half *>(data)[index]);
}

template <typename T> class AliasFreeActivationKernel;

template <typename T>
static void
launch_alias_free_activation(::sycl::queue *q, const ggml_tensor *x,
                             const ggml_tensor *up, const ggml_tensor *down,
                             const ggml_tensor *alpha, const ggml_tensor *beta,
                             ggml_tensor *dst) {
  constexpr int64_t tile = 128;
  constexpr int64_t kernel = 12;
  constexpr int64_t padding = 5;
  constexpr int64_t local_samples = 2 * tile + kernel - 2;
  const int64_t length = x->ne[0];
  const int64_t channels = x->ne[1];
  const int64_t batch2 = x->ne[2];
  const int64_t batch = batch2 * x->ne[3];
  const int64_t tiles = (length + tile - 1) / tile;
  const int64_t groups = batch * channels * tiles;
  const int up_type = up->type;
  const int down_type = down->type;
  const int alpha_type = alpha->type;
  const int beta_type = beta->type;
  const void *x_data = x->data;
  const void *up_data = up->data;
  const void *down_data = down->data;
  const void *alpha_data = alpha->data;
  const void *beta_data = beta->data;
  void *dst_data = dst->data;
  const size_t x_nb0 = x->nb[0];
  const size_t x_nb1 = x->nb[1];
  const size_t x_nb2 = x->nb[2];
  const size_t x_nb3 = x->nb[3];
  const size_t up_nb0 = up->nb[0];
  const size_t up_nb2 = up->nb[2];
  const size_t down_nb0 = down->nb[0];
  const size_t down_nb2 = down->nb[2];
  const size_t dst_nb0 = dst->nb[0];
  const size_t dst_nb1 = dst->nb[1];
  const size_t dst_nb2 = dst->nb[2];
  const size_t dst_nb3 = dst->nb[3];

  q->submit([&](::sycl::handler &cgh) {
    ::sycl::local_accessor<float, 1> activated(::sycl::range<1>(local_samples),
                                               cgh);
    cgh.parallel_for<AliasFreeActivationKernel<T>>(
        ::sycl::nd_range<1>(groups * tile, tile), [=](::sycl::nd_item<1> item) {
          const int64_t lane = item.get_local_id(0);
          const int64_t group_id = item.get_group(0);
          const int64_t tile_id = group_id % tiles;
          const int64_t channel_batch = group_id / tiles;
          const int64_t channel = channel_batch % channels;
          const int64_t batch_index = channel_batch / channels;
          const int64_t i2 = batch_index % batch2;
          const int64_t i3 = batch_index / batch2;
          const int64_t output_start = tile_id * tile;
          const int64_t upsample_start = 2 * output_start - padding;

          for (int64_t local_u = lane; local_u < local_samples;
               local_u += tile) {
            const int64_t u = upsample_start + local_u;
            float value = 0.0f;
            if (u >= 0 && u < 2 * length)
              for (int64_t ku = 0; ku < kernel; ++ku) {
                const int64_t numerator = u + padding - ku;
                if (numerator < 0 || (numerator & 1) != 0)
                  continue;
                const int64_t input_index = numerator / 2;
                if (input_index >= length)
                  continue;
                const T input_value = *reinterpret_cast<const T *>(
                    reinterpret_cast<const char *>(x_data) + i3 * x_nb3 +
                    i2 * x_nb2 + channel * x_nb1 + input_index * x_nb0);
                const int64_t filter_offset = channel * up_nb2 + ku * up_nb0;
                const float filter_value =
                    up_type == GGML_TYPE_F32
                        ? *reinterpret_cast<const float *>(
                              reinterpret_cast<const char *>(up_data) +
                              filter_offset)
                        : static_cast<float>(
                              *reinterpret_cast<const ::sycl::half *>(
                                  reinterpret_cast<const char *>(up_data) +
                                  filter_offset));
                value += static_cast<float>(input_value) * filter_value;
              }
            if (u >= 0 && u < 2 * length) {
              value *= 2.0f;
              const float a = load_float<T>(alpha_data, alpha_type, channel);
              const float b = load_float<T>(beta_data, beta_type, channel);
              const float sine = ::sycl::sin(value * a);
              // Same near-zero beta convention as the snake kernels: identity.
              if (::sycl::fabs(b) >= 1e-6f) {
                value += sine * sine / b;
              }
            }
            activated[local_u] = value;
          }
          item.barrier(::sycl::access::fence_space::local_space);

          const int64_t output_index = output_start + lane;
          if (output_index < length) {
            float sum = 0.0f;
            for (int64_t kd = 0; kd < kernel; ++kd) {
              const int64_t filter_offset = channel * down_nb2 + kd * down_nb0;
              const float filter_value =
                  down_type == GGML_TYPE_F32
                      ? *reinterpret_cast<const float *>(
                            reinterpret_cast<const char *>(down_data) +
                            filter_offset)
                      : static_cast<float>(
                            *reinterpret_cast<const ::sycl::half *>(
                                reinterpret_cast<const char *>(down_data) +
                                filter_offset));
              sum += activated[2 * lane + kd] * filter_value;
            }
            *reinterpret_cast<T *>(reinterpret_cast<char *>(dst_data) +
                                   i3 * dst_nb3 + i2 * dst_nb2 +
                                   channel * dst_nb1 + output_index * dst_nb0) =
                static_cast<T>(sum);
          }
        });
  });
}

bool ggml_sycl_op_alias_free_activation_entry(ggml_backend_t backend,
                                              ggml_tensor *node) {
  if (!node)
    return false;
  ops_request request;
  request.op_id = GGML_OP_OPS_VIRT_ALIAS_FREE_ACTIVATION;
  request.srcs = node->src;
  request.n_srcs = 5;
  request.output = node;
  if (!ops_validate_alias_free_activation(request))
    return false;
  ::sycl::queue *q =
      static_cast<::sycl::queue *>(ggml_ops_ext_bridge_sycl_get_queue(backend));
  if (!q)
    return false;
  if (node->type == GGML_TYPE_F32) {
    launch_alias_free_activation<float>(q, node->src[0], node->src[1],
                                        node->src[2], node->src[3],
                                        node->src[4], node);
  } else if (node->type == GGML_TYPE_F16) {
    launch_alias_free_activation<::sycl::half>(q, node->src[0], node->src[1],
                                               node->src[2], node->src[3],
                                               node->src[4], node);
  } else {
    return false;
  }
  return true;
}

} // namespace sycl
} // namespace ggml_ops_ext
