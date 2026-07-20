#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

namespace ggml_ops_ext::sycl {

class KVCacheUpdateSYCLKernel;
class KVCacheCopySYCLKernel;
class KVCacheUpdateResultSYCLKernel;

static bool launch_cache_update(::sycl::queue &queue, ggml_tensor *cache,
                                const ggml_tensor *values,
                                const ggml_tensor *position) {
  const int64_t blocks =
      values->ne[3] * values->ne[2] * values->ne[1] * (values->ne[0] / 32);
  const int cache_type = cache->type;
  const int value_type = values->type;
  void *cache_data = cache->data;
  const void *values_data = values->data;
  const int32_t *position_data = static_cast<const int32_t *>(position->data);
  const int64_t width = values->ne[0];
  const int64_t value_length = values->ne[1];
  const int64_t heads = values->ne[2];
  const int64_t capacity = cache->ne[1];
  const size_t cache_nb1 = cache->nb[1];
  const size_t cache_nb2 = cache->nb[2];
  const size_t cache_nb3 = cache->nb[3];
  const size_t value_nb0 = values->nb[0];
  const size_t value_nb1 = values->nb[1];
  const size_t value_nb2 = values->nb[2];
  const size_t value_nb3 = values->nb[3];
  if (cache_type == GGML_TYPE_F32 || cache_type == GGML_TYPE_F16) {
    const int64_t rows = values->ne[3] * values->ne[2] * value_length;
    constexpr size_t local_size = 256;
    const size_t elements = static_cast<size_t>(rows * width);
    const size_t global_size =
        (elements + local_size - 1) / local_size * local_size;
    queue.submit([&](::sycl::handler &handler) {
      handler.parallel_for<KVCacheCopySYCLKernel>(
          ::sycl::nd_range<1>(global_size, local_size),
          [=](::sycl::nd_item<1> item) {
            const int64_t index =
                static_cast<int64_t>(item.get_global_linear_id());
            if (index >= rows * width)
              return;
            const int64_t channel = index % width;
            int64_t row = index / width;
            const int64_t token = row % value_length;
            row /= value_length;
            const int64_t head = row % heads;
            const int64_t batch = row / heads;
            const int32_t target = *position_data + static_cast<int32_t>(token);
            if (target < 0 || target >= capacity)
              return;
            const char *source = static_cast<const char *>(values_data) +
                                 batch * value_nb3 + head * value_nb2 +
                                 token * value_nb1 + channel * value_nb0;
            char *destination = static_cast<char *>(cache_data) +
                                batch * cache_nb3 + head * cache_nb2 +
                                target * cache_nb1;
            const float value =
                value_type == GGML_TYPE_F32
                    ? *reinterpret_cast<const float *>(source)
                    : static_cast<float>(
                          *reinterpret_cast<const ::sycl::half *>(source));
            if (cache_type == GGML_TYPE_F32) {
              reinterpret_cast<float *>(destination)[channel] = value;
            } else {
              reinterpret_cast<::sycl::half *>(destination)[channel] = value;
            }
          });
    });
    return true;
  }
  queue.submit([&](::sycl::handler &handler) {
    ::sycl::local_accessor<float, 1> maxima(::sycl::range<1>(32), handler);
    ::sycl::local_accessor<int, 1> quantized(::sycl::range<1>(32), handler);
    handler.parallel_for<KVCacheUpdateSYCLKernel>(
        ::sycl::nd_range<1>(::sycl::range<1>(blocks * 32),
                            ::sycl::range<1>(32)),
        [=](::sycl::nd_item<1> item) {
          const int lane = static_cast<int>(item.get_local_id(0));
          const int64_t group = item.get_group(0);
          const int block_in_row = group % (width / 32);
          int64_t row = group / (width / 32);
          const int64_t token = row % value_length;
          row /= value_length;
          const int64_t head = row % heads;
          const int64_t batch = row / heads;
          const int32_t target = *position_data + static_cast<int32_t>(token);
          if (target < 0 || target >= capacity)
            return;
          const char *source = static_cast<const char *>(values_data) +
                               batch * value_nb3 + head * value_nb2 +
                               token * value_nb1 +
                               (block_in_row * 32 + lane) * value_nb0;
          char *destination = static_cast<char *>(cache_data) +
                              batch * cache_nb3 + head * cache_nb2 +
                              target * cache_nb1;
          const float value =
              value_type == GGML_TYPE_F32
                  ? *reinterpret_cast<const float *>(source)
                  : static_cast<float>(
                        *reinterpret_cast<const ::sycl::half *>(source));
          maxima[lane] = ::sycl::fabs(value);
          item.barrier(::sycl::access::fence_space::local_space);
          for (int offset = 16; offset > 0; offset >>= 1) {
            if (lane < offset)
              maxima[lane] = ::sycl::fmax(maxima[lane], maxima[lane + offset]);
            item.barrier(::sycl::access::fence_space::local_space);
          }
          if (cache_type == GGML_TYPE_Q8_0) {
            auto *output =
                reinterpret_cast<block_q8_0 *>(destination) + block_in_row;
            const float d = maxima[0] / 127.0f;
            if (lane == 0)
              output->d = static_cast<::sycl::half>(d);
            output->qs[lane] =
                static_cast<int8_t>(::sycl::rint(d == 0.0f ? 0.0f : value / d));
          } else {
            auto *output =
                reinterpret_cast<block_q4_0 *>(destination) + block_in_row;
            const float d = maxima[0] / 8.0f;
            if (lane == 0)
              output->d = static_cast<::sycl::half>(d);
            quantized[lane] = ::sycl::clamp(
                static_cast<int>(::sycl::rint(d == 0.0f ? 0.0f : value / d)) +
                    8,
                0, 15);
            item.barrier(::sycl::access::fence_space::local_space);
            if (lane < 16) {
              output->qs[lane] = static_cast<uint8_t>(
                  quantized[lane] | (quantized[lane + 16] << 4));
            }
          }
        });
  });
  return true;
}

bool ggml_sycl_op_kv_cache_update_entry(ggml_backend_t backend,
                                        ggml_tensor *node) {
  ops_kv_cache_update_params params;
  if (!ops_extract_kv_cache_update_params(node, params))
    return false;
  auto *queue =
      static_cast<::sycl::queue *>(ggml_ops_ext_bridge_sycl_get_queue(backend));
  if (!queue)
    return false;
  if (!launch_cache_update(*queue, params.cache_k, params.new_k,
                           params.position) ||
      !launch_cache_update(*queue, params.cache_v, params.new_v,
                           params.position))
    return false;
  int32_t *result_data = static_cast<int32_t *>(node->data);
  const int32_t *position_data =
      static_cast<const int32_t *>(params.position->data);
  const int32_t count = static_cast<int32_t>(params.new_k->ne[1]);
  queue->submit([&](::sycl::handler &handler) {
    handler.single_task<KVCacheUpdateResultSYCLKernel>(
        [=]() { *result_data = *position_data + count; });
  });
  return true;
}

} // namespace ggml_ops_ext::sycl
