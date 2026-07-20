#include "ops/ops.h"
#include "ops_sycl.h"

#include "common.hpp"

#include <algorithm>

namespace ggml_ops_ext::sycl {
namespace {

template <typename TW, typename TR> class RelativePeValuesKernel;

template <typename TW, typename TR>
void launch(::sycl::queue *queue, const TW *weights, const TR *relative,
            TW *output, int64_t width, int64_t tokens, int64_t heads,
            int64_t relative_length, int32_t window) {
  constexpr size_t local_size = 128;
  const int64_t channel_groups = (width + local_size - 1) / local_size;
  const int64_t maximum_keys = std::min<int64_t>(tokens, 2LL * window + 1);
  const size_t groups = static_cast<size_t>(tokens * heads * channel_groups);
  queue->submit([&](::sycl::handler &handler) {
    ::sycl::local_accessor<float, 1> shared_weights(
        static_cast<size_t>(maximum_keys), handler);
    handler.parallel_for<RelativePeValuesKernel<TW, TR>>(
        ::sycl::nd_range<1>(groups * local_size, local_size),
        [=](::sycl::nd_item<1> item) {
          const int64_t group =
              static_cast<int64_t>(item.get_group_linear_id());
          const int64_t lane = static_cast<int64_t>(item.get_local_linear_id());
          const int64_t channel_group = group % channel_groups;
          const int64_t head = (group / channel_groups) % heads;
          const int64_t token = group / (channel_groups * heads);
          const int64_t key_begin = ::sycl::max<int64_t>(0, token - window);
          const int64_t key_end =
              ::sycl::min<int64_t>(tokens, token + window + 1);
          const int64_t key_count = key_end - key_begin;
          for (int64_t index = lane; index < key_count; index += local_size) {
            shared_weights[index] = static_cast<float>(
                weights[(head * tokens + token) * tokens + key_begin + index]);
          }
          item.barrier(::sycl::access::fence_space::local_space);

          const int64_t channel = channel_group * local_size + lane;
          if (channel < width) {
            float sum = 0.0f;
            for (int64_t index = 0; index < key_count; ++index) {
              const int64_t key = key_begin + index;
              const int64_t relative_index = key - token + window;
              sum += shared_weights[index] *
                     static_cast<float>(
                         relative[(head * relative_length + relative_index) *
                                      width +
                                  channel]);
            }
            output[(token * heads + head) * width + channel] =
                static_cast<TW>(sum);
          }
        });
  });
}

template <typename TW>
bool dispatch_relative(::sycl::queue *queue, const TW *weights,
                       const ggml_tensor *relative, TW *output, int64_t width,
                       int64_t tokens, int64_t heads, int64_t relative_length,
                       int32_t window) {
  if (relative->type == GGML_TYPE_F32) {
    launch(queue, weights, static_cast<const float *>(relative->data), output,
           width, tokens, heads, relative_length, window);
    return true;
  }
  if (relative->type == GGML_TYPE_F16) {
    launch(queue, weights, static_cast<const ::sycl::half *>(relative->data),
           output, width, tokens, heads, relative_length, window);
    return true;
  }
  return false;
}

} // namespace

bool ggml_sycl_op_relative_pe_values_entry(ggml_backend_t backend,
                                           ggml_tensor *node) {
  ops_relative_pe_values_params params;
  if (!ops_extract_relative_pe_values_params(node, params))
    return false;
  auto *queue =
      static_cast<::sycl::queue *>(ggml_ops_ext_bridge_sycl_get_queue(backend));
  if (!queue)
    return false;
  const ggml_tensor *weights = params.attn_w;
  const int64_t width = params.emb_rel_v->ne[0];
  const int64_t tokens = weights->ne[1];
  const int64_t heads = weights->ne[2];
  const int64_t relative_length = params.emb_rel_v->ne[1];
  if (weights->type == GGML_TYPE_F32) {
    return dispatch_relative(queue, static_cast<const float *>(weights->data),
                             params.emb_rel_v, static_cast<float *>(node->data),
                             width, tokens, heads, relative_length,
                             params.window_size);
  }
  if (weights->type == GGML_TYPE_F16) {
    return dispatch_relative(
        queue, static_cast<const ::sycl::half *>(weights->data),
        params.emb_rel_v, static_cast<::sycl::half *>(node->data), width,
        tokens, heads, relative_length, params.window_size);
  }
  return false;
}

} // namespace ggml_ops_ext::sycl
