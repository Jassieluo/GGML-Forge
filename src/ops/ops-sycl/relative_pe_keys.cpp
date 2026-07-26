#include "ops/ops.h"
#include "ops_sycl.h"

#include "common.hpp"

#include <algorithm>

namespace ggml_ops_ext::sycl {
namespace {

template <typename TQ, typename TR> class RelativePeKeysKernel;

// emb_head_stride: per-head row stride into the embedding table — 0 when the
// table is shared across heads (emb ne[2] == 1).
template <typename TQ, typename TR>
void launch(::sycl::queue *queue, const TQ *query, const TR *relative,
            TQ *output, int64_t width, int64_t tokens, int64_t heads,
            int64_t relative_length, int64_t emb_head_stride, int32_t window,
            float scale) {
  constexpr size_t local_size = 32;
  const int64_t active_relative =
      std::min<int64_t>(relative_length, 2LL * window + 1);
  const size_t groups = static_cast<size_t>(heads * tokens * active_relative);
  const ::sycl::event clear_event = queue->memset(
      output, 0, static_cast<size_t>(tokens * tokens * heads) * sizeof(TQ));
  queue->submit([&](::sycl::handler &handler) {
    handler.depends_on(clear_event);
    ::sycl::local_accessor<float, 1> partial(local_size, handler);
    handler.parallel_for<RelativePeKeysKernel<TQ, TR>>(
        ::sycl::nd_range<1>(groups * local_size, local_size),
        [=](::sycl::nd_item<1> item) {
          const int64_t group =
              static_cast<int64_t>(item.get_group_linear_id());
          const int64_t lane = static_cast<int64_t>(item.get_local_linear_id());
          const int64_t relative_index = group % active_relative;
          const int64_t token = (group / active_relative) % tokens;
          const int64_t head = group / (active_relative * tokens);
          const int64_t key = token + relative_index - window;
          float sum = 0.0f;
          if (key >= 0 && key < tokens) {
            const TQ *query_row = query + (head * tokens + token) * width;
            const TR *relative_row =
                relative + (head * emb_head_stride + relative_index) * width;
            for (int64_t channel = lane; channel < width;
                 channel += local_size) {
              sum += static_cast<float>(query_row[channel]) *
                     static_cast<float>(relative_row[channel]);
            }
          }
          partial[lane] = sum;
          item.barrier(::sycl::access::fence_space::local_space);
          for (int offset = local_size / 2; offset > 0; offset /= 2) {
            if (lane < offset)
              partial[lane] += partial[lane + offset];
            item.barrier(::sycl::access::fence_space::local_space);
          }
          if (lane == 0 && key >= 0 && key < tokens) {
            output[(head * tokens + token) * tokens + key] =
                static_cast<TQ>(partial[0] * scale);
          }
        });
  });
}

template <typename TQ>
bool dispatch_relative(::sycl::queue *queue, const TQ *query,
                       const ggml_tensor *relative, TQ *output, int64_t width,
                       int64_t tokens, int64_t heads, int64_t relative_length,
                       int64_t emb_head_stride, int32_t window, float scale) {
  if (relative->type == GGML_TYPE_F32) {
    launch(queue, query, static_cast<const float *>(relative->data), output,
           width, tokens, heads, relative_length, emb_head_stride, window,
           scale);
    return true;
  }
  if (relative->type == GGML_TYPE_F16) {
    launch(queue, query, static_cast<const ::sycl::half *>(relative->data),
           output, width, tokens, heads, relative_length, emb_head_stride,
           window, scale);
    return true;
  }
  return false;
}

} // namespace

bool ggml_sycl_op_relative_pe_keys_entry(ggml_backend_t backend,
                                         ggml_tensor *node) {
  ops_relative_pe_keys_params params;
  if (!ops_extract_relative_pe_keys_params(node, params))
    return false;
  auto *queue =
      static_cast<::sycl::queue *>(ggml_ops_ext_bridge_sycl_get_queue(backend));
  if (!queue)
    return false;
  const ggml_tensor *query = params.q;
  const int64_t width = query->ne[0];
  const int64_t tokens = query->ne[1];
  const int64_t heads = query->ne[2];
  const int64_t relative_length = params.emb_rel_k->ne[1];
  const int64_t emb_head_stride =
      params.emb_rel_k->ne[2] == 1 ? 0 : relative_length;
  if (query->type == GGML_TYPE_F32) {
    return dispatch_relative(queue, static_cast<const float *>(query->data),
                             params.emb_rel_k, static_cast<float *>(node->data),
                             width, tokens, heads, relative_length,
                             emb_head_stride, params.window_size, params.scale);
  }
  if (query->type == GGML_TYPE_F16) {
    return dispatch_relative(
        queue, static_cast<const ::sycl::half *>(query->data), params.emb_rel_k,
        static_cast<::sycl::half *>(node->data), width, tokens, heads,
        relative_length, emb_head_stride, params.window_size, params.scale);
  }
  return false;
}

} // namespace ggml_ops_ext::sycl
