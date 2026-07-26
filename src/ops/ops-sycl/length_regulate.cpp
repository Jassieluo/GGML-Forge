#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

class LengthRegulateSYCLKernel;

// Duration-based frame expansion; mirrors the CUDA kernel: one workgroup
// builds the clamped duration prefix sum in local memory, then threads
// scatter output frames via binary search. Probe caps frames at
// ops_sycl_length_regulate_max_frames so the prefix fits in local memory.
static void launch_length_regulate(::sycl::queue* queue, const float* x, const int32_t* dur,
                                   float* out, int64_t channels, int64_t frames,
                                   int64_t total) {
    constexpr size_t local_size = 256;
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<int32_t, 1> prefix(
            ::sycl::range<1>(static_cast<size_t>(frames + 1)), handler);
        handler.parallel_for<LengthRegulateSYCLKernel>(
            ::sycl::nd_range<1>(local_size, local_size), [=](::sycl::nd_item<1> item) {
                if (item.get_local_linear_id() == 0) {
                    prefix[0] = 0;
                    for (int64_t t = 0; t < frames; ++t) {
                        const int32_t d = dur[t] > 0 ? dur[t] : 0;
                        int64_t next = static_cast<int64_t>(prefix[t]) + d;
                        if (next > total) next = total;
                        prefix[t + 1] = static_cast<int32_t>(next);
                    }
                }
                item.barrier(::sycl::access::fence_space::local_space);

                const int64_t count = channels * total;
                for (int64_t idx = static_cast<int64_t>(item.get_local_linear_id());
                     idx < count; idx += static_cast<int64_t>(local_size)) {
                    const int64_t pos = idx / channels;
                    const int64_t c = idx % channels;
                    int found = -1;
                    int lo = 0, hi = static_cast<int>(frames) - 1;
                    while (lo <= hi) {
                        const int mid = (lo + hi) / 2;
                        if (static_cast<int64_t>(prefix[mid]) <= pos &&
                            pos < static_cast<int64_t>(prefix[mid + 1])) {
                            found = mid;
                            break;
                        }
                        if (pos < static_cast<int64_t>(prefix[mid])) {
                            hi = mid - 1;
                        } else {
                            lo = mid + 1;
                        }
                    }
                    out[pos * channels + c] =
                        found >= 0 ? x[static_cast<int64_t>(found) * channels + c] : 0.0f;
                }
            });
    });
}

bool ggml_sycl_op_length_regulate(ggml_backend_t backend, struct ggml_tensor* node) {
    ::sycl::queue* queue =
        static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }

    ops_length_regulate_params params;
    std::memcpy(&params, node->op_params, sizeof(params));
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* durations = node->src[1];

    try {
        launch_length_regulate(queue,
                               static_cast<const float*>(x->data),
                               static_cast<const int32_t*>(durations->data),
                               static_cast<float*>(node->data),
                               x->ne[0], x->ne[1], params.total);
    } catch (const ::sycl::exception& error) {
        std::fprintf(stderr, "[ops-sycl] length_regulate failed: %s\n", error.what());
        return false;
    }
    return true;
}

} // namespace sycl
} // namespace ggml_ops_ext
