#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <cstdio>
#include <cstring>
#include <limits>

namespace ggml_ops_ext {
namespace sycl {

class SampleDistSYCLKernel;

// Deterministic fused top-k/top-p/temperature sampling. The whole op runs in a
// single work-group: the (logit, index) pairs are bitonic-sorted in shared
// local memory to the exact order the CPU reference uses (logit descending,
// index ascending — indices are unique so the order is a strict total order),
// then work-item 0 replays the host algorithm sequentially so the float
// summation order matches the reference bit for bit. Padding slots sort last
// (lowest float value, INT32_MAX index).
//
// SLM budget: ops_sycl_sample_dist_max_vocab (4096) padded pairs * 8 bytes =
// 32 KiB, within the 64 KiB limit of the target iGPU. The supports_ probe in
// ops_sycl.cpp rejects larger vocabularies.

bool ggml_sycl_op_sample_dist_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0] || !node->src[1]) {
            return false;
        }
        const ggml_tensor* logits = node->src[0];
        const ggml_tensor* uniform = node->src[1];
        if (logits->type != GGML_TYPE_F32 || uniform->type != GGML_TYPE_F32 ||
            node->type != GGML_TYPE_I32) {
            return false;
        }

        ops_sample_dist_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        const int64_t vocab = logits->ne[0];
        if (vocab <= 0 || vocab > ops_sycl_sample_dist_max_vocab) {
            return false;
        }

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        size_t padded = 1;
        while (padded < static_cast<size_t>(vocab)) {
            padded <<= 1;
        }

        constexpr size_t workgroup_size = 256;
        const float* logits_data = static_cast<const float*>(logits->data);
        const float* uniform_data = static_cast<const float*>(uniform->data);
        int32_t* out_data = static_cast<int32_t*>(node->data);
        const int32_t top_k = params.top_k;
        const float top_p = params.top_p;
        const float temperature = params.temperature;

        queue->submit([&](::sycl::handler& handler) {
            ::sycl::local_accessor<float, 1> value_l(::sycl::range<1>(padded), handler);
            ::sycl::local_accessor<int32_t, 1> index_l(::sycl::range<1>(padded), handler);
            handler.parallel_for<SampleDistSYCLKernel>(
                ::sycl::nd_range<1>(workgroup_size, workgroup_size),
                [=](::sycl::nd_item<1> item) {
                    const size_t tid = item.get_local_linear_id();
                    const size_t n = padded;

                    for (size_t i = tid; i < n; i += workgroup_size) {
                        if (i < static_cast<size_t>(vocab)) {
                            value_l[i] = logits_data[i];
                            index_l[i] = static_cast<int32_t>(i);
                        } else {
                            value_l[i] = std::numeric_limits<float>::lowest();
                            index_l[i] = std::numeric_limits<int32_t>::max();
                        }
                    }
                    item.barrier(::sycl::access::fence_space::local_space);

                    // Bitonic sort to (logit desc, index asc). The k/j loop
                    // bounds are uniform across the work-group, so every
                    // work-item reaches every barrier.
                    for (size_t k = 2; k <= n; k <<= 1) {
                        for (size_t j = k >> 1; j > 0; j >>= 1) {
                            for (size_t i = tid; i < n; i += workgroup_size) {
                                const size_t partner = i ^ j;
                                if (partner > i) {
                                    const float value_a = value_l[i];
                                    const float value_b = value_l[partner];
                                    const int32_t index_a = index_l[i];
                                    const int32_t index_b = index_l[partner];
                                    // True when the partner element belongs
                                    // before element i in the target order.
                                    const bool out_of_order =
                                        value_b > value_a ||
                                        (value_b == value_a && index_b < index_a);
                                    const bool target_order_block = (i & k) == 0;
                                    if (out_of_order == target_order_block) {
                                        value_l[i] = value_b;
                                        value_l[partner] = value_a;
                                        index_l[i] = index_b;
                                        index_l[partner] = index_a;
                                    }
                                }
                            }
                            item.barrier(::sycl::access::fence_space::local_space);
                        }
                    }

                    // Sequential tail on a single work-item: replays the host
                    // reference exactly (same accumulation order). No
                    // collectives follow, so the divergence is safe.
                    if (tid == 0) {
                        float u = uniform_data[0];
                        u = ::sycl::fmin(::sycl::fmax(u, 0.0f), 0.999999f);
                        const int32_t keep =
                            top_k > 0
                                ? (top_k < static_cast<int32_t>(vocab)
                                       ? top_k
                                       : static_cast<int32_t>(vocab))
                                : static_cast<int32_t>(vocab);

                        // Softmax over the kept candidates (max first). The
                        // sorted logits are overwritten in place with probs;
                        // max_logit is latched before the first overwrite.
                        const float max_logit = value_l[0];
                        float total = 0.0f;
                        for (int32_t i = 0; i < keep; ++i) {
                            const float p =
                                ::sycl::exp((value_l[i] - max_logit) / temperature);
                            value_l[i] = p;
                            total += p;
                        }
                        for (int32_t i = 0; i < keep; ++i) {
                            value_l[i] /= total;
                        }

                        int32_t kept = keep;
                        if (top_p < 1.0f) {
                            float cumulative = 0.0f;
                            for (int32_t i = 0; i < keep; ++i) {
                                cumulative += value_l[i];
                                if (cumulative >= top_p) {
                                    kept = i + 1;
                                    break;
                                }
                            }
                        }

                        float kept_total = 0.0f;
                        for (int32_t i = 0; i < kept; ++i) {
                            kept_total += value_l[i];
                        }
                        const float threshold = u * kept_total;
                        float cumulative = 0.0f;
                        int32_t selected = index_l[kept - 1];
                        for (int32_t i = 0; i < kept; ++i) {
                            cumulative += value_l[i];
                            if (cumulative > threshold) {
                                selected = index_l[i];
                                break;
                            }
                        }
                        *out_data = selected;
                    }
                });
        });
        return true;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL SampleDist Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL SampleDist Exception: %s\n", e.what());
        return false;
    }
}

} // namespace sycl
} // namespace ggml_ops_ext
