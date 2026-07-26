#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

namespace ggml_ops_ext::cpu {

// Deterministic fused sampling. Ordering ties break by lower index so every
// backend that mirrors this algorithm picks identical tokens for identical
// inputs.
bool ops_cpu_op_sample_dist(ggml_backend_t backend, ggml_tensor* node) {
    (void)backend;
    const ggml_tensor* logits = node->src[0];
    const ggml_tensor* uniform = node->src[1];
    ops_sample_dist_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t vocab = logits->ne[0];
    const float* values = static_cast<const float*>(logits->data);
    float u = *static_cast<const float*>(uniform->data);
    u = std::min(std::max(u, 0.0f), 0.999999f);

    std::vector<int32_t> candidates(static_cast<size_t>(vocab));
    std::iota(candidates.begin(), candidates.end(), 0);
    const auto by_logit_desc = [&](int32_t a, int32_t b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    };
    const int64_t keep = params.top_k > 0 ? std::min<int64_t>(params.top_k, vocab) : vocab;
    if (keep < vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + keep, candidates.end(),
                          by_logit_desc);
        candidates.resize(static_cast<size_t>(keep));
    } else {
        std::sort(candidates.begin(), candidates.end(), by_logit_desc);
    }

    // Softmax over the kept candidates (descending order, max first).
    std::vector<float> probs(candidates.size());
    const float max_logit = values[candidates[0]];
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((values[candidates[i]] - max_logit) / params.temperature);
        total += probs[i];
    }
    for (float& p : probs) p /= total;

    size_t kept = candidates.size();
    if (params.top_p < 1.0f) {
        float cumulative = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= params.top_p) {
                kept = i + 1;
                break;
            }
        }
    }

    float kept_total = 0.0f;
    for (size_t i = 0; i < kept; ++i) kept_total += probs[i];
    const float threshold = u * kept_total;
    float cumulative = 0.0f;
    int32_t selected = candidates[kept - 1];
    for (size_t i = 0; i < kept; ++i) {
        cumulative += probs[i];
        if (cumulative > threshold) {
            selected = candidates[i];
            break;
        }
    }
    *static_cast<int32_t*>(node->data) = selected;
    return true;
}

} // namespace ggml_ops_ext::cpu
