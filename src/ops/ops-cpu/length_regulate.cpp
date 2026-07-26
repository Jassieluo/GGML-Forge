#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cstring>

namespace ggml_ops_ext::cpu {

bool ops_cpu_op_length_regulate(ggml_backend_t backend, ggml_tensor* node) {
    (void)backend;
    const ggml_tensor* x = node->src[0];
    const ggml_tensor* durations = node->src[1];
    ops_length_regulate_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t channels = x->ne[0];
    const int64_t frames = x->ne[1];
    const int64_t total = params.total;
    const float* in = static_cast<const float*>(x->data);
    const int32_t* dur = static_cast<const int32_t*>(durations->data);
    float* out = static_cast<float*>(node->data);
    const size_t frame_bytes = static_cast<size_t>(channels) * sizeof(float);

    int64_t pos = 0;
    for (int64_t t = 0; t < frames && pos < total; ++t) {
        const int32_t count = dur[t];
        for (int32_t k = 0; k < count && pos < total; ++k) {
            std::memcpy(out + pos * channels, in + t * channels, frame_bytes);
            ++pos;
        }
    }
    if (pos < total) {
        std::memset(out + pos * channels,
                    0, static_cast<size_t>(total - pos) * frame_bytes);
    }
    return true;
}

} // namespace ggml_ops_ext::cpu
