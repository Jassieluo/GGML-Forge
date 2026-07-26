#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>

namespace ggml_ops_ext::cpu {
namespace {

template <typename T>
void add_pos_encoding(const T* x, T* dst, int64_t width, int64_t tokens, int64_t outer,
                      int64_t start, float base, int threads) {
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t t = 0; t < tokens; ++t) {
            const T* x_row = x + (o * tokens + t) * width;
            T* dst_row = dst + (o * tokens + t) * width;
            const float position = static_cast<float>(start + t);
            for (int64_t i = 0; i < width; i += 2) {
                const float angle =
                    position * std::exp(-std::log(base) * static_cast<float>(i) /
                                        static_cast<float>(width));
                write_val(dst_row + i, read_val(x_row + i) + std::sin(angle));
                if (i + 1 < width) {
                    write_val(dst_row + i + 1, read_val(x_row + i + 1) + std::cos(angle));
                }
            }
        }
    }
}

} // namespace

bool ops_cpu_op_pos_encoding(ggml_backend_t backend, ggml_tensor* node) {
    ggml_tensor* x = node->src[0];
    ggml_tensor* position = node->src[1];
    ops_pos_encoding_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t width = x->ne[0];
    const int64_t tokens = x->ne[1];
    const int64_t outer = x->ne[2] * x->ne[3];
    const int64_t start = params.offset +
        (position ? *static_cast<const int32_t*>(position->data) : 0);
    const int threads = backend_thread_count(backend);

    if (x->type == GGML_TYPE_F32) {
        add_pos_encoding(static_cast<const float*>(x->data), static_cast<float*>(node->data),
                         width, tokens, outer, start, params.base, threads);
        return true;
    }
    if (x->type == GGML_TYPE_F16) {
        add_pos_encoding(static_cast<const ggml_fp16_t*>(x->data),
                         static_cast<ggml_fp16_t*>(node->data), width, tokens, outer, start,
                         params.base, threads);
        return true;
    }
    return false;
}

} // namespace ggml_ops_ext::cpu
