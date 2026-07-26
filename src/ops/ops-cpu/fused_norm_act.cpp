#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

inline float apply_activation(float value, int32_t activation) {
    switch (activation) {
        case 0: // silu
            return value / (1.0f + std::exp(-value));
        case 1: // gelu (tanh approximation, matching ggml)
            return 0.5f * value *
                   (1.0f + std::tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
        case 2: // relu
            return value > 0.0f ? value : 0.0f;
        default: // identity
            return value;
    }
}

template <typename T>
void compute_rows(const T* x, const T* gamma, const T* beta, const T* residual, T* dst,
                  int64_t width, int64_t rows, float eps, int32_t activation, int threads) {
#pragma omp parallel num_threads(threads)
    {
        std::vector<float> row(static_cast<size_t>(width));
#pragma omp for schedule(static)
        for (int64_t index = 0; index < rows; ++index) {
            const T* x_row = x + index * width;
            const T* residual_row = residual ? residual + index * width : nullptr;
            T* dst_row = dst + index * width;
            float mean = 0.0f;
            for (int64_t i = 0; i < width; ++i) {
                float value = read_val(x_row + i);
                if (residual_row) value += read_val(residual_row + i);
                row[i] = value;
                mean += value;
            }
            mean /= static_cast<float>(width);
            float variance = 0.0f;
            for (int64_t i = 0; i < width; ++i) {
                const float centered = row[i] - mean;
                variance += centered * centered;
            }
            variance /= static_cast<float>(width);
            const float inv_std = 1.0f / std::sqrt(variance + eps);
            for (int64_t i = 0; i < width; ++i) {
                const float normed =
                    (row[i] - mean) * inv_std * read_val(gamma + i) + read_val(beta + i);
                write_val(dst_row + i, apply_activation(normed, activation));
            }
        }
    }
}

} // namespace

bool ops_cpu_op_fused_norm_act(ggml_backend_t backend, ggml_tensor* node) {
    ggml_tensor* x = node->src[0];
    ggml_tensor* gamma = node->src[1];
    ggml_tensor* beta = node->src[2];
    ggml_tensor* residual = node->src[3];
    ops_fused_norm_act_params params;
    std::memcpy(&params, node->op_params, sizeof(params));

    const int64_t width = x->ne[0];
    const int64_t rows = ggml_nelements(x) / width;
    const int threads = backend_thread_count(backend);

    if (x->type == GGML_TYPE_F32) {
        compute_rows(static_cast<const float*>(x->data), static_cast<const float*>(gamma->data),
                     static_cast<const float*>(beta->data),
                     residual ? static_cast<const float*>(residual->data) : nullptr,
                     static_cast<float*>(node->data), width, rows, params.eps, params.activation,
                     threads);
        return true;
    }
    if (x->type == GGML_TYPE_F16) {
        compute_rows(static_cast<const ggml_fp16_t*>(x->data),
                     static_cast<const ggml_fp16_t*>(gamma->data),
                     static_cast<const ggml_fp16_t*>(beta->data),
                     residual ? static_cast<const ggml_fp16_t*>(residual->data) : nullptr,
                     static_cast<ggml_fp16_t*>(node->data), width, rows, params.eps,
                     params.activation, threads);
        return true;
    }
    return false;
}

} // namespace ggml_ops_ext::cpu
