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

template <typename T> class FusedNormActSYCLKernel;

// Activation ids match ggml_ops_gate_activation (0 = silu, 1 = gelu tanh
// approximation, 2 = relu, everything else = identity). Float-only math: the
// target iGPU has no fp64 support, so every literal stays a float.
static inline float fused_norm_act_apply(float value, int32_t activation) {
    switch (activation) {
        case 0: // silu
            return value / (1.0f + ::sycl::exp(-value));
        case 1: // gelu (tanh approximation, matching ggml)
            return 0.5f * value *
                   (1.0f + ::sycl::tanh(0.7978845608028654f *
                                        (value + 0.044715f * value * value * value)));
        case 2: // relu
            return value > 0.0f ? value : 0.0f;
        default: // identity
            return value;
    }
}

// One sub-group per row, mirroring layer_norm.cpp. The mean/variance use the
// same two-pass formulation as the CPU reference kernel so results agree
// within rounding of the reduction order.
template <typename T>
static void launch_fused_norm_act(::sycl::queue* queue, const T* x, const T* gamma, const T* beta,
                                  const T* residual, T* dst, int64_t width, int64_t rows,
                                  float eps, int32_t activation) {
    constexpr size_t subgroup_size = 32;
    constexpr size_t workgroup_size = 256;
    constexpr size_t rows_per_workgroup = workgroup_size / subgroup_size;
    const size_t workgroups =
        (static_cast<size_t>(rows) + rows_per_workgroup - 1) / rows_per_workgroup;

    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<FusedNormActSYCLKernel<T>>(
            ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
            [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t row =
                    static_cast<int64_t>(item.get_group_linear_id()) * rows_per_workgroup +
                    static_cast<int64_t>(subgroup.get_group_linear_id());
                if (row >= rows) {
                    // Uniform across the whole sub-group: every collective
                    // below is sub-group scoped, so no live lane diverges.
                    return;
                }
                const int64_t lane = subgroup.get_local_linear_id();
                const T* x_row = x + row * width;
                const T* residual_row = residual ? residual + row * width : nullptr;
                T* dst_row = dst + row * width;

                float local_sum = 0.0f;
                for (int64_t column = lane; column < width; column += subgroup_size) {
                    float value = static_cast<float>(x_row[column]);
                    if (residual_row) {
                        value += static_cast<float>(residual_row[column]);
                    }
                    local_sum += value;
                }
                const float mean =
                    ::sycl::reduce_over_group(subgroup, local_sum, ::sycl::plus<float>()) /
                    static_cast<float>(width);

                float local_squared = 0.0f;
                for (int64_t column = lane; column < width; column += subgroup_size) {
                    float value = static_cast<float>(x_row[column]);
                    if (residual_row) {
                        value += static_cast<float>(residual_row[column]);
                    }
                    const float centered = value - mean;
                    local_squared += centered * centered;
                }
                const float variance =
                    ::sycl::reduce_over_group(subgroup, local_squared, ::sycl::plus<float>()) /
                    static_cast<float>(width);
                const float inv_std = 1.0f / ::sycl::sqrt(variance + eps);

                for (int64_t column = lane; column < width; column += subgroup_size) {
                    float value = static_cast<float>(x_row[column]);
                    if (residual_row) {
                        value += static_cast<float>(residual_row[column]);
                    }
                    const float normed = (value - mean) * inv_std *
                                             static_cast<float>(gamma[column]) +
                                         static_cast<float>(beta[column]);
                    dst_row[column] = static_cast<T>(fused_norm_act_apply(normed, activation));
                }
            });
    });
}

bool ggml_sycl_op_fused_norm_act_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0] || !node->src[1] || !node->src[2]) {
            return false;
        }
        ggml_tensor* x = node->src[0];
        ggml_tensor* gamma = node->src[1];
        ggml_tensor* beta = node->src[2];
        ggml_tensor* residual = node->src[3]; // optional, may be null

        ops_fused_norm_act_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        const int64_t width = x->ne[0];
        const int64_t rows = ggml_nelements(x) / width;

        if (x->type == GGML_TYPE_F32) {
            launch_fused_norm_act(queue, static_cast<const float*>(x->data),
                                  static_cast<const float*>(gamma->data),
                                  static_cast<const float*>(beta->data),
                                  residual ? static_cast<const float*>(residual->data) : nullptr,
                                  static_cast<float*>(node->data), width, rows, params.eps,
                                  params.activation);
            return true;
        }
        if (x->type == GGML_TYPE_F16) {
            launch_fused_norm_act(
                queue, static_cast<const ::sycl::half*>(x->data),
                static_cast<const ::sycl::half*>(gamma->data),
                static_cast<const ::sycl::half*>(beta->data),
                residual ? static_cast<const ::sycl::half*>(residual->data) : nullptr,
                static_cast<::sycl::half*>(node->data), width, rows, params.eps,
                params.activation);
            return true;
        }
        return false;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL FusedNormAct Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL FusedNormAct Exception: %s\n", e.what());
        return false;
    }
}

} // namespace sycl
} // namespace ggml_ops_ext
