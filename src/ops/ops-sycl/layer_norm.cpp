#include "common.hpp" // From ggml-sycl
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"
#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

template <typename T> class LayerNormSubgroupSYCLKernel;

template <typename T>
void launch_layer_norm_sycl(::sycl::queue* queue, const T* input, const T* gamma, const T* beta, T* output,
                            int64_t width, int64_t rows, float eps) {
    constexpr size_t subgroup_size = 32;
    constexpr size_t workgroup_size = 256;
    constexpr size_t rows_per_workgroup = workgroup_size / subgroup_size;
    const size_t workgroups = (static_cast<size_t>(rows) + rows_per_workgroup - 1) / rows_per_workgroup;

    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<LayerNormSubgroupSYCLKernel<T>>(
            ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
            [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t row = static_cast<int64_t>(item.get_group_linear_id()) * rows_per_workgroup +
                                    static_cast<int64_t>(subgroup.get_group_linear_id());
                if (row >= rows) {
                    return;
                }
                const int64_t lane = subgroup.get_local_linear_id();
                const T* input_row = input + row * width;
                float local_sum = 0.0f;
                float local_squared_sum = 0.0f;
                for (int64_t column = lane; column < width; column += subgroup_size) {
                    const float value = static_cast<float>(input_row[column]);
                    local_sum += value;
                    local_squared_sum += value * value;
                }
                const float sum = ::sycl::reduce_over_group(subgroup, local_sum, ::sycl::plus<float>());
                const float squared_sum = ::sycl::reduce_over_group(subgroup, local_squared_sum, ::sycl::plus<float>());
                const float mean = sum / static_cast<float>(width);
                const float variance = ::sycl::fmax(squared_sum / static_cast<float>(width) - mean * mean, 0.0f);
                const float inverse_std = ::sycl::rsqrt(variance + eps);
                T* output_row = output + row * width;
                for (int64_t column = lane; column < width; column += subgroup_size) {
                    const float normalized = (static_cast<float>(input_row[column]) - mean) * inverse_std;
                    output_row[column] = static_cast<T>(normalized * static_cast<float>(gamma[column]) +
                                                        static_cast<float>(beta[column]));
                }
            });
    });
}

bool ggml_sycl_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* x, struct ggml_tensor* gamma,
                             struct ggml_tensor* beta, struct ggml_tensor* dst, float eps) {
    if (!x || !gamma || !beta || !dst) {
        return false;
    }

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q)
        return false;

    GGML_ASSERT(x->type == gamma->type);
    GGML_ASSERT(x->type == beta->type);
    GGML_ASSERT(x->type == dst->type);

    int64_t ne0 = dst->ne[0]; // Columns
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Batch
    int64_t ne3 = dst->ne[3];
    const int64_t rows = ne1 * ne2 * ne3;

    if (x->type == GGML_TYPE_F32) {
        launch_layer_norm_sycl(q, static_cast<const float*>(x->data), static_cast<const float*>(gamma->data),
                               static_cast<const float*>(beta->data), static_cast<float*>(dst->data), ne0, rows, eps);
    } else if (x->type == GGML_TYPE_F16) {
        launch_layer_norm_sycl(
            q, static_cast<const ::sycl::half*>(x->data), static_cast<const ::sycl::half*>(gamma->data),
            static_cast<const ::sycl::half*>(beta->data), static_cast<::sycl::half*>(dst->data), ne0, rows, eps);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps = 1e-5f;
    if (node) {
        std::memcpy(&eps, node->op_params, sizeof(float));
    }
    return ggml_sycl_op_layer_norm(backend, node ? node->src[0] : nullptr, node ? node->src[1] : nullptr,
                                   node ? node->src[2] : nullptr, node, eps);
}

// ==================== Fused AdaLN SYCL Operator ====================

class AdaLNSYCLKernelF32;
class AdaLNSYCLKernelF16;

static inline size_t get_sycl_tensor_offset(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t ne1, int64_t ne2,
                                            int64_t ne3, size_t nb0, size_t nb1, size_t nb2, size_t nb3,
                                            int64_t dst_ne2) {
    int64_t s0 = i0;
    int64_t s1 = 0;
    int64_t s2 = 0;
    int64_t s3 = 0;

    if (ne2 > 1) {
        s2 = i2 % ne2;
        s1 = i1 % ne1;
    } else if (ne1 > 1) {
        if (ne1 == dst_ne2) {
            s1 = i2;
        } else {
            s1 = i1 % ne1;
        }
    }
    if (ne3 > 1) {
        s3 = i3 % ne3;
    }

    return s3 * nb3 + s2 * nb2 + s1 * nb1 + s0 * nb0;
}

template <typename T, typename KernelName>
void launch_ada_ln_sycl(::sycl::queue* q, const T* x_d, const T* scale_d, const T* shift_d, T* dst_d, int64_t ne0,
                        int64_t ne1, int64_t ne2, int64_t ne3, float eps, size_t nb_x0, size_t nb_x1, size_t nb_x2,
                        size_t nb_x3, int64_t scale_ne1, int64_t scale_ne2, int64_t scale_ne3, size_t nb_scale0,
                        size_t nb_scale1, size_t nb_scale2, size_t nb_scale3, int64_t shift_ne1, int64_t shift_ne2,
                        int64_t shift_ne3, size_t nb_shift0, size_t nb_shift1, size_t nb_shift2, size_t nb_shift3,
                        size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3) {
    constexpr size_t subgroup_size = 32;
    constexpr size_t workgroup_size = 256;
    constexpr size_t rows_per_workgroup = workgroup_size / subgroup_size;
    const int64_t num_rows = ne1 * ne2 * ne3;
    const size_t workgroups =
        (static_cast<size_t>(num_rows) + rows_per_workgroup - 1) / rows_per_workgroup;

    q->submit([&](::sycl::handler& handler) {
        handler.parallel_for<KernelName>(
            ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
            [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t row = static_cast<int64_t>(item.get_group_linear_id()) * rows_per_workgroup +
                                    static_cast<int64_t>(subgroup.get_group_linear_id());
                if (row >= num_rows) {
                    return;
                }
                const int64_t lane = subgroup.get_local_linear_id();
                const int64_t i1 = row % ne1;
                const int64_t remainder = row / ne1;
                const int64_t i2 = remainder % ne2;
                const int64_t i3 = remainder / ne2;
                const auto* input_row = reinterpret_cast<const T*>(
                    reinterpret_cast<const char*>(x_d) + i3 * nb_x3 + i2 * nb_x2 + i1 * nb_x1);
                auto* output_row = reinterpret_cast<T*>(
                    reinterpret_cast<char*>(dst_d) + i3 * nb_dst3 + i2 * nb_dst2 + i1 * nb_dst1);
                const size_t scale_base = get_sycl_tensor_offset(0, i1, i2, i3, scale_ne1, scale_ne2,
                                                                  scale_ne3, nb_scale0, nb_scale1, nb_scale2,
                                                                  nb_scale3, ne2);
                const size_t shift_base = get_sycl_tensor_offset(0, i1, i2, i3, shift_ne1, shift_ne2,
                                                                  shift_ne3, nb_shift0, nb_shift1, nb_shift2,
                                                                  nb_shift3, ne2);
                float local_sum = 0.0f;
                float local_squared_sum = 0.0f;
                for (int64_t column = lane; column < ne0; column += subgroup_size) {
                    const float value = static_cast<float>(*reinterpret_cast<const T*>(
                        reinterpret_cast<const char*>(input_row) + column * nb_x0));
                    local_sum += value;
                    local_squared_sum += value * value;
                }
                const float sum = ::sycl::reduce_over_group(subgroup, local_sum, ::sycl::plus<float>());
                const float squared_sum =
                    ::sycl::reduce_over_group(subgroup, local_squared_sum, ::sycl::plus<float>());
                const float mean = sum / static_cast<float>(ne0);
                const float variance =
                    ::sycl::fmax(squared_sum / static_cast<float>(ne0) - mean * mean, 0.0f);
                const float inverse_std = ::sycl::rsqrt(variance + eps);
                for (int64_t column = lane; column < ne0; column += subgroup_size) {
                    const float value = static_cast<float>(*reinterpret_cast<const T*>(
                        reinterpret_cast<const char*>(input_row) + column * nb_x0));
                    const float scale_value = static_cast<float>(*reinterpret_cast<const T*>(
                        reinterpret_cast<const char*>(scale_d) + scale_base + column * nb_scale0));
                    const float shift_value = static_cast<float>(*reinterpret_cast<const T*>(
                        reinterpret_cast<const char*>(shift_d) + shift_base + column * nb_shift0));
                    *reinterpret_cast<T*>(reinterpret_cast<char*>(output_row) + column * nb_dst0) =
                        static_cast<T>((value - mean) * inverse_std * (1.0f + scale_value) + shift_value);
                }
            });
    });
}

bool ggml_sycl_op_ada_ln(ggml_backend_t backend, struct ggml_tensor* x, struct ggml_tensor* scale,
                         struct ggml_tensor* shift, struct ggml_tensor* dst, float eps) {
    if (!x || !scale || !shift || !dst) {
        return false;
    }

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q)
        return false;

    GGML_ASSERT(x->type == scale->type);
    GGML_ASSERT(x->type == shift->type);
    GGML_ASSERT(x->type == dst->type);

    int64_t ne0 = dst->ne[0];
    int64_t ne1 = dst->ne[1];
    int64_t ne2 = dst->ne[2];
    int64_t ne3 = dst->ne[3];

    if (x->type == GGML_TYPE_F32) {
        launch_ada_ln_sycl<float, AdaLNSYCLKernelF32>(
            q, (const float*)x->data, (const float*)scale->data, (const float*)shift->data, (float*)dst->data, ne0, ne1,
            ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3], scale->ne[1], scale->ne[2], scale->ne[3],
            scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3], shift->ne[1], shift->ne[2], shift->ne[3],
            shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else if (x->type == GGML_TYPE_F16) {
        launch_ada_ln_sycl<::sycl::half, AdaLNSYCLKernelF16>(
            q, (const ::sycl::half*)x->data, (const ::sycl::half*)scale->data, (const ::sycl::half*)shift->data,
            (::sycl::half*)dst->data, ne0, ne1, ne2, ne3, eps, x->nb[0], x->nb[1], x->nb[2], x->nb[3], scale->ne[1],
            scale->ne[2], scale->ne[3], scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3], shift->ne[1],
            shift->ne[2], shift->ne[3], shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3], dst->nb[0], dst->nb[1],
            dst->nb[2], dst->nb[3]);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps = 1e-5f;
    if (node) {
        std::memcpy(&eps, node->op_params, sizeof(float));
    }
    return ggml_sycl_op_ada_ln(backend, node ? node->src[0] : nullptr, node ? node->src[1] : nullptr,
                               node ? node->src[2] : nullptr, node, eps);
}

} // namespace sycl
} // namespace ggml_ops_ext
