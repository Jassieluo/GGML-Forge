#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

class LayerNormSYCLKernelF32;
class LayerNormSYCLKernelF16;

template <typename T, typename KernelName>
void launch_layer_norm_sycl(
    ::sycl::queue* q, const T* x_d, const T* gamma_d, const T* beta_d, T* dst_d,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    float eps,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    size_t nb_gamma0, size_t nb_beta0,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t num_rows = ne1 * ne2 * ne3;
    size_t group_size = 256;

    // Adjust group size if ne0 is very small
    if (ne0 < 256) {
        if (ne0 <= 32) group_size = 32;
        else if (ne0 <= 64) group_size = 64;
        else if (ne0 <= 128) group_size = 128;
    }

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            ::sycl::nd_range<1>(::sycl::range<1>(num_rows * group_size), ::sycl::range<1>(group_size)),
            [=](::sycl::nd_item<1> item) {
                auto grp = item.get_group();
                int64_t bid = grp.get_group_linear_id();
                int64_t lid = item.get_local_id(0);

                int64_t i1 = bid % ne1;
                int64_t tmp = bid / ne1;
                int64_t i2 = tmp % ne2;
                int64_t i3 = tmp / ne2;

                if (i3 < ne3) {
                    // 1. Collaborative calculation of mean
                    float local_sum = 0.0f;
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        local_sum += (float)*px;
                    }

                    float sum = ::sycl::reduce_over_group(grp, local_sum, ::sycl::plus<float>());
                    float mean = sum / ne0;

                    // 2. Collaborative calculation of variance
                    float local_sum_sq = 0.0f;
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        float diff = (float)*px - mean;
                        local_sum_sq += diff * diff;
                    }

                    float sum_sq = ::sycl::reduce_over_group(grp, local_sum_sq, ::sycl::plus<float>());
                    float variance = sum_sq / ne0;
                    float inv_std = 1.0f / ::sycl::sqrt(variance + eps);

                    // 3. Collaborative normalize, scale, and shift
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        const T* pgamma = (const T*)((const char*)gamma_d + i0*nb_gamma0);
                        const T* pbeta = (const T*)((const char*)beta_d + i0*nb_beta0);

                        T* pdst = (T*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                        float norm_val = ((float)*px - mean) * inv_std;
                        *pdst = (T)(norm_val * (float)*pgamma + (float)*pbeta);
                    }
                }
            }
        );
    });
}

bool ggml_sycl_op_layer_norm(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* gamma,
    struct ggml_tensor* beta,
    struct ggml_tensor* dst,
    float eps
) {
    if (!x || !gamma || !beta || !dst) {
        return false;
    }

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    GGML_ASSERT(x->type == gamma->type);
    GGML_ASSERT(x->type == beta->type);
    GGML_ASSERT(x->type == dst->type);

    int64_t ne0 = dst->ne[0]; // Columns
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Batch
    int64_t ne3 = dst->ne[3];

    if (x->type == GGML_TYPE_F32) {
        launch_layer_norm_sycl<float, LayerNormSYCLKernelF32>(
            q, (const float*)x->data, (const float*)gamma->data, (const float*)beta->data, (float*)dst->data,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            gamma->nb[0], beta->nb[0],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    } else if (x->type == GGML_TYPE_F16) {
        launch_layer_norm_sycl<::sycl::half, LayerNormSYCLKernelF16>(
            q, (const ::sycl::half*)x->data, (const ::sycl::half*)gamma->data, (const ::sycl::half*)beta->data, (::sycl::half*)dst->data,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            gamma->nb[0], beta->nb[0],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    } else {
        return false;
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_layer_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps = 1e-5f;
    if (node) {
        std::memcpy(&eps, node->op_params, sizeof(float));
    }
    return ggml_sycl_op_layer_norm(backend, node ? node->src[0] : nullptr, node ? node->src[1] : nullptr, node ? node->src[2] : nullptr, node, eps);
}

// ==================== Fused AdaLN SYCL Operator ====================

class AdaLNSYCLKernelF32;
class AdaLNSYCLKernelF16;

static inline size_t get_sycl_tensor_offset(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t ne1, int64_t ne2, int64_t ne3,
    size_t nb0, size_t nb1, size_t nb2, size_t nb3,
    int64_t dst_ne2
) {
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
void launch_ada_ln_sycl(
    ::sycl::queue* q, const T* x_d, const T* scale_d, const T* shift_d, T* dst_d,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
    float eps,
    size_t nb_x0, size_t nb_x1, size_t nb_x2, size_t nb_x3,
    int64_t scale_ne1, int64_t scale_ne2, int64_t scale_ne3,
    size_t nb_scale0, size_t nb_scale1, size_t nb_scale2, size_t nb_scale3,
    int64_t shift_ne1, int64_t shift_ne2, int64_t shift_ne3,
    size_t nb_shift0, size_t nb_shift1, size_t nb_shift2, size_t nb_shift3,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2, size_t nb_dst3
) {
    int64_t num_rows = ne1 * ne2 * ne3;
    size_t group_size = 256;

    if (ne0 < 256) {
        if (ne0 <= 32) group_size = 32;
        else if (ne0 <= 64) group_size = 64;
        else if (ne0 <= 128) group_size = 128;
    }

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            ::sycl::nd_range<1>(::sycl::range<1>(num_rows * group_size), ::sycl::range<1>(group_size)),
            [=](::sycl::nd_item<1> item) {
                auto grp = item.get_group();
                int64_t bid = grp.get_group_linear_id();
                int64_t lid = item.get_local_id(0);

                int64_t i1 = bid % ne1;
                int64_t tmp = bid / ne1;
                int64_t i2 = tmp % ne2;
                int64_t i3 = tmp / ne2;

                if (i3 < ne3) {
                    // 1. Collaborative calculation of mean
                    float local_sum = 0.0f;
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        local_sum += (float)*px;
                    }

                    float sum = ::sycl::reduce_over_group(grp, local_sum, ::sycl::plus<float>());
                    float mean = sum / ne0;

                    // 2. Collaborative calculation of variance
                    float local_sum_sq = 0.0f;
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        float diff = (float)*px - mean;
                        local_sum_sq += diff * diff;
                    }

                    float sum_sq = ::sycl::reduce_over_group(grp, local_sum_sq, ::sycl::plus<float>());
                    float variance = sum_sq / ne0;
                    float inv_std = 1.0f / ::sycl::sqrt(variance + eps);

                    // 3. Collaborative normalize, scale, and shift
                    for (int64_t i0 = lid; i0 < ne0; i0 += group_size) {
                        const T* px = (const T*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                        
                        size_t scale_offset = get_sycl_tensor_offset(
                            i0, i1, i2, i3,
                            scale_ne1, scale_ne2, scale_ne3,
                            nb_scale0, nb_scale1, nb_scale2, nb_scale3,
                            ne2
                        );
                        size_t shift_offset = get_sycl_tensor_offset(
                            i0, i1, i2, i3,
                            shift_ne1, shift_ne2, shift_ne3,
                            nb_shift0, nb_shift1, nb_shift2, nb_shift3,
                            ne2
                        );

                        const T* pscale = (const T*)((const char*)scale_d + scale_offset);
                        const T* pshift = (const T*)((const char*)shift_d + shift_offset);

                        T* pdst = (T*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);

                        float norm_val = ((float)*px - mean) * inv_std;
                        *pdst = (T)(norm_val * (1.0f + (float)*pscale) + (float)*pshift);
                    }
                }
            }
        );
    });
}

bool ggml_sycl_op_ada_ln(
    ggml_backend_t backend,
    struct ggml_tensor* x,
    struct ggml_tensor* scale,
    struct ggml_tensor* shift,
    struct ggml_tensor* dst,
    float eps
) {
    if (!x || !scale || !shift || !dst) {
        return false;
    }

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    GGML_ASSERT(x->type == scale->type);
    GGML_ASSERT(x->type == shift->type);
    GGML_ASSERT(x->type == dst->type);

    int64_t ne0 = dst->ne[0];
    int64_t ne1 = dst->ne[1];
    int64_t ne2 = dst->ne[2];
    int64_t ne3 = dst->ne[3];

    if (x->type == GGML_TYPE_F32) {
        launch_ada_ln_sycl<float, AdaLNSYCLKernelF32>(
            q, (const float*)x->data, (const float*)scale->data, (const float*)shift->data, (float*)dst->data,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            scale->ne[1], scale->ne[2], scale->ne[3],
            scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
            shift->ne[1], shift->ne[2], shift->ne[3],
            shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    } else if (x->type == GGML_TYPE_F16) {
        launch_ada_ln_sycl<::sycl::half, AdaLNSYCLKernelF16>(
            q, (const ::sycl::half*)x->data, (const ::sycl::half*)scale->data, (const ::sycl::half*)shift->data, (::sycl::half*)dst->data,
            ne0, ne1, ne2, ne3,
            eps,
            x->nb[0], x->nb[1], x->nb[2], x->nb[3],
            scale->ne[1], scale->ne[2], scale->ne[3],
            scale->nb[0], scale->nb[1], scale->nb[2], scale->nb[3],
            shift->ne[1], shift->ne[2], shift->ne[3],
            shift->nb[0], shift->nb[1], shift->nb[2], shift->nb[3],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
        );
    } else {
        return false;
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_ada_ln_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    float eps = 1e-5f;
    if (node) {
        std::memcpy(&eps, node->op_params, sizeof(float));
    }
    return ggml_sycl_op_ada_ln(backend, node ? node->src[0] : nullptr, node ? node->src[1] : nullptr, node ? node->src[2] : nullptr, node, eps);
}

} // namespace sycl
} // namespace ggml_ops_ext
