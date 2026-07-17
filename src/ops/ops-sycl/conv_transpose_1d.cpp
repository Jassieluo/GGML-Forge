#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include "quantized_conv.h"
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <algorithm>

namespace ggml_ops_ext {
namespace sycl {

template <typename T_w, typename T_x, typename T_dst>
class DirectConvTranspose1DSYCLKernel;

template <typename SrcT, typename DstT>
class ConvTransposeCastKernel;

template <typename T>
class ConvTransposeCol2ImKernel;

template <typename SrcT, typename DstT>
static void cast_sycl(::sycl::queue* q, const SrcT* src, DstT* dst, int64_t count) {
    constexpr int64_t block_size = 256;
    const int64_t global_size = ((count + block_size - 1) / block_size) * block_size;
    q->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<ConvTransposeCastKernel<SrcT, DstT>>(
            ::sycl::nd_range<1>(global_size, block_size),
            [=](::sycl::nd_item<1> item) {
                const int64_t i = item.get_global_id(0);
                if (i < count) dst[i] = static_cast<DstT>(src[i]);
            }
        );
    });
}

template <typename T>
static void col2im_sycl(
    ::sycl::queue* q,
    const T* col,
    const void* bias,
    int bias_type,
    T* dst,
    int64_t C_out,
    int64_t L_in,
    int64_t L_out,
    int64_t kW,
    int stride,
    int padding,
    int dilation,
    size_t nb_dst0,
    size_t nb_dst1
) {
    const int64_t total = C_out * L_out;
    constexpr int64_t block_size = 256;
    const int64_t global_size = ((total + block_size - 1) / block_size) * block_size;
    q->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<ConvTransposeCol2ImKernel<T>>(
            ::sycl::nd_range<1>(global_size, block_size),
            [=](::sycl::nd_item<1> item) {
                const int64_t idx = item.get_global_id(0);
                if (idx >= total) return;

                const int64_t ow = idx % L_out;
                const int64_t c = idx / L_out;
                float sum = 0.0f;
                for (int64_t ik = 0; ik < kW; ++ik) {
                    const int64_t iw_stride = ow + padding - ik * dilation;
                    if (iw_stride >= 0 && iw_stride % stride == 0) {
                        const int64_t iw = iw_stride / stride;
                        if (iw < L_in) {
                            sum += static_cast<float>(col[iw * (C_out * kW) + c * kW + ik]);
                        }
                    }
                }

                if (bias) {
                    sum += bias_type == 0
                        ? static_cast<const float*>(bias)[c]
                        : static_cast<float>(static_cast<const ::sycl::half*>(bias)[c]);
                }
                T* out = reinterpret_cast<T*>(reinterpret_cast<char*>(dst) + c * nb_dst1 + ow * nb_dst0);
                *out = static_cast<T>(sum);
            }
        );
    });
}

template <typename T>
static void gemm_conv_transpose_sycl(
    ::sycl::queue* q,
    const T* w,
    const T* x,
    T* col,
    int64_t C_out,
    int64_t C_in,
    int64_t L_in,
    int64_t kW
) {
    const int64_t rows = C_out * kW;
    oneapi::mkl::blas::column_major::gemm(
        *q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans,
        rows, L_in, C_in,
        T(1), w, rows,
        x, L_in,
        T(0), col, rows
    );
}

template <typename T_w, typename T_x, typename T_dst>
static void direct_conv_transpose_1d_sycl_template(
    ::sycl::queue* q,
    const void* w,
    const void* x,
    const void* bias,
    int bias_type,
    void* dst,
    int64_t C_in, int64_t L_in, int64_t L_out, int64_t kW, int64_t C_out, int64_t batch,
    int stride, int padding, int dilation,
    size_t nb_w0, size_t nb_w1, size_t nb_w2,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    size_t nb_dst0, size_t nb_dst1, size_t nb_dst2
) {
    int64_t total = batch * C_out * L_out;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<DirectConvTranspose1DSYCLKernel<T_w, T_x, T_dst>>(::sycl::range<1>(total), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t ow = idx % L_out;
            int64_t tmp = idx / L_out;
            int64_t c = tmp % C_out;
            int64_t n = tmp / C_out;

            float sum = 0.0f;
            for (int64_t ik = 0; ik < kW; ++ik) {
                int64_t iw_stride = ow + padding - ik * dilation;
                if (iw_stride % stride == 0) {
                    int64_t iw = iw_stride / stride;
                    if (iw >= 0 && iw < L_in) {
                        for (int64_t ic = 0; ic < C_in; ++ic) {
                            T_w val_w_typed = *(const T_w*)((const char*)w + ik * nb_w0 + c * nb_w1 + ic * nb_w2);
                            T_x val_x_typed = *(const T_x*)((const char*)x + iw * nb_x0 + ic * nb_x1 + n * nb_x2);
                            sum += (float)val_w_typed * (float)val_x_typed;
                        }
                    }
                }
            }
            float b_val = 0.0f;
            if (bias) {
                if (bias_type == 0) {
                    b_val = ((const float*)bias)[c];
                } else {
                    b_val = (float)((const ::sycl::half*)bias)[c];
                }
            }
            T_dst* pdst = (T_dst*)((char*)dst + ow * nb_dst0 + c * nb_dst1 + n * nb_dst2);
            *pdst = (T_dst)(sum + b_val);
        });
    });
}

bool ggml_sycl_op_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* bias,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation,
    int groups
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    int64_t L_in = x->ne[0];
    int64_t C_in = x->ne[1];
    int64_t batch = (x->ne[2] > 0) ? x->ne[2] : 1;

    ops_conv_weight_desc weight_desc = {};
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, w, x, groups, weight_desc)) return false;
    int64_t kW = weight_desc.kernel;
    int64_t C_out = weight_desc.output_channels_per_group;
    int64_t L_out = dst->ne[0];

    const void* bias_data = bias ? bias->data : nullptr;
    int bias_type = bias ? ((bias->type == GGML_TYPE_F32) ? 0 : 1) : 0;

    int64_t C_in_group = C_in / groups;
    int64_t C_out_group = C_out;

    if (ggml_is_quantized(w->type)) {
#define LAUNCH_DIRECT_QUANT_CONVT_SYCL(weight_type, value_type) \
        launch_quantized_conv_transpose_1d_direct_sycl<weight_type, value_type>( \
            q, w->data, static_cast<const value_type*>(x->data), bias_data, bias_type, \
            static_cast<value_type*>(dst->data), L_in, L_out, C_in, C_out, kW, batch, \
            stride, padding, dilation, groups, x->nb[0], x->nb[1], x->nb[2], \
            dst->nb[0], dst->nb[1], dst->nb[2])
        if (x->type == GGML_TYPE_F16) {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q4_0, ::sycl::half); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q4_K, ::sycl::half); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q8_0, ::sycl::half); break;
                default: return false;
            }
        } else {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q4_0, float); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q4_K, float); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONVT_SYCL(GGML_TYPE_Q8_0, float); break;
                default: return false;
            }
        }
#undef LAUNCH_DIRECT_QUANT_CONVT_SYCL
        return true;
    }

    size_t bias_elem_size = bias ? ((bias->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(::sycl::half)) : sizeof(float);

    ggml_type w_storage_type = w->type;
    const void* prepared_w = w->data;
    size_t prepared_nb0 = w->nb[0];
    size_t prepared_nb1 = w->nb[1];
    size_t prepared_nb2 = w->nb[2];

    const size_t w_element_size = ggml_type_size(w_storage_type);
    const size_t x_element_size = ggml_type_size(x->type);
    const bool dense_inputs = prepared_nb0 == w_element_size &&
                              prepared_nb1 == static_cast<size_t>(kW) * w_element_size &&
                              prepared_nb2 == static_cast<size_t>(kW * C_out) * w_element_size &&
                              x->nb[0] == x_element_size &&
                              x->nb[1] == static_cast<size_t>(L_in) * x_element_size;
    const bool use_gemm = dense_inputs && C_in_group >= 16 && C_out_group * kW >= 16 && L_in >= 16;
    ops_sycl_pool_alloc<float> w_f32(backend);
    const void* gemm_w = prepared_w;
    if (use_gemm && w_storage_type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32) {
        if (!w_f32.alloc(ggml_nelements(w))) return false;
        cast_sycl(q, static_cast<const ::sycl::half*>(prepared_w), w_f32.get(), ggml_nelements(w));
        gemm_w = w_f32.get();
    }

    ops_sycl_pool_alloc<float> col_f32(backend);
    ops_sycl_pool_alloc<::sycl::half> col_f16(backend);
    if (use_gemm) {
        const size_t col_elements = C_out_group * kW * L_in;
        if (x->type == GGML_TYPE_F32) {
            if (!col_f32.alloc(col_elements)) return false;
        } else {
            if (!col_f16.alloc(col_elements)) return false;
        }
    }

    for (int g = 0; g < groups; ++g) {
        const void* w_g = (const char*)prepared_w + g * C_in_group * prepared_nb2;
        const void* x_g = (const char*)x->data + g * C_in_group * x->nb[1];
        void* dst_g = (char*)dst->data + g * C_out_group * dst->nb[1];
        const void* bias_g = bias_data ? (const char*)bias_data + g * C_out_group * bias_elem_size : nullptr;

        if (use_gemm) {
            const size_t w_group_offset = g * C_in_group * C_out_group * kW;
            const void* gemm_w_g = x->type == GGML_TYPE_F32
                ? static_cast<const void*>(static_cast<const float*>(gemm_w) + w_group_offset)
                : static_cast<const void*>(static_cast<const ::sycl::half*>(gemm_w) + w_group_offset);

            for (int64_t n = 0; n < batch; ++n) {
                const void* x_n = static_cast<const char*>(x_g) + n * x->nb[2];
                void* dst_n = static_cast<char*>(dst_g) + n * dst->nb[2];
                if (x->type == GGML_TYPE_F32) {
                    gemm_conv_transpose_sycl(
                        q, static_cast<const float*>(gemm_w_g), static_cast<const float*>(x_n), col_f32.get(),
                        C_out_group, C_in_group, L_in, kW
                    );
                    col2im_sycl(
                        q, col_f32.get(), bias_g, bias_type, static_cast<float*>(dst_n),
                        C_out_group, L_in, L_out, kW, stride, padding, dilation, dst->nb[0], dst->nb[1]
                    );
                } else {
                    gemm_conv_transpose_sycl(
                        q, static_cast<const ::sycl::half*>(gemm_w_g), static_cast<const ::sycl::half*>(x_n), col_f16.get(),
                        C_out_group, C_in_group, L_in, kW
                    );
                    col2im_sycl(
                        q, col_f16.get(), bias_g, bias_type, static_cast<::sycl::half*>(dst_n),
                        C_out_group, L_in, L_out, kW, stride, padding, dilation, dst->nb[0], dst->nb[1]
                    );
                }
            }
            continue;
        }

        if (w_storage_type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            direct_conv_transpose_1d_sycl_template<float, float, float>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                prepared_nb0, prepared_nb1, prepared_nb2,
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else if (w_storage_type == GGML_TYPE_F16 && x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
            direct_conv_transpose_1d_sycl_template<::sycl::half, float, float>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                prepared_nb0, prepared_nb1, prepared_nb2,
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else if (w_storage_type == GGML_TYPE_F16 && x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
            direct_conv_transpose_1d_sycl_template<::sycl::half, ::sycl::half, ::sycl::half>(
                q, w_g, x_g, bias_g, bias_type, dst_g,
                C_in_group, L_in, L_out, kW, C_out_group, batch,
                stride, padding, dilation,
                prepared_nb0, prepared_nb1, prepared_nb2,
                x->nb[0], x->nb[1], x->nb[2],
                dst->nb[0], dst->nb[1], dst->nb[2]
            );
        } else {
            std::cerr << "[ops-sycl] Conv Transpose 1D error: unsupported type combination" << std::endl;
            return false;
        }
    }
    return true;
}

bool ggml_sycl_op_conv_transpose_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }
    return ggml_sycl_op_conv_transpose_1d(backend, params.w, params.x, params.bias, node, params.stride, params.padding, params.dilation, params.groups);
}

} // namespace sycl
} // namespace ggml_ops_ext
