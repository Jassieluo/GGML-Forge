#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include "quantized_conv.h"
#include <oneapi/mkl/blas.hpp>
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace ggml_ops_ext {
namespace sycl {

// Kernel names for SYCL compilation
template <typename SrcT, typename DstT>
class CastKernel;

template <typename SrcT, typename DstT>
class TransposeWeightsKernel;

template <typename T_in, typename T_out>
class Im2ColKernel;

template <typename T_a, typename T_b, typename T_c>
class CustomGEMMKernel;

template <typename SrcT, typename DstT>
static void cast_tensor_sycl(::sycl::queue* q, const void* src, void* dst, int64_t n) {
    int64_t local_size = 256;
    int64_t global_size = ((n + local_size - 1) / local_size) * local_size;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<CastKernel<SrcT, DstT>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                int64_t idx = item.get_global_id(0);
                if (idx < n) {
                    ((DstT*)dst)[idx] = (DstT)((const SrcT*)src)[idx];
                }
            }
        );
    });
}

template <typename T>
class AddBiasKernel;

template <typename T>
static void add_bias_sycl(
    ::sycl::queue* q, T* dst, const void* bias, int bias_type, int64_t ne0, int64_t ne1, int64_t ne2,
    size_t nb0, size_t nb1, size_t nb2
) {
    int64_t total = ne0 * ne1 * ne2;
    int64_t local_size = 256;
    int64_t global_size = ((total + local_size - 1) / local_size) * local_size;

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<AddBiasKernel<T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                int64_t idx = item.get_global_id(0);
                if (idx < total) {
                    int64_t i0 = idx % ne0;
                    int64_t tmp = idx / ne0;
                    int64_t i1 = tmp % ne1; // channel
                    int64_t i2 = tmp / ne1; // batch

                    T* pdst = (T*)((char*)dst + i2*nb2 + i1*nb1 + i0*nb0);
                    float b_val = 0.0f;
                    if (bias_type == 0) { // GGML_TYPE_F32
                        b_val = ((const float*)bias)[i1];
                    } else { // GGML_TYPE_F16
                        b_val = (float)((const ::sycl::half*)bias)[i1];
                    }
                    *pdst = (T)((float)*pdst + b_val);
                }
            }
        );
    });
}

template <typename SrcT, typename DstT>
static void transpose_weights_sycl(
    ::sycl::queue* q,
    const SrcT* src,
    DstT* dst,
    int64_t K_dim, int64_t N_ch, int64_t groups
) {
    int64_t total = K_dim * N_ch;
    const int64_t channels_per_group = N_ch / groups;
    int64_t local_size = 256;
    int64_t global_size = ((total + local_size - 1) / local_size) * local_size;

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<TransposeWeightsKernel<SrcT, DstT>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                int64_t idx = item.get_global_id(0);
                if (idx >= total) return;

                const int64_t local_col = idx % channels_per_group;
                const int64_t k = (idx / channels_per_group) % K_dim;
                const int64_t group = idx / (channels_per_group * K_dim);
                const int64_t global_col = group * channels_per_group + local_col;

                dst[idx] = (DstT)src[global_col * K_dim + k];
            }
        );
    });
}

template <typename T_in, typename T_out, typename KernelName>
static void launch_im2col_1d_sycl(
    ::sycl::queue* q,
    const T_in* x, T_out* data_col,
    int64_t C, int64_t W, int64_t OW, int64_t kW,
    int stride, int padding, int dilation,
    int64_t N,
    size_t nb_x0, size_t nb_x1, size_t nb_x2,
    int64_t ow_start, int64_t cur_chunk_size
) {
    int64_t K_dim = C * kW;
    int64_t total = N * cur_chunk_size * K_dim;
    int64_t local_size = 256;
    int64_t global_size = ((total + local_size - 1) / local_size) * local_size;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                int64_t idx = item.get_global_id(0);
                if (idx >= total) return;

                int64_t k = idx % K_dim;
                int64_t tmp = idx / K_dim;
                int64_t ow_offset = tmp % cur_chunk_size;
                int64_t n = tmp / cur_chunk_size;

                int64_t ic = k / kW;
                int64_t ik = k % kW;

                int64_t ow = ow_start + ow_offset;
                int64_t iw = ow * stride - padding + ik * dilation;
                T_out val = 0.0f;
                if (iw >= 0 && iw < W) {
                    const T_in* px = (const T_in*)((const char*)x + n * nb_x2 + ic * nb_x1 + iw * nb_x0);
                    val = (T_out)(float)*px;
                }
                data_col[n * (cur_chunk_size * K_dim) + ow_offset * K_dim + k] = val;
            }
        );
    });
}

template <typename T_a, typename T_b, typename T_c>
static void launch_custom_gemm_sycl(
    ::sycl::queue* q,
    int64_t M, int64_t N, int64_t K_dim,
    const T_a* a,
    const T_b* b,
    T_c* c,
    int64_t ldc
) {
    constexpr int BLOCK_SIZE = 16;

    int64_t global_size_x = ((M + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int64_t global_size_y = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    q->submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<float, 2> tile_a(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
        ::sycl::local_accessor<float, 2> tile_b(::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

        cgh.parallel_for<CustomGEMMKernel<T_a, T_b, T_c>>(
            ::sycl::nd_range<2>(
                ::sycl::range<2>(global_size_x, global_size_y),
                ::sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
            ),
            [=](::sycl::nd_item<2> item) {
                int64_t row = item.get_global_id(0);
                int64_t col = item.get_global_id(1);

                int tx = item.get_local_id(0);
                int ty = item.get_local_id(1);

                float sum = 0.0f;

                for (int64_t t = 0; t < (K_dim + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
                    int64_t a_col = t * BLOCK_SIZE + ty;
                    if (row < M && a_col < K_dim) {
                        tile_a[tx][ty] = (float)a[row * K_dim + a_col];
                    } else {
                        tile_a[tx][ty] = 0.0f;
                    }

                    int64_t b_row = t * BLOCK_SIZE + tx;
                    if (b_row < K_dim && col < N) {
                        tile_b[tx][ty] = (float)b[b_row * N + col];
                    } else {
                        tile_b[tx][ty] = 0.0f;
                    }

                    item.barrier(::sycl::access::fence_space::local_space);

                    #pragma unroll
                    for (int k = 0; k < BLOCK_SIZE; ++k) {
                        sum += tile_a[tx][k] * tile_b[k][ty];
                    }

                    item.barrier(::sycl::access::fence_space::local_space);
                }

                if (row < M && col < N) {
                    c[col * ldc + row] = (T_c)sum;
                }
            }
        );
    });
}

template <typename T>
static void launch_gemm_sycl(
    ::sycl::queue* q,
    int64_t M, int64_t N, int64_t K_dim,
    const T* a,
    const T* b,
    T* c,
    int64_t ldc
) {
    // A and B are row-major. Interpreting them as transposed column-major
    // matrices lets oneMKL write directly into GGML's channel-major output.
    if (M >= 16 && N >= 16 && K_dim >= 16) {
        try {
            oneapi::mkl::blas::column_major::gemm(
                *q,
                oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::trans,
                M, N, K_dim,
                T(1), a, K_dim,
                b, N,
                T(0), c, ldc
            );
            return;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ops-sycl] oneMKL Conv1D GEMM unavailable, using tiled kernel: %s\n", e.what());
        }
    }

    launch_custom_gemm_sycl<T, T, T>(q, M, N, K_dim, a, b, c, ldc);
}

bool ggml_sycl_op_conv_1d(
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

    int64_t N = x->ne[2];
    int64_t C = x->ne[1];
    int64_t W = x->ne[0];
    ops_conv_weight_desc weight_desc = {};
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_1D, w, x, groups, weight_desc)) return false;
    int64_t K = weight_desc.output_channels;
    int64_t kW = weight_desc.kernel;
    int64_t OW = dst->ne[0];

    const void* w_d = w->data;
    const void* x_d = x->data;
    void* dst_d = dst->data;

    size_t dst_elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(::sycl::half) : sizeof(float);

    int64_t C_in_group = C / groups;
    int64_t C_out_group = K / groups;

    if (ggml_is_quantized(w->type)) {
        const int bias_type = bias && bias->type == GGML_TYPE_F16 ? 1 : 0;
#define LAUNCH_DIRECT_QUANT_CONV_SYCL(weight_type, value_type) \
        launch_quantized_conv_1d_direct_sycl<weight_type, value_type>( \
            q, w->data, static_cast<const value_type*>(x->data), bias ? bias->data : nullptr, bias_type, \
            static_cast<value_type*>(dst->data), W, OW, C, K, kW, N, stride, padding, dilation, groups, \
            x->nb[0], x->nb[1], x->nb[2], dst->nb[0], dst->nb[1], dst->nb[2])
        if (x->type == GGML_TYPE_F16) {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q4_0, ::sycl::half); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q4_K, ::sycl::half); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q8_0, ::sycl::half); break;
                default: return false;
            }
        } else {
            switch (w->type) {
                case GGML_TYPE_Q4_0: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q4_0, float); break;
                case GGML_TYPE_Q4_K: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q4_K, float); break;
                case GGML_TYPE_Q8_0: LAUNCH_DIRECT_QUANT_CONV_SYCL(GGML_TYPE_Q8_0, float); break;
                default: return false;
            }
        }
#undef LAUNCH_DIRECT_QUANT_CONV_SYCL
        return true;
    }

    int64_t w_len = ggml_nelements(w);
    const void* w_d_actual = nullptr;
    ops_sycl_pool_alloc<float> transposed_w_f32(backend);
    ops_sycl_pool_alloc<::sycl::half> transposed_w_f16(backend);
    ops_sycl_pool_alloc<float> converted_w_f32(backend);
    ops_sycl_pool_alloc<::sycl::half> converted_w_f16(backend);

    if (x->type == GGML_TYPE_F32) {
        if (!transposed_w_f32.alloc(w_len)) return false;
        if (w->type == GGML_TYPE_F16) {
            if (!converted_w_f32.alloc(w_len)) return false;
            cast_tensor_sycl<::sycl::half, float>(q, w_d, converted_w_f32.get(), w_len);
            transpose_weights_sycl<float, float>(q, converted_w_f32.get(), transposed_w_f32.get(), C_in_group * kW, K, groups);
        } else {
            transpose_weights_sycl<float, float>(q, (const float*)w_d, transposed_w_f32.get(), C_in_group * kW, K, groups);
        }
        w_d_actual = transposed_w_f32.get();
    } else {
        if (!transposed_w_f16.alloc(w_len)) return false;
        if (w->type == GGML_TYPE_F32) {
            if (!converted_w_f16.alloc(w_len)) return false;
            cast_tensor_sycl<float, ::sycl::half>(q, w_d, converted_w_f16.get(), w_len);
            transpose_weights_sycl<::sycl::half, ::sycl::half>(q, converted_w_f16.get(), transposed_w_f16.get(), C_in_group * kW, K, groups);
        } else {
            transpose_weights_sycl<::sycl::half, ::sycl::half>(q, (const ::sycl::half*)w_d, transposed_w_f16.get(), C_in_group * kW, K, groups);
        }
        w_d_actual = transposed_w_f16.get();
    }

    const int64_t CHUNK_SIZE = 2048;
    int64_t cur_chunk_size = std::min(CHUNK_SIZE, OW);
    size_t workspace_size = N * C_in_group * kW * cur_chunk_size;

    ops_sycl_pool_alloc<float> data_col_f32(backend);
    ops_sycl_pool_alloc<::sycl::half> data_col_f16(backend);
    void* data_col = nullptr;

    if (x->type == GGML_TYPE_F16) {
        if (!data_col_f16.alloc(workspace_size)) return false;
        data_col = data_col_f16.get();
    } else {
        if (!data_col_f32.alloc(workspace_size)) return false;
        data_col = data_col_f32.get();
    }

    size_t w_actual_elem_size = (x->type == GGML_TYPE_F16) ? sizeof(::sycl::half) : sizeof(float);
    size_t w_channel_stride_bytes = C_in_group * kW * w_actual_elem_size;

    for (int64_t ow_start = 0; ow_start < OW; ow_start += CHUNK_SIZE) {
        int64_t cur_chunk_size = std::min<int64_t>(CHUNK_SIZE, OW - ow_start);

        for (int g = 0; g < groups; ++g) {
            const void* x_d_g = (const char*)x_d + g * C_in_group * x->nb[1];
            const void* w_d_g = (const char*)w_d_actual + g * C_out_group * w_channel_stride_bytes;
            void* dst_d_g = (char*)dst_d + g * C_out_group * dst->nb[1];

            if (x->type == GGML_TYPE_F16) {
                launch_im2col_1d_sycl<::sycl::half, ::sycl::half, Im2ColKernel<::sycl::half, ::sycl::half>>(
                    q, (const ::sycl::half*)x_d_g, (::sycl::half*)data_col,
                    C_in_group, W, OW, kW, stride, padding, dilation,
                    N, x->nb[0], x->nb[1], x->nb[2],
                    ow_start, cur_chunk_size
                );
            } else {
                launch_im2col_1d_sycl<float, float, Im2ColKernel<float, float>>(
                    q, (const float*)x_d_g, (float*)data_col,
                    C_in_group, W, OW, kW, stride, padding, dilation,
                    N, x->nb[0], x->nb[1], x->nb[2],
                    ow_start, cur_chunk_size
                );
            }

            for (int64_t n = 0; n < N; ++n) {
                if (x->type == GGML_TYPE_F16) {
                    const ::sycl::half* cur_data_col = (const ::sycl::half*)data_col + n * (cur_chunk_size * C_in_group * kW);
                    launch_gemm_sycl<::sycl::half>(
                        q,
                        cur_chunk_size, C_out_group, C_in_group * kW,
                        cur_data_col,
                        (const ::sycl::half*)w_d_g,
                        (::sycl::half*)((char*)dst_d_g + n * dst->nb[2] + ow_start * dst->nb[0]), dst->nb[1] / dst_elem_size
                    );
                } else {
                    const float* cur_data_col = (const float*)data_col + n * (cur_chunk_size * C_in_group * kW);
                    launch_gemm_sycl<float>(
                        q,
                        cur_chunk_size, C_out_group, C_in_group * kW,
                        cur_data_col,
                        (const float*)w_d_g,
                        (float*)((char*)dst_d_g + n * dst->nb[2] + ow_start * dst->nb[0]), dst->nb[1] / dst_elem_size
                    );
                }
            }
        }
    }

    if (bias != nullptr) {
        int bias_type = (bias->type == GGML_TYPE_F32) ? 0 : 1;
        if (dst->type == GGML_TYPE_F32) {
            add_bias_sycl<float>(q, (float*)dst_d, bias->data, bias_type, dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2]);
        } else if (dst->type == GGML_TYPE_F16) {
            add_bias_sycl<::sycl::half>(q, (::sycl::half*)dst_d, bias->data, bias_type, dst->ne[0], dst->ne[1], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2]);
        }
    }

    return true;
}

bool ggml_sycl_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) {
        return false;
    }
    return ggml_sycl_op_conv_1d(backend, params.w, params.x, params.bias, node, params.stride, params.padding, params.dilation, params.groups);
}

} // namespace sycl
} // namespace ggml_ops_ext
