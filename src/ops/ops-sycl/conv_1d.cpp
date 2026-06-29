#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <cstdio>
#include <cstring>
#include <algorithm>

namespace ggml_ops_ext {
namespace sycl {

// Kernel names for SYCL compilation
template <typename SrcT, typename DstT>
class CastKernel;

template <typename T_in, typename T_out>
class Im2ColKernel;

template <typename T_a, typename T_b, typename T_c>
class CustomGEMMKernel;

template <typename SrcT, typename DstT>
static void cast_tensor_sycl(::sycl::queue* q, const void* src, void* dst, int64_t n) {
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<CastKernel<SrcT, DstT>>(::sycl::range<1>(n), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
            ((DstT*)dst)[idx] = (DstT)((const SrcT*)src)[idx];
        });
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
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<KernelName>(::sycl::range<1>(total), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
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
        });
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
    int64_t total = M * N;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<CustomGEMMKernel<T_a, T_b, T_c>>(::sycl::range<1>(total), [=](::sycl::id<1> id) {
            int64_t idx = id[0];
            int64_t row = idx % M;
            int64_t col = idx / M;

            float sum = 0.0f;
            for (int64_t k = 0; k < K_dim; ++k) {
                sum += (float)a[row * K_dim + k] * (float)b[col * K_dim + k];
            }
            c[col * ldc + row] = (T_c)sum;
        });
    });
}

template <typename T>
struct sycl_device_alloc {
    ::sycl::queue* q;
    T* ptr;
    size_t size;

    sycl_device_alloc(::sycl::queue* q) : q(q), ptr(nullptr), size(0) {}
    ~sycl_device_alloc() {
        if (ptr) {
            ::sycl::free(ptr, *q);
        }
    }
    void alloc(size_t n) {
        if (ptr) {
            ::sycl::free(ptr, *q);
        }
        size = n;
        ptr = (T*)::sycl::malloc_device(n * sizeof(T), *q);
    }
    T* get() { return ptr; }
};

bool ggml_sycl_op_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;

    int64_t N = x->ne[2];
    int64_t C = x->ne[1];
    int64_t W = x->ne[0];
    int64_t K = w->ne[2];
    int64_t kW = w->ne[0];
    int64_t OW = dst->ne[0];

    const void* w_d = w->data;
    const void* x_d = x->data;
    void* dst_d = dst->data;

    size_t dst_elem_size = (dst->type == GGML_TYPE_F16) ? sizeof(::sycl::half) : sizeof(float);

    // Cast weights if needed (e.g. w is F16 but x/dst are F32)
    const void* w_d_actual = w_d;
    sycl_device_alloc<float> w_f32_alloc(q);
    sycl_device_alloc<::sycl::half> w_f16_alloc(q);

    if (w->type != x->type) {
        int64_t w_len = ggml_nelements(w);
        if (x->type == GGML_TYPE_F32) {
            w_f32_alloc.alloc(w_len);
            cast_tensor_sycl<::sycl::half, float>(q, w_d, w_f32_alloc.get(), w_len);
            w_d_actual = w_f32_alloc.get();
        } else if (x->type == GGML_TYPE_F16) {
            w_f16_alloc.alloc(w_len);
            cast_tensor_sycl<float, ::sycl::half>(q, w_d, w_f16_alloc.get(), w_len);
            w_d_actual = w_f16_alloc.get();
        }
        q->wait();
    }

    const int64_t CHUNK_SIZE = 2048;
    int64_t cur_chunk_size = std::min(CHUNK_SIZE, OW);
    size_t workspace_size = N * C * kW * cur_chunk_size;

    sycl_device_alloc<float> data_col_f32(q);
    sycl_device_alloc<::sycl::half> data_col_f16(q);
    void* data_col = nullptr;

    if (x->type == GGML_TYPE_F16) {
        data_col_f16.alloc(workspace_size);
        data_col = data_col_f16.get();
    } else {
        data_col_f32.alloc(workspace_size);
        data_col = data_col_f32.get();
    }

    for (int64_t ow_start = 0; ow_start < OW; ow_start += CHUNK_SIZE) {
        cur_chunk_size = std::min(CHUNK_SIZE, OW - ow_start);

        if (x->type == GGML_TYPE_F16) {
            launch_im2col_1d_sycl<::sycl::half, ::sycl::half, Im2ColKernel<::sycl::half, ::sycl::half>>(
                q, (const ::sycl::half*)x_d, (::sycl::half*)data_col,
                C, W, OW, kW, stride, padding, dilation,
                N, x->nb[0], x->nb[1], x->nb[2],
                ow_start, cur_chunk_size
            );
        } else {
            launch_im2col_1d_sycl<float, float, Im2ColKernel<float, float>>(
                q, (const float*)x_d, (float*)data_col,
                C, W, OW, kW, stride, padding, dilation,
                N, x->nb[0], x->nb[1], x->nb[2],
                ow_start, cur_chunk_size
            );
        }

        q->wait();

        for (int64_t n = 0; n < N; ++n) {
            if (x->type == GGML_TYPE_F16) {
                const ::sycl::half* cur_data_col = (const ::sycl::half*)data_col + n * (cur_chunk_size * C * kW);
                launch_custom_gemm_sycl<::sycl::half, ::sycl::half, ::sycl::half>(
                    q,
                    cur_chunk_size, K, C * kW,
                    cur_data_col,
                    (const ::sycl::half*)w_d_actual,
                    (::sycl::half*)((char*)dst_d + n * (K * OW * dst_elem_size) + ow_start * dst_elem_size), OW
                );
            } else {
                const float* cur_data_col = (const float*)data_col + n * (cur_chunk_size * C * kW);
                launch_custom_gemm_sycl<float, float, float>(
                    q,
                    cur_chunk_size, K, C * kW,
                    cur_data_col,
                    (const float*)w_d_actual,
                    (float*)((char*)dst_d + n * (K * OW * dst_elem_size) + ow_start * dst_elem_size), OW
                );
            }
        }
    }

    q->wait();
    return true;
}

bool ggml_sycl_op_conv_1d_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    if (!node) return false;
    int32_t params[3];
    std::memcpy(params, node->op_params, sizeof(params));
    int stride = params[0];
    int padding = params[1];
    int dilation = params[2];
    return ggml_sycl_op_conv_1d(backend, node->src[0], node->src[1], node, stride, padding, dilation);
}

} // namespace sycl
} // namespace ggml_ops_ext
