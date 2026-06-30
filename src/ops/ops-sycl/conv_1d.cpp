#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <mutex>

namespace ggml_ops_ext {
namespace sycl {

// Thread-safe cache for transposed weight buffers to avoid copying/transposing on every step
struct CachedWeight {
    void* ptr = nullptr;
    size_t size = 0;
    float val0 = 0.0f;
    float val1 = 0.0f;
};

static std::unordered_map<const void*, CachedWeight> g_weight_cache;
static std::mutex g_cache_mutex;

static void* get_cached_transposed_weight(::sycl::queue* q, const void* orig_ptr, size_t num_elements, size_t elem_size, bool& is_new) {
    float vals[2] = { 0.0f, 0.0f };
    if (num_elements > 0) {
        if (elem_size == sizeof(float)) {
            float host_vals[2] = { 0.0f, 0.0f };
            q->memcpy(&host_vals[0], (const float*)orig_ptr, sizeof(float)).wait();
            q->memcpy(&host_vals[1], (const float*)orig_ptr + num_elements / 2, sizeof(float)).wait();
            vals[0] = host_vals[0];
            vals[1] = host_vals[1];
        } else if (elem_size == sizeof(::sycl::half)) {
            ::sycl::half host_vals[2];
            q->memcpy(&host_vals[0], (const ::sycl::half*)orig_ptr, sizeof(::sycl::half)).wait();
            q->memcpy(&host_vals[1], (const ::sycl::half*)orig_ptr + num_elements / 2, sizeof(::sycl::half)).wait();
            vals[0] = (float)host_vals[0];
            vals[1] = (float)host_vals[1];
        }
    }

    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_weight_cache.find(orig_ptr);
    if (it != g_weight_cache.end()) {
        if (it->second.size == num_elements * elem_size &&
            it->second.val0 == vals[0] &&
            it->second.val1 == vals[1]) {
            is_new = false;
            return it->second.ptr;
        }
        // Pointer was reused with different size or values; free old cached buffer
        ::sycl::free(it->second.ptr, *q);
        g_weight_cache.erase(it);
    }
    is_new = true;
    void* dev_ptr = ::sycl::malloc_device(num_elements * elem_size, *q);
    g_weight_cache[orig_ptr] = { dev_ptr, num_elements * elem_size, vals[0], vals[1] };
    return dev_ptr;
}

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

template <typename SrcT, typename DstT>
static void transpose_weights_sycl(
    ::sycl::queue* q,
    const SrcT* src,
    DstT* dst,
    int64_t K_dim, int64_t N_ch
) {
    int64_t total = K_dim * N_ch;
    int64_t local_size = 256;
    int64_t global_size = ((total + local_size - 1) / local_size) * local_size;

    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<TransposeWeightsKernel<SrcT, DstT>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                int64_t idx = item.get_global_id(0);
                if (idx >= total) return;

                int64_t k = idx % K_dim;
                int64_t col = idx / K_dim;

                dst[k * N_ch + col] = (DstT)src[col * K_dim + k];
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

    // Retrieve cached weight buffer or populate it
    int64_t w_len = ggml_nelements(w);
    bool is_new = false;
    const void* w_d_actual = nullptr;

    if (x->type == GGML_TYPE_F32) {
        float* cached_w = (float*)get_cached_transposed_weight(q, w_d, w_len, sizeof(float), is_new);
        if (is_new) {
            sycl_device_alloc<float> w_f32_alloc(q);
            if (w->type == GGML_TYPE_F16) {
                w_f32_alloc.alloc(w_len);
                cast_tensor_sycl<::sycl::half, float>(q, w_d, w_f32_alloc.get(), w_len);
                q->wait();
                transpose_weights_sycl<float, float>(q, w_f32_alloc.get(), cached_w, C * kW, K);
            } else {
                transpose_weights_sycl<float, float>(q, (const float*)w_d, cached_w, C * kW, K);
            }
            q->wait();
        }
        w_d_actual = cached_w;
    } else {
        ::sycl::half* cached_w = (::sycl::half*)get_cached_transposed_weight(q, w_d, w_len, sizeof(::sycl::half), is_new);
        if (is_new) {
            sycl_device_alloc<::sycl::half> w_f16_alloc(q);
            if (w->type == GGML_TYPE_F32) {
                w_f16_alloc.alloc(w_len);
                cast_tensor_sycl<float, ::sycl::half>(q, w_d, w_f16_alloc.get(), w_len);
                q->wait();
                transpose_weights_sycl<::sycl::half, ::sycl::half>(q, w_f16_alloc.get(), cached_w, C * kW, K);
            } else {
                transpose_weights_sycl<::sycl::half, ::sycl::half>(q, (const ::sycl::half*)w_d, cached_w, C * kW, K);
            }
            q->wait();
        }
        w_d_actual = cached_w;
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
                    (::sycl::half*)((char*)dst_d + n * (OW * dst_elem_size) + ow_start * dst_elem_size), N * OW
                );
            } else {
                const float* cur_data_col = (const float*)data_col + n * (cur_chunk_size * C * kW);
                launch_custom_gemm_sycl<float, float, float>(
                    q,
                    cur_chunk_size, K, C * kW,
                    cur_data_col,
                    (const float*)w_d_actual,
                    (float*)((char*)dst_d + n * (OW * dst_elem_size) + ow_start * dst_elem_size), N * OW
                );
            }
        }
        q->wait();
    }

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
