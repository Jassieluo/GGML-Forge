#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp" // From ggml-sycl
#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace ggml_ops_ext {
namespace sycl {

template <typename T> static __dpct_inline__ float t2f32(T val) {
    return (float) val;
}

template <> float __dpct_inline__ t2f32<::sycl::half>(::sycl::half val) {
  return ::sycl::vec<::sycl::half, 1>(val)
      .convert<float, ::sycl::rounding_mode::automatic>()[0];
}

struct soft_max_params {
    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    
    int64_t nb00;
    int64_t nb01;
    int64_t nb02;
    int64_t nb03;
    
    int64_t nb10;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;
    
    int64_t nb20;
    int64_t nb21;
    int64_t nb22;
    int64_t nb23;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

template <bool use_shared, int ncols_template, int block_size_template, typename T>
static void soft_max_f32_sycl_kernel(
    const float *         x,
    const T *             mask,
    const float *         sinks,
    float *               dst,
    const soft_max_params p,
    uint8_t *             dpct_local
) {
    auto      item_ct1 = ::sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    const int block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    const int nthreads = block_size;
    const int nwarps = nthreads / WARP_SIZE;
    const size_t nreduce = nwarps / WARP_SIZE;

    const int tid = item_ct1.get_local_id(2);

    const int64_t i03 = item_ct1.get_group(0);
    const int64_t i02 = item_ct1.get_group(1);
    const int64_t i01 = item_ct1.get_group(2);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    // Stride-aware row offsets
    const float* r_src0 = (const float*)((const char*)x + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03);
    float* r_dst = (float*)((char*)dst + i01 * p.nb11 + i02 * p.nb12 + i03 * p.nb13);
    const T* r_mask = mask ? (const T*)((const char*)mask + i11 * p.nb21 + i12 * p.nb22 + i13 * p.nb23) : nullptr;

    const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    float * buf_iw = (float *) dpct_local;

    // shared memory buffer to cache values between iterations:
    float *vals = use_shared ? buf_iw + ::sycl::max(nwarps, (int)WARP_SIZE) : dst;
    float max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = *(const float*)((const char*)r_src0 + col * p.nb00) * p.scale + 
                          (r_mask ? slope * t2f32(*(const T*)((const char*)r_mask + col * p.nb20)) : 0.0f);

        if (use_shared) {
            vals[col] = val;
        } else {
            *(float*)((char*)r_dst + col * p.nb10) = val;
        }
        max_val   = ::sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    if (block_size > WARP_SIZE) {
        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        for (int i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = ::sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        float raw_val = use_shared ? vals[col] : *(float*)((char*)r_dst + col * p.nb10);
        // Apply clamping before passing to sycl::native::exp to prevent underflow NaNs on Intel GPU!
        const float val = ::sycl::native::exp(::sycl::max(raw_val - max_val, -80.0f));
        tmp += val;
        
        if (use_shared) {
            vals[col] = val;
        } else {
            *(float*)((char*)r_dst + col * p.nb10) = val;
        }
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    if (block_size > WARP_SIZE) {
        item_ct1.barrier();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            for (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        item_ct1.barrier();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        for (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    if (sinks) {
        tmp += ::sycl::native::exp(::sycl::max(sinks[i02] - max_val, -80.0f));
    }
    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        float final_val = (use_shared ? vals[col] : *(float*)((char*)r_dst + col * p.nb10)) * inv_sum;
        *(float*)((char*)r_dst + col * p.nb10) = final_val;
    }
}

template <int... Ns, typename T>
static void launch_soft_max_kernels(const float *           x,
                                    const T *               mask,
                                    const float *           sinks,
                                    float *                 dst,
                                    const soft_max_params & p,
                                    ::sycl::queue *         stream,
                                    ::sycl::range<3>        block_dims,
                                    ::sycl::range<3>        block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);
        if (p.ncols == ncols) {
            stream->submit([&](::sycl::handler &cgh) {
                ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    ::sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    ::sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](::sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32_sycl_kernel<true, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<::sycl::access::decorated::no>()
                                .get());
                        (void)item_ct1;
                    });
            });
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    stream->submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            ::sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            ::sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](::sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32_sycl_kernel<true, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<::sycl::access::decorated::no>()
                            .get());
                    (void)item_ct1;
                });
    });
}

bool ggml_sycl_op_softmax(
    ggml_backend_t backend,
    struct ggml_tensor* dst
) {
    struct ggml_tensor* src0 = dst->src[0];
    struct ggml_tensor* src1 = dst->src[1];
    struct ggml_tensor* src2 = dst->src[2];

    ::sycl::queue* q = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q) return false;
    if (!q) return false;

    // Read scale and max_bias from op_params
    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, (float*)dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (float*)dst->op_params + 1, sizeof(float));

    // Get shape and stride details
    int64_t ne00 = src0->ne[0];
    int64_t ne01 = src0->ne[1];
    int64_t ne02 = src0->ne[2];
    int64_t ne03 = src0->ne[3];

    int64_t nb00 = src0->nb[0];
    int64_t nb01 = src0->nb[1];
    int64_t nb02 = src0->nb[2];
    int64_t nb03 = src0->nb[3];

    int64_t nb10 = dst->nb[0];
    int64_t nb11 = dst->nb[1];
    int64_t nb12 = dst->nb[2];
    int64_t nb13 = dst->nb[3];

    int64_t nb20 = src1 ? src1->nb[0] : 0;
    int64_t nb21 = src1 ? src1->nb[1] : 0;
    int64_t nb22 = src1 ? src1->nb[2] : 0;
    int64_t nb23 = src1 ? src1->nb[3] : 0;

    int64_t ne12 = src1 ? src1->ne[2] : 1;
    int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t)std::floor(std::log2((double)n_head));

    const float m0 = std::pow(2.0f, -(max_bias) / n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const float* src0_d = (const float*)src0->data;
    float* dst_d = (float*)dst->data;
    const float* src2_d = src2 ? (const float*)src2->data : nullptr;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = ggml_nrows(src0);
    params.nrows_y = src0->ne[1];
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb00 = nb00;
    params.nb01 = nb01;
    params.nb02 = nb02;
    params.nb03 = nb03;
    params.nb10 = nb10;
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.nb20 = nb20;
    params.nb21 = nb21;
    params.nb22 = nb22;
    params.nb23 = nb23;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    int nth = WARP_SIZE;
    ::sycl::device dev = q->get_device();
    int max_block_size = dev.get_info<::sycl::info::device::max_work_group_size>();
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < max_block_size) nth *= 2;
    if (nth > max_block_size) nth = max_block_size;

    const ::sycl::range<3> block_dims(1, 1, nth);
    const ::sycl::range<3> block_nums(params.ne03, params.ne02, params.ne01);
    const size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);

    size_t smpbo = dev.get_info<::sycl::info::device::local_mem_size>();

    if (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        if (src1) {
            if (src1->type == GGML_TYPE_F16) {
                const ::sycl::half* src1_d = (const ::sycl::half*)src1->data;
                launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
                    src0_d, src1_d, src2_d, dst_d, params, q, block_dims, block_nums,
                    nbytes_shared);
            } else {
                const float* src1_d = (const float*)src1->data;
                launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
                    src0_d, src1_d, src2_d, dst_d, params, q, block_dims, block_nums,
                    nbytes_shared);
            }
        } else {
            launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
                src0_d, (const float*)nullptr, src2_d, dst_d, params, q, block_dims, block_nums,
                nbytes_shared);
        }
    } else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);

        q->submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                ::sycl::range<1>(nbytes_shared_low), cgh);

            cgh.parallel_for(
                ::sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](::sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    if (src1) {
                        if (src1->type == GGML_TYPE_F16) {
                            const ::sycl::half* src1_d = (const ::sycl::half*)src1->data;
                            soft_max_f32_sycl_kernel<false, 0, 0>(
                                src0_d, src1_d, src2_d, dst_d, params,
                                dpct_local_acc_ct1
                                    .get_multi_ptr<::sycl::access::decorated::no>()
                                    .get());
                        } else {
                            const float* src1_d = (const float*)src1->data;
                            soft_max_f32_sycl_kernel<false, 0, 0>(
                                src0_d, src1_d, src2_d, dst_d, params,
                                dpct_local_acc_ct1
                                    .get_multi_ptr<::sycl::access::decorated::no>()
                                    .get());
                        }
                    } else {
                        soft_max_f32_sycl_kernel<false, 0, 0>(
                            src0_d, (const float*)nullptr, src2_d, dst_d, params,
                            dpct_local_acc_ct1
                                .get_multi_ptr<::sycl::access::decorated::no>()
                                .get());
                    }
                    (void)item_ct1;
                });
        });
    }

    q->wait();
    return true;
}

} // namespace sycl
} // namespace ggml_ops_ext
