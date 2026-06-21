#include "softmax.hpp"
#include <cstdint>
#include <cmath>


template <typename T> static __dpct_inline__ float t2f32(T val) {
    return (float) val;
}

template <> float __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  return sycl::vec<sycl::half, 1>(val)
      .convert<float, sycl::rounding_mode::automatic>()[0];
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
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

static void soft_max_back_f32(const float *grad, const float *dstf, float *dst,
                              const int ncols, const float scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid      = item_ct1.get_local_id(2);
    const int rowx     = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

static void soft_max_back_f32_sycl(const float *   grad,
                                   const float *   dstf,
                                   float *         dst,
                                   const int       ncols,
                                   const int       nrows,
                                   const float     scale,
                                   dpct::queue_ptr stream) {
    const dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    const dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

template <typename T>
static void soft_max_f32_device(const float *x,
                                const T *mask,
                                const float *sinks,
                                float *dst,
                                const soft_max_params p,
                                float *scratch) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid = item_ct1.get_local_id(2);
    const int nth = item_ct1.get_local_range(2);

    const int64_t rowx = item_ct1.get_group(2) +
                         item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                         item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                             item_ct1.get_group_range(1);

    const int64_t i01 = rowx % p.ne01;
    const int64_t t   = rowx / p.ne01;
    const int64_t i02 = t % p.ne02;
    const int64_t i03 = t / p.ne02;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    const float *x_row = x + rowx * p.ncols;
    const T *mask_row = mask ? mask + (i11 * p.nb11 + i12 * p.nb12 + i13 * p.nb13) / sizeof(T) : nullptr;
    float *dst_row = dst + rowx * p.ncols;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    float local_max = sinks ? sinks[i02] : -INFINITY;
    for (int64_t col = tid; col < p.ncols; col += nth) {
        const float val = x_row[col] * p.scale + (mask_row ? slope * t2f32(mask_row[col]) : 0.0f);
        local_max = sycl::fmax(local_max, val);
    }

    scratch[tid] = local_max;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int offset = nth / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] = sycl::fmax(scratch[tid], scratch[tid + offset]);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    const float row_max = scratch[0];

    float local_sum = 0.0f;
    for (int64_t col = tid; col < p.ncols; col += nth) {
        const float val = x_row[col] * p.scale + (mask_row ? slope * t2f32(mask_row[col]) : 0.0f);
        const float ex = sycl::exp(val - row_max);
        dst_row[col] = ex;
        local_sum += ex;
    }

    if (tid == 0 && sinks) {
        local_sum += sycl::exp(sinks[i02] - row_max);
    }

    scratch[tid] = local_sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int offset = nth / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    const float inv_sum = 1.0f / scratch[0];
    for (int64_t col = tid; col < p.ncols; col += nth) {
        dst_row[col] *= inv_sum;
    }
}

template <typename T>
static void soft_max_f32_sycl(const float *x,
                              const T *mask,
                              const float *sinks,
                              float *dst,
                              const soft_max_params &params,
                              dpct::queue_ptr stream,
                              int device) {
    int nth = WARP_SIZE;
    const int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    while (nth < params.ncols && nth < max_block_size && nth < SYCL_SOFT_MAX_BLOCK_SIZE) {
        nth <<= 1;
    }
    if (nth > max_block_size) {
        nth = max_block_size;
    }
    if ((nth & (nth - 1)) != 0) {
        int pow2 = 1;
        while ((pow2 << 1) <= nth) {
            pow2 <<= 1;
        }
        nth = pow2;
    }

    const dpct::dim3 block_dims(nth, 1, 1);
    const dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t local_bytes = (size_t) nth * sizeof(float);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> scratch_acc(sycl::range<1>(nth), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                soft_max_f32_device(x, mask, sinks, dst, params,
                                    scratch_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
                GGML_UNUSED(item_ct1);
            });
    });
}

#if GGML_SYCL_DNNL
static bool ggml_sycl_op_soft_max_dnnl(ggml_backend_sycl_context & ctx, ggml_tensor * dst, float scale) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    if (src1 != nullptr || src2 != nullptr) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);
    if (ncols <= 0 || nrows <= 0) {
        return false;
    }

    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    dpct::queue_ptr q = ctx.stream();
    auto eng = ctx.engine_dnnl(q);
    auto stream = ctx.stream_dnnl(q);

    ggml_sycl_pool_alloc<float> scaled_src(ctx.pool());
    const float * softmax_src = src0_d;
    if (scale != 1.0f) {
        float * scaled_ptr = scaled_src.alloc((size_t) ggml_nelements(src0));
        softmax_src = scaled_ptr;
        const size_t ne = (size_t) ggml_nelements(src0);
        q->parallel_for(sycl::range<1>(ne), [=](sycl::id<1> id) {
            scaled_ptr[id[0]] = src0_d[id[0]] * scale;
        });
    }

    const auto md = dnnl::memory::desc(
        dnnl::memory::dims { (dnnl_dim_t) nrows, (dnnl_dim_t) ncols },
        dnnl::memory::data_type::f32,
        dnnl::memory::dims { (dnnl_dim_t) ncols, 1 });

    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto pd = dnnl::softmax_forward::primitive_desc(
        eng,
        dnnl::prop_kind::forward_inference,
        dnnl::algorithm::softmax_accurate,
        md,
        md,
        1,
        attr);

    auto src_mem = dnnl::memory(pd.src_desc(), eng, const_cast<float *>(softmax_src));
    auto dst_mem = dnnl::memory(pd.dst_desc(), eng, dst_d);
    auto scratchpad_mem = ctx.get_scratchpad_mem(pd.scratchpad_desc(), eng, q);

    dnnl::softmax_forward(pd).execute(stream, {
        { DNNL_ARG_SRC, src_mem },
        { DNNL_ARG_DST, dst_mem },
        { DNNL_ARG_SCRATCHPAD, scratchpad_mem },
    });
    return true;
}
#endif

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const float * src2_d = src2 ? (const float *) src2->data : nullptr;
    float       * dst_d  = (float *) dst->data;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

#if GGML_SYCL_DNNL
    if (!g_ggml_sycl_disable_dnn && ggml_sycl_op_soft_max_dnnl(ctx, dst, scale)) {
        return;
    }
#endif

    dpct::queue_ptr stream = ctx.stream();
    if (src1 && src1->type == GGML_TYPE_F16) {
        soft_max_f32_sycl(src0_d, (const sycl::half *) src1_d, src2_d, dst_d, params, stream, ctx.device);
    } else {
        soft_max_f32_sycl(src0_d, (const float *) src1_d, src2_d, dst_d, params, stream, ctx.device);
    }
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
