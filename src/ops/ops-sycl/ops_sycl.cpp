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

namespace tts {
namespace ops {
namespace sycl {

bool compute_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    (void)backend; (void)w; (void)x; (void)dst; (void)stride; (void)padding; (void)dilation;
    return false;
}

template <typename WeightT>
class ConvTranspose1DSYCLKernel;

template <typename WeightT>
void run_conv_transpose_1d_gpu(
    ::sycl::queue* q,
    const WeightT* w_d,
    const float* x_d,
    float* dst_d,
    int stride,
    int padding,
    int dilation,
    int64_t L_in, int64_t C_in,
    int64_t kW, int64_t C_out,
    int64_t L_out, int64_t batch,
    int64_t nb_w0, int64_t nb_w1, int64_t nb_w2,
    int64_t nb_x0, int64_t nb_x1, int64_t nb_x2,
    int64_t nb_dst0, int64_t nb_dst1, int64_t nb_dst2
) {
    int64_t output_size = L_out * C_out * batch;
    q->submit([&](::sycl::handler &cgh) {
        cgh.parallel_for<ConvTranspose1DSYCLKernel<WeightT>>(
            ::sycl::range<1>(output_size),
            [=](::sycl::id<1> id) {
                int64_t global_index = id[0];
                int64_t ow = global_index % L_out;
                int64_t tmp = global_index / L_out;
                int64_t c_out = tmp % C_out;
                int64_t b = tmp / C_out;

                float accumulator = 0.0f;

                for (int64_t c_in = 0; c_in < C_in; ++c_in) {
                    for (int64_t kw = 0; kw < kW; ++kw) {
                        int64_t iw_stride = ow + padding - kw * dilation;
                        if (iw_stride >= 0 && iw_stride % stride == 0) {
                            int64_t iw = iw_stride / stride;
                            if (iw < L_in) {
                                const WeightT* ptr_w = (const WeightT*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1 + kw * nb_w0);
                                const float* ptr_x = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1 + iw * nb_x0);
                                accumulator += (float)(*ptr_w) * (*ptr_x);
                            }
                        }
                    }
                }

                float* ptr_dst = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1 + ow * nb_dst0);
                *ptr_dst = accumulator;
            }
        );
    });
}

bool compute_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {

    ggml_backend_sycl_context* sycl_ctx = (ggml_backend_sycl_context*)backend->context;
    if (!sycl_ctx) return false;
    ::sycl::queue* q = sycl_ctx->stream();
    if (!q) return false;

    int64_t L_in = x->ne[0];
    int64_t C_in = x->ne[1];
    int64_t batch = x->ne[2];

    int64_t kW = w->ne[0];
    int64_t C_out = w->ne[1];

    int64_t L_out = dst->ne[0];

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    if (w->type == GGML_TYPE_F32) {
        const float* w_d = (const float*)w->data;
        run_conv_transpose_1d_gpu<float>(
            q, w_d, x_d, dst_d,
            stride, padding, dilation,
            L_in, C_in, kW, C_out, L_out, batch,
            w->nb[0], w->nb[1], w->nb[2],
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else if (w->type == GGML_TYPE_F16) {
        const ::sycl::half* w_d = (const ::sycl::half*)w->data;
        run_conv_transpose_1d_gpu<::sycl::half>(
            q, w_d, x_d, dst_d,
            stride, padding, dilation,
            L_in, C_in, kW, C_out, L_out, batch,
            w->nb[0], w->nb[1], w->nb[2],
            x->nb[0], x->nb[1], x->nb[2],
            dst->nb[0], dst->nb[1], dst->nb[2]
        );
    } else {
        std::cerr << "[ops-sycl] compute_conv_transpose_1d error: unsupported weight type: " << w->type << std::endl;
        return false;
    }

    q->wait();
    return true;
}

template <typename MaskT>
class SoftMaxSYCLKernel;

template <typename MaskT>
void run_non_contiguous_softmax_gpu(
    ::sycl::queue* q,
    const float* src0_d,
    const MaskT* src1_d,
    const float* src2_d,
    float* dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    int64_t nb20, int64_t nb21, int64_t nb22, int64_t nb23,
    int64_t ne12, int64_t ne13,
    float scale,
    float max_bias,
    float m0,
    float m1,
    uint32_t n_head_log2
) {
    int64_t total_rows = ne01 * ne02 * ne03;
    
    try {
        std::cout << "[ops-sycl] inside run_non_contiguous_softmax_gpu, total_rows=" << total_rows << std::endl;
        q->submit([&](::sycl::handler &cgh) {
            std::cout << "[ops-sycl] inside queue submit command group handler" << std::endl;
            cgh.parallel_for<SoftMaxSYCLKernel<MaskT>>(
                ::sycl::range<1>(total_rows),
                [=](::sycl::id<1> id) {
                    int64_t row_idx = id[0];
                    
                    int64_t i01 = row_idx % ne01;
                    int64_t tmp = row_idx / ne01;
                    int64_t i02 = tmp % ne02;
                    int64_t i03 = tmp / ne02;
                    
                    int64_t i11 = i01;
                    int64_t i12 = i02 % ne12;
                    int64_t i13 = i03 % ne13;
                    
                    const float* r_src0 = (const float*)((const char*)src0_d + i01 * nb01 + i02 * nb02 + i03 * nb03);
                    float* r_dst = (float*)((char*)dst_d + i01 * nb11 + i02 * nb12 + i03 * nb13);
                    const MaskT* r_src1 = src1_d ? (const MaskT*)((const char*)src1_d + i11 * nb21 + i12 * nb22 + i13 * nb23) : nullptr;
                    
                    // Use native get_alibi_slope helper from common.hpp
                    float slope = get_alibi_slope(max_bias, i02, n_head_log2, m0, m1);
                    
                    // Pass 1: Find max value
                    float max_val = -INFINITY;
                    for (int64_t col = 0; col < ne00; ++col) {
                        float val = *(const float*)((const char*)r_src0 + col * nb00) * scale;
                        if (r_src1) {
                            float m_val = (float)*(const MaskT*)((const char*)r_src1 + col * nb20);
                            val += slope * m_val;
                        }
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                    
                    if (src2_d) {
                        float sink_val = src2_d[i02];
                        if (sink_val > max_val) {
                            max_val = sink_val;
                        }
                    }
                    
                    // Pass 2: Exp and sum
                    float sum = 0.0f;
                    for (int64_t col = 0; col < ne00; ++col) {
                        float val = *(const float*)((const char*)r_src0 + col * nb00) * scale;
                        if (r_src1) {
                            float m_val = (float)*(const MaskT*)((const char*)r_src1 + col * nb20);
                            val += slope * m_val;
                        }
                        float ex = ::sycl::exp(val - max_val);
                        *(float*)((char*)r_dst + col * nb10) = ex;
                        sum += ex;
                    }
                    
                    if (src2_d) {
                        sum += ::sycl::exp(src2_d[i02] - max_val);
                    }
                    
                    // Pass 3: Normalize
                    float inv_sum = 1.0f / sum;
                    for (int64_t col = 0; col < ne00; ++col) {
                        *(float*)((char*)r_dst + col * nb10) *= inv_sum;
                    }
                }
            );
        });
        std::cout << "[ops-sycl] queue submit finished successfully" << std::endl;
    } catch (const ::sycl::exception& e) {
        std::cerr << "[ops-sycl] SYCL Exception caught in run_non_contiguous_softmax_gpu: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ops-sycl] Std Exception caught in run_non_contiguous_softmax_gpu: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[ops-sycl] Unknown exception caught in run_non_contiguous_softmax_gpu!" << std::endl;
    }
}

bool compute_softmax(
    ggml_backend_t backend,
    struct ggml_tensor* dst
) {
    std::cout << "[ops-sycl] Entered compute_softmax" << std::endl;
    struct ggml_tensor* src0 = dst->src[0];
    struct ggml_tensor* src1 = dst->src[1];
    struct ggml_tensor* src2 = dst->src[2];

    std::cout << "[ops-sycl] Retrieving SYCL context..." << std::endl;
    ggml_backend_sycl_context* sycl_ctx = (ggml_backend_sycl_context*)backend->context;
    if (!sycl_ctx) {
        std::cerr << "[ops-sycl] Error: sycl_ctx is NULL!" << std::endl;
        return false;
    }
    std::cout << "[ops-sycl] Retrieving SYCL queue..." << std::endl;
    ::sycl::queue* q = sycl_ctx->stream();
    if (!q) {
        std::cerr << "[ops-sycl] Error: sycl queue is NULL!" << std::endl;
        return false;
    }
    std::cout << "[ops-sycl] SYCL context and queue successfully retrieved!" << std::endl;

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

    std::cout << "[ops-sycl] Layout: ne00=" << ne00 << ", ne01=" << ne01 << ", ne02=" << ne02 << ", ne03=" << ne03 << std::endl;
    std::cout << "[ops-sycl] Strides: nb00=" << nb00 << ", nb01=" << nb01 << ", nb02=" << nb02 << ", nb03=" << nb03 << std::endl;
    std::cout << "[ops-sycl] scale=" << scale << ", max_bias=" << max_bias << std::endl;

    if (src1) {
        std::cout << "[ops-sycl] Mask src1 exists. Type=" << src1->type << std::endl;
        if (src1->type == GGML_TYPE_F16) {
            std::cout << "[ops-sycl] Launching run_non_contiguous_softmax_gpu with half mask" << std::endl;
            const ::sycl::half* src1_d = (const ::sycl::half*)src1->data;
            run_non_contiguous_softmax_gpu<::sycl::half>(
                q, src0_d, src1_d, src2_d, dst_d,
                ne00, ne01, ne02, ne03,
                nb00, nb01, nb02, nb03,
                nb10, nb11, nb12, nb13,
                nb20, nb21, nb22, nb23,
                ne12, ne13,
                scale, max_bias, m0, m1, n_head_log2
            );
        } else {
            std::cout << "[ops-sycl] Launching run_non_contiguous_softmax_gpu with float mask" << std::endl;
            const float* src1_d = (const float*)src1->data;
            run_non_contiguous_softmax_gpu<float>(
                q, src0_d, src1_d, src2_d, dst_d,
                ne00, ne01, ne02, ne03,
                nb00, nb01, nb02, nb03,
                nb10, nb11, nb12, nb13,
                nb20, nb21, nb22, nb23,
                ne12, ne13,
                scale, max_bias, m0, m1, n_head_log2
            );
        }
    } else {
        std::cout << "[ops-sycl] Launching run_non_contiguous_softmax_gpu without mask" << std::endl;
        run_non_contiguous_softmax_gpu<float>(
            q, src0_d, (const float*)nullptr, src2_d, dst_d,
            ne00, ne01, ne02, ne03,
            nb00, nb01, nb02, nb03,
            nb10, nb11, nb12, nb13,
            0, 0, 0, 0,
            1, 1,
            scale, max_bias, m0, m1, n_head_log2
        );
    }

    std::cout << "[ops-sycl] Kernel submission done. Waiting for stream..." << std::endl;
    q->wait();
    std::cout << "[ops-sycl] Stream wait done successfully!" << std::endl;

    return true;
}

void register_backend() {
    ops_backend_interface iface = {};
    iface.backend_name_prefix = "SYCL";
    iface.compute_conv_1d = compute_conv_1d;
    iface.compute_conv_transpose_1d = compute_conv_transpose_1d;
    iface.compute_softmax = compute_softmax;
    register_ops_backend(iface);
}

} // namespace sycl
} // namespace ops
} // namespace tts
