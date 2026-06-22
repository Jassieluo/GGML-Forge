#include "ops/ops.h"
#include "ggml.h"
#include <cstring>
#include <cstdio>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_conv_transpose_1d(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    ops_conv_transpose_1d_params params;
    if (!ops_extract_conv_transpose_1d_params(node, params)) {
        return false;
    }

    struct ggml_tensor* w = params.w;
    struct ggml_tensor* x = params.x;
    struct ggml_tensor* dst = node;

    int stride = params.stride;
    int padding = params.padding;
    int dilation = params.dilation;

    const int kW = (int)w->ne[0];
    const int C_out = (int)w->ne[1];
    const int C_in = (int)w->ne[2];
    const int L_in = (int)x->ne[0];
    const int batch = (int)x->ne[2];

    const int L_out = (int)dst->ne[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float* dst_d = (float*)dst->data;
    const float* x_d = (const float*)x->data;

    size_t total_elements = ggml_nelements(dst);
    std::memset(dst_d, 0, total_elements * sizeof(float));

    int64_t nb_w0 = w->nb[0];
    int64_t nb_w1 = w->nb[1];
    int64_t nb_w2 = w->nb[2];

    int64_t nb_x0 = x->nb[0];
    int64_t nb_x1 = x->nb[1];
    int64_t nb_x2 = x->nb[2];

    int64_t nb_dst0 = dst->nb[0];
    int64_t nb_dst1 = dst->nb[1];
    int64_t nb_dst2 = dst->nb[2];

    bool standard_strides = (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float) && nb_w0 == (w->type == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t)));

    if (w->type == GGML_TYPE_F32) {
        const float* w_d = (const float*)w->data;
        if (standard_strides) {
            for (int b = 0; b < batch; ++b) {
                #pragma omp parallel for
                for (int c_out = 0; c_out < C_out; ++c_out) {
                    float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        const float* x_row = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1);
                        const float* w_row = (const float*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1);
                        for (int iw = 0; iw < L_in; ++iw) {
                            float val_x = x_row[iw];
                            if (val_x == 0.0f) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int ow = iw * stride - padding + kw * dilation;
                                if (ow >= 0 && ow < L_out) {
                                    dst_row[ow] += w_row[kw] * val_x;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Fallback for non-standard strides
            for (int b = 0; b < batch; ++b) {
                #pragma omp parallel for
                for (int c_out = 0; c_out < C_out; ++c_out) {
                    float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        const float* x_row = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1);
                        const float* w_row = (const float*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1);
                        for (int iw = 0; iw < L_in; ++iw) {
                            float val_x = *(const float*)((const char*)x_row + iw * nb_x0);
                            if (val_x == 0.0f) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int ow = iw * stride - padding + kw * dilation;
                                if (ow >= 0 && ow < L_out) {
                                    float val_w = *(const float*)((const char*)w_row + kw * nb_w0);
                                    float* ptr_dst = (float*)((char*)dst_row + ow * nb_dst0);
                                    *ptr_dst += val_w * val_x;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if (w->type == GGML_TYPE_F16) {
        const ggml_fp16_t* w_d = (const ggml_fp16_t*)w->data;
        if (standard_strides) {
            for (int b = 0; b < batch; ++b) {
                #pragma omp parallel for
                for (int c_out = 0; c_out < C_out; ++c_out) {
                    float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        const float* x_row = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1);
                        const ggml_fp16_t* w_row = (const ggml_fp16_t*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1);
                        for (int iw = 0; iw < L_in; ++iw) {
                            float val_x = x_row[iw];
                            if (val_x == 0.0f) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int ow = iw * stride - padding + kw * dilation;
                                if (ow >= 0 && ow < L_out) {
                                    dst_row[ow] += ggml_fp16_to_fp32(w_row[kw]) * val_x;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Fallback for non-standard strides
            for (int b = 0; b < batch; ++b) {
                #pragma omp parallel for
                for (int c_out = 0; c_out < C_out; ++c_out) {
                    float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        const float* x_row = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1);
                        const ggml_fp16_t* w_row = (const ggml_fp16_t*)((const char*)w_d + c_in * nb_w2 + c_out * nb_w1);
                        for (int iw = 0; iw < L_in; ++iw) {
                            float val_x = *(const float*)((const char*)x_row + iw * nb_x0);
                            if (val_x == 0.0f) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                int ow = iw * stride - padding + kw * dilation;
                                if (ow >= 0 && ow < L_out) {
                                    ggml_fp16_t val_w_16 = *(const ggml_fp16_t*)((const char*)w_row + kw * nb_w0);
                                    float val_w = ggml_fp16_to_fp32(val_w_16);
                                    float* ptr_dst = (float*)((char*)dst_row + ow * nb_dst0);
                                    *ptr_dst += val_w * val_x;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        fprintf(stderr, "[ops-cpu] compute_conv_transpose_1d error: unsupported weight type: %d\n", w->type);
        return false;
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
