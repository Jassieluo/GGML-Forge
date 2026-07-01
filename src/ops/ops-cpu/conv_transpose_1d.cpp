#include "ops/ops.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include <cstring>
#include <cstdio>
#include <vector>

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
    const int C_out_group = (int)w->ne[1];
    const int C_in = (int)w->ne[2];
    const int L_in = (int)x->ne[0];
    const int batch = (int)x->ne[2];
    const int groups = params.groups;
    const int C_in_group = C_in / groups;

    const int L_out = (int)dst->ne[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float* dst_d = (float*)dst->data;
    const float* x_d = (const float*)x->data;

    size_t total_elements = ggml_nelements(dst);
    std::memset(dst_d, 0, total_elements * sizeof(float));

    int64_t nb_x0 = x->nb[0];
    int64_t nb_x1 = x->nb[1];
    int64_t nb_x2 = x->nb[2];

    int64_t nb_dst0 = dst->nb[0];
    int64_t nb_dst1 = dst->nb[1];
    int64_t nb_dst2 = dst->nb[2];

    bool standard_strides = (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float));

    // Dequantize/copy weights into a flat float vector: [C_in, C_out_group, kW]
    std::vector<float> w_dequant(C_in * C_out_group * kW);
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int c_out = 0; c_out < C_out_group; ++c_out) {
            for (int kw = 0; kw < kW; ++kw) {
                float val = 0.0f;
                if (w->type == GGML_TYPE_F32) {
                    size_t offset = c_in * w->nb[2] + c_out * w->nb[1] + kw * w->nb[0];
                    val = *(const float *)((const char *)w->data + offset);
                } else if (w->type == GGML_TYPE_F16) {
                    size_t offset = c_in * w->nb[2] + c_out * w->nb[1] + kw * w->nb[0];
                    val = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)w->data + offset));
                } else if (w->type == GGML_TYPE_Q8_0) {
                    const block_q8_0 * blocks = (const block_q8_0 *)w->data;
                    size_t flat_index = c_in * (C_out_group * kW) + c_out * kW + kw;
                    size_t ib = flat_index / 32;
                    size_t is = flat_index % 32;
                    val = ggml_fp16_to_fp32(blocks[ib].d) * blocks[ib].qs[is];
                } else if (w->type == GGML_TYPE_Q4_0) {
                    const block_q4_0 * blocks = (const block_q4_0 *)w->data;
                    size_t flat_index = c_in * (C_out_group * kW) + c_out * kW + kw;
                    size_t ib = flat_index / 32;
                    size_t is = flat_index % 32;
                    uint8_t vi = (blocks[ib].qs[is / 2] >> ((is % 2) * 4)) & 0x0F;
                    val = ggml_fp16_to_fp32(blocks[ib].d) * (vi - 8.0f);
                }
                w_dequant[c_in * (C_out_group * kW) + c_out * kW + kw] = val;
            }
        }
    }

    for (int b = 0; b < batch; ++b) {
        for (int g = 0; g < groups; ++g) {
            #pragma omp parallel for
            for (int c_out_in_group = 0; c_out_in_group < C_out_group; ++c_out_in_group) {
                int c_out = g * C_out_group + c_out_in_group;
                float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                for (int c_in_in_group = 0; c_in_in_group < C_in_group; ++c_in_in_group) {
                    int c_in = g * C_in_group + c_in_in_group;
                    const float* x_row = (const float*)((const char*)x_d + b * nb_x2 + c_in * nb_x1);
                    const float* w_row = w_dequant.data() + c_in * (C_out_group * kW) + c_out_in_group * kW;
                    for (int iw = 0; iw < L_in; ++iw) {
                        float val_x = standard_strides ? x_row[iw] : *(const float*)((const char*)x_row + iw * nb_x0);
                        if (val_x == 0.0f) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            int ow = iw * stride - padding + kw * dilation;
                            if (ow >= 0 && ow < L_out) {
                                float val_w = w_row[kw];
                                if (standard_strides) {
                                    dst_row[ow] += val_w * val_x;
                                } else {
                                    float* ptr_dst = (float*)((char*)dst_row + ow * nb_dst0);
                                    *ptr_dst += val_w * val_x;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
