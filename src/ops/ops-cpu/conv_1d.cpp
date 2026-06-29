#include "ops/ops.h"
#include "ggml.h"
#include "matmul_f32.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cmath>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_conv_1d(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;

    ops_conv_1d_params params;
    if (!ops_extract_conv_1d_params(node, params)) return false;

    const struct ggml_tensor * w   = params.w;
    const struct ggml_tensor * x   = params.x;
    struct ggml_tensor *       dst = node;

    int stride   = params.stride;
    int padding  = params.padding;
    int dilation = params.dilation;

    const int64_t kW    = w->ne[0];
    const int64_t C_in  = w->ne[1];
    const int64_t C_out = w->ne[2];
    const int64_t L_in  = x->ne[0];
    const int64_t batch = (x->ne[2] > 0) ? x->ne[2] : 1;

    const int64_t L_out = (L_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    GGML_ASSERT(L_out == dst->ne[0]);
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * x_d   = (const float *)x->data;
    float *       dst_d = (float *)dst->data;

    // Pre-convert F16 weight → F32
    std::vector<float> w_f32;
    const float * w_d = nullptr;
    if (w->type == GGML_TYPE_F32) {
        w_d = (const float *)w->data;
    } else if (w->type == GGML_TYPE_F16) {
        int64_t n = ggml_nelements(w);
        w_f32.resize(n);
        const ggml_fp16_t * w16 = (const ggml_fp16_t *)w->data;
        for (int64_t i = 0; i < n; ++i) w_f32[i] = ggml_fp16_to_fp32(w16[i]);
        w_d = w_f32.data();
    } else return false;

    // Transpose weights to [C_out, kW, C_in] row-major layout
    std::vector<float> w_transposed(C_out * kW * C_in);
    for (int64_t oc = 0; oc < C_out; ++oc) {
        for (int64_t k = 0; k < kW; ++k) {
            for (int64_t ic = 0; ic < C_in; ++ic) {
                w_transposed[oc * (kW * C_in) + k * C_in + ic] = w_d[oc * (C_in * kW) + ic * kW + k];
            }
        }
    }

    std::vector<float> x_transposed(L_in * C_in);
    std::vector<float> out_row_major(L_out * C_out);

    for (int64_t b = 0; b < batch; ++b) {
        const float * x_b = x_d + b * (C_in * L_in);
        float * dst_b = dst_d + b * (C_out * L_out);

        // Transpose input to [L_in, C_in] row-major layout
        #pragma omp parallel for collapse(2)
        for (int64_t iw = 0; iw < L_in; ++iw) {
            for (int64_t ic = 0; ic < C_in; ++ic) {
                x_transposed[iw * C_in + ic] = x_b[ic * L_in + iw];
            }
        }

        // Direct Vectorized Convolution loop
        #pragma omp parallel for collapse(2)
        for (int64_t oc = 0; oc < C_out; ++oc) {
            for (int64_t ow = 0; ow < L_out; ++ow) {
                float sum = 0.0f;
                for (int64_t k = 0; k < kW; ++k) {
                    int64_t iw = ow * stride - padding + k * dilation;
                    if (iw >= 0 && iw < L_in) {
                        const float * vec_x = x_transposed.data() + iw * C_in;
                        const float * vec_w = w_transposed.data() + oc * (kW * C_in) + k * C_in;
                        sum += ops_vec_dot_f32((int)C_in, vec_x, vec_w);
                    }
                }
                out_row_major[ow * C_out + oc] = sum;
            }
        }

        // Transpose row-major output to column-major dst_b
        #pragma omp parallel for collapse(2)
        for (int64_t oc = 0; oc < C_out; ++oc) {
            for (int64_t ow = 0; ow < L_out; ++ow) {
                dst_b[oc * L_out + ow] = out_row_major[ow * C_out + oc];
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
