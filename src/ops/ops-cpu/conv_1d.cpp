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

// -----------------------------------------------------------------------
// im2col for 1D convolution (F32)
//   x:   [L_in, C_in]  row-major
//   col: [L_out, IK]   each row = C_in*kW contiguous elements
// -----------------------------------------------------------------------
static void im2col_1d_f32(
    const float * x, float * col,
    int64_t C_in, int64_t L_in, int64_t L_out,
    int64_t kW, int stride, int padding, int dilation)
{
    const int64_t IK = C_in * kW;
    for (int64_t ow = 0; ow < L_out; ++ow) {
        float * row = col + ow * IK;
        for (int64_t ic = 0; ic < C_in; ++ic)
            for (int64_t k = 0; k < kW; ++k) {
                int64_t iw = ow * stride - padding + k * dilation;
                row[ic * kW + k] = (iw >= 0 && iw < L_in) ? x[ic * L_in + iw] : 0.0f;
            }
    }
}

// ===================================================================
// ops_cpu_op_conv_1d — enters the operator dispatch hook.
// ===================================================================
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

    const int64_t IK = C_in * kW;
    const bool  is_1x1 = (kW == 1 && stride == 1 && padding == 0 && dilation == 1);

    // Reshape weight [kW, C_in, C_out] → [C_out, IK]
    std::vector<float> w_gemm;
    const float * w_gemm_ptr = w_d;
    if (!is_1x1) {
        w_gemm.resize(C_out * IK);
        for (int64_t oc = 0; oc < C_out; ++oc)
            for (int64_t ic = 0; ic < C_in; ++ic)
                for (int64_t k = 0; k < kW; ++k)
                    w_gemm[oc * IK + ic * kW + k] =
                        w_d[k + ic * kW + oc * C_in * kW];
        w_gemm_ptr = w_gemm.data();
    }

    std::vector<float> col;    // im2col workspace
    std::vector<float> out;    // matmul workspace (only for non-1x1)

    for (int64_t b = 0; b < batch; ++b) {
        const float * x_b = x_d + b * (C_in * L_in);
        float * dst_b = dst_d + b * (C_out * L_out);

        if (is_1x1) {
            // 1×1 shortcut: direct matmul
            // dst[L_out, C_out] = x[L_in, C_in] × w[C_in, C_out]
            ops_matmul_f32(L_out, C_out, C_in, x_b, w_gemm_ptr, dst_b);
        } else {
            col.resize(L_out * IK);
            out.resize(L_out * C_out);

            im2col_1d_f32(x_b, col.data(), C_in, L_in, L_out,
                          kW, stride, padding, dilation);

            // dst[L_out, C_out] = col[L_out, IK] × w_gemm[IK, C_out]^T
            // i.e. out[ow][oc] = sum_{ik} col[ow][ik] * w_gemm[oc][ik]
            ops_matmul_f32(L_out, C_out, IK, col.data(), w_gemm_ptr, out.data());
            std::memcpy(dst_b, out.data(), L_out * C_out * sizeof(float));
        }
    }
    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
