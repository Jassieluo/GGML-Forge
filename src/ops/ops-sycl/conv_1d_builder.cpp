#include "ops/ops.h"
#include "ggml-impl.h"
#include "ggml.h"

namespace ggml_ops_ext {
namespace sycl {

static struct ggml_tensor* force_w_f32(struct ggml_context* ctx, struct ggml_tensor* w) {
    if (!w) return nullptr;
    if (w->type == GGML_TYPE_F32) return w;
    struct ggml_tensor* casted = ggml_cast(ctx, w, GGML_TYPE_F32);
    return ggml_cont(ctx, casted);
}

struct ggml_tensor* sycl_conv_1d_builder(
    struct ggml_context* ctx,
    int op_id,
    struct ggml_tensor** srcs,
    int n_srcs,
    const int32_t* params,
    int n_params,
    ggml_backend_t backend
) {
    (void)op_id;
    (void)n_srcs;
    (void)n_params;
    (void)backend;

    struct ggml_tensor* w = srcs[0];
    struct ggml_tensor* x = srcs[1];
    int stride = params[0];
    int padding = params[1];
    int dilation = params[2];

    struct ggml_tensor* w_f32 = force_w_f32(ctx, w);
    
    struct ggml_tensor* x_reshaped = x;
    if (ggml_n_dims(x) == 1) {
        x_reshaped = ggml_reshape_2d(ctx, x, x->ne[0], 1);
    }
    
    struct ggml_tensor* x_t = ggml_cont(ctx, ggml_transpose(ctx, x_reshaped));
    
    int64_t kernel_size = w_f32->ne[0];
    int64_t in_channels = w_f32->ne[1];
    int64_t out_channels = w_f32->ne[2];
    int64_t seq_len = x_t->ne[1];
    
    int64_t out_seq_len = (seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t max_padded_idx = (out_seq_len - 1) * stride + (kernel_size - 1) * dilation;
    int64_t req_padded_len = max_padded_idx + 1;
    
    int64_t left_pad = padding;
    int64_t right_pad = req_padded_len - seq_len - left_pad;
    if (right_pad < 0) right_pad = 0;
    
    struct ggml_tensor* x_pad = ggml_cont(ctx, ggml_pad_ext(ctx, x_t, 0, 0, left_pad, right_pad, 0, 0, 0, 0));
    
    struct ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_f32, 2, 0, 1, 3));
    
    struct ggml_tensor* sum = nullptr;
    for (int k = 0; k < kernel_size; ++k) {
        struct ggml_tensor* x_k_view = ggml_view_2d(ctx, x_pad, in_channels, out_seq_len, stride * x_pad->nb[1], k * dilation * x_pad->nb[1]);
        struct ggml_tensor* x_k = ggml_cont(ctx, x_k_view);
        
        struct ggml_tensor* w_k_view = ggml_view_2d(ctx, w_perm, in_channels, out_channels, w_perm->nb[1], k * w_perm->nb[2]);
        struct ggml_tensor* w_k = ggml_cont(ctx, w_k_view);
        
        struct ggml_tensor* prod = ggml_mul_mat(ctx, x_k, w_k);
        
        if (sum == nullptr) {
            sum = prod;
        } else {
            sum = ggml_add(ctx, sum, prod);
        }
    }
    
    return ggml_cont(ctx, sum);
}

} // namespace sycl
} // namespace ggml_ops_ext
