#include "nn/nn.h"

namespace nn {

namespace functional {

struct ggml_tensor* conv1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend
) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_1d(ctx, w, x_trans, stride, padding, dilation, groups, backend, b);
    if (!conv) return nullptr;
    return ggml_cont(ctx, ggml_transpose(ctx, conv));
}

struct ggml_tensor* conv1d_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int dilation,
    int groups,
    ggml_backend_t backend
) {
    return ggml_ops_conv_1d(ctx, w, x, stride, padding, dilation, groups, backend, b);
}

struct ggml_tensor* conv_transpose1d(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    int groups,
    ggml_backend_t backend
) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_transpose_1d(ctx, w, x_trans, stride, padding, 1, groups, backend, b);
    if (!conv) return nullptr;
    return ggml_cont(ctx, ggml_transpose(ctx, conv));
}

struct ggml_tensor* conv_transpose1d_no_transpose(
    struct ggml_context* ctx,
    struct ggml_tensor* x,
    struct ggml_tensor* w,
    struct ggml_tensor* b,
    int stride,
    int padding,
    ggml_backend_t backend
) {
    return ggml_ops_conv_transpose_1d(ctx, w, x, stride, padding, 1, 1, backend, b);
}

} // namespace functional

struct ggml_tensor* Conv1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t backend_to_use = backend ? backend : this->backend;
    return F::conv1d(ctx, x, weight.tensor(), bias.local_tensor(), stride, padding, dilation, groups, backend_to_use);
}

struct ggml_tensor* ConvTranspose1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    ggml_backend_t backend_to_use = backend ? backend : this->backend;
    return F::conv_transpose1d(ctx, x, weight.tensor(), bias.local_tensor(), stride, padding, groups, backend_to_use);
}

} // namespace nn
