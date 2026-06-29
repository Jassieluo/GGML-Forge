#include "nn/nn.h"

namespace nn {

// Conv1d
Conv1d::Conv1d(struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, int dilation)
    : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation) {}

struct ggml_tensor* Conv1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_1d(ctx, weight, x_trans, stride, padding, dilation, backend);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_transpose(ctx, conv));
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

// ConvTranspose1d
ConvTranspose1d::ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b, int stride, int padding, int dilation)
    : weight(w), bias(b), stride(stride), padding(padding), dilation(dilation) {}

struct ggml_tensor* ConvTranspose1d::forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
    // x shape: [in_channels, seq_len] -> transpose to [seq_len, in_channels]
    struct ggml_tensor* x_trans = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor* conv = ggml_ops_conv_transpose_1d(ctx, weight, x_trans, stride, padding, dilation, backend);
    struct ggml_tensor* out = ggml_cont(ctx, ggml_transpose(ctx, conv));
    if (bias) {
        struct ggml_tensor* b_reshaped = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
        out = ggml_add(ctx, out, b_reshaped);
    }
    return out;
}

} // namespace nn
