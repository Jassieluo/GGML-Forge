#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ops/ops.h"
#include <cmath>

namespace nn {

// Base class for all neural network modules
class Module {
public:
    virtual ~Module() = default;
};

// 1. Linear (Dense) layer
class Linear : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [in_features, out_features]
    struct ggml_tensor* bias = nullptr;   // [out_features] (optional)

    Linear() = default;
    Linear(struct ggml_tensor* w, struct ggml_tensor* b = nullptr);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x);
};

// 2. 1D Convolution
class Conv1d : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [kernel_size, in_channels, out_channels]
    struct ggml_tensor* bias = nullptr;   // [out_channels] (optional)
    int stride = 1;
    int padding = 0;
    int dilation = 1;

    Conv1d() = default;
    Conv1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 3. 1D Transposed Convolution
class ConvTranspose1d : public Module {
public:
    struct ggml_tensor* weight = nullptr; // [kernel_size, out_channels, in_channels]
    struct ggml_tensor* bias = nullptr;   // [out_channels] (optional)
    int stride = 1;
    int padding = 0;
    int dilation = 1;

    ConvTranspose1d() = default;
    ConvTranspose1d(struct ggml_tensor* w, struct ggml_tensor* b = nullptr, int stride = 1, int padding = 0, int dilation = 1);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 4. Layer Normalization
class LayerNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    LayerNorm() = default;
    LayerNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 5. Instance Normalization
class InstanceNorm : public Module {
public:
    struct ggml_tensor* gamma = nullptr; // [channels]
    struct ggml_tensor* beta = nullptr;  // [channels]
    float eps = 1e-5f;

    InstanceNorm() = default;
    InstanceNorm(struct ggml_tensor* gamma, struct ggml_tensor* beta, float eps = 1e-5f);

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 6. Gated Linear Unit (GLU)
class GLU : public Module {
public:
    GLU() = default;

    struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend);
};

// 7. Multi-Head Self Attention (MHA)
class MultiHeadAttention : public Module {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear out_proj;
    int n_heads = 1;
    int head_dim = 64;

    MultiHeadAttention() = default;
    MultiHeadAttention(
        struct ggml_tensor* qw, struct ggml_tensor* qb,
        struct ggml_tensor* kw, struct ggml_tensor* kb,
        struct ggml_tensor* vw, struct ggml_tensor* vb,
        struct ggml_tensor* ow, struct ggml_tensor* ob,
        int n_heads, int head_dim
    );

    struct ggml_tensor* forward(
        struct ggml_context* ctx,
        struct ggml_tensor* x,
        struct ggml_tensor* mask,
        ggml_backend_t backend
    );
};

// 8. Activation Functions (wrapped for convenience)
struct Mish {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
        return ggml_ops_mish(ctx, x, backend);
    }
};

struct DoubleSwish {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, ggml_backend_t backend) {
        return ggml_ops_double_swish(ctx, x, backend);
    }
};

struct GatedTanhSigmoid {
    static struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x, int hidden_channels, ggml_backend_t backend) {
        return ggml_ops_gated_tanh_sigmoid(ctx, x, hidden_channels, backend);
    }
};

} // namespace nn
