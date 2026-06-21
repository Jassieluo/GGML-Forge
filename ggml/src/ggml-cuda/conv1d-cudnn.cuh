#pragma once

#include "common.cuh"

void ggml_cuda_op_conv_1d_cudnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
