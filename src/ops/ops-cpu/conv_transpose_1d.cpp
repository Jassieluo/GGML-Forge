#include "ops/ops.h"
#include "ops/cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-common.h"
#include "quantized_block_avx2.h"
#include "matmul_f32.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <immintrin.h>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

static inline void axpy_f32(int n, float scale, const float* weight, float* output) {
    int i = 0;
#if defined(__AVX2__)
    const __m256 scale8 = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        const __m256 w8 = _mm256_loadu_ps(weight + i);
        const __m256 out8 = _mm256_loadu_ps(output + i);
        _mm256_storeu_ps(output + i, _mm256_fmadd_ps(scale8, w8, out8));
    }
#endif
    for (; i < n; ++i) output[i] += scale * weight[i];
}

bool ops_cpu_op_conv_transpose_1d(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);
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

    ops_conv_weight_desc weight_desc = {};
    if (!ops_describe_conv_weight(GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D, w, x, params.groups, weight_desc)) return false;
    const int kW = (int)weight_desc.kernel;
    const int C_out_group = (int)weight_desc.output_channels_per_group;
    const int C_in = (int)x->ne[1];
    const int L_in = (int)x->ne[0];
    const int batch = (int)x->ne[2];
    const int groups = params.groups;
    const int C_in_group = C_in / groups;

    const int L_out = (int)dst->ne[0];

    if (w->type != GGML_TYPE_F32) {
        const bool quantized = ggml_is_quantized(w->type);
        const int block_size = quantized ? (int)ggml_blck_size(w->type) : 1;
        const size_t type_size = quantized ? ggml_type_size(w->type) : sizeof(ggml_fp16_t);
        const size_t row_size = quantized ? ggml_row_size(w->type, C_out_group) : 0;
        if (quantized &&
            w->type != GGML_TYPE_Q4_0 && w->type != GGML_TYPE_Q4_K && w->type != GGML_TYPE_Q8_0) {
            return false;
        }
        if (quantized &&
            (block_size <= 0 || block_size > QK_K || row_size == 0 || C_out_group % block_size != 0)) {
            return false;
        }
        constexpr int output_block = 32;
        const int output_blocks = (C_out_group + output_block - 1) / output_block;
        #pragma omp parallel for collapse(3) num_threads(omp_threads)
        for (int b = 0; b < batch; ++b) {
            for (int group = 0; group < groups; ++group) {
              for (int output_block_index = 0; output_block_index < output_blocks; ++output_block_index) {
                const int output_start = output_block_index * output_block;
                const int output_count = std::min(output_block, C_out_group - output_start);
                std::vector<float> accum((size_t)L_out * output_count);
                for (int ow = 0; ow < L_out; ++ow) {
                    float* output_row = accum.data() + (size_t)ow * output_count;
                    for (int lane = 0; lane < output_count; ++lane) {
                        const int local_oc = output_start + lane;
                        const int oc = group * C_out_group + local_oc;
                        float bias_value = 0.0f;
                        if (params.bias) {
                            const char* bias_ptr = static_cast<const char*>(params.bias->data) + oc * params.bias->nb[0];
                            bias_value = params.bias->type == GGML_TYPE_F16
                                ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(bias_ptr))
                                : *reinterpret_cast<const float*>(bias_ptr);
                        }
                        output_row[lane] = bias_value;
                    }
                }
                alignas(32) float decoded[32];
                quantized_32_microtile quant_tile = {};
                for (int local_ic = 0; local_ic < C_in_group; ++local_ic) {
                    const int ic = group * C_in_group + local_ic;
                    for (int kw = 0; kw < kW; ++kw) {
                        const int row = ic * kW + kw;
                        if (quantized) {
                            const int quant_block = output_start / block_size;
                            const int block_offset = output_start % block_size;
                            const char* block_data = static_cast<const char*>(w->data) + row * row_size +
                                quant_block * type_size;
                            decode_quantized_32_avx2(w->type, block_data, block_offset, quant_tile);
                        } else {
                            for (int lane = 0; lane < output_count; ++lane) {
                                const int local_oc = output_start + lane;
                                const size_t offset = ic * w->nb[2] + local_oc * w->nb[1] + kw * w->nb[0];
                                decoded[lane] = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(
                                    static_cast<const char*>(w->data) + offset));
                            }
                        }
                        for (int iw = 0; iw < L_in; ++iw) {
                            const int ow = iw * stride - padding + kw * dilation;
                            if (ow < 0 || ow >= L_out) continue;
                            const char* input_ptr = static_cast<const char*>(x->data) +
                                b * x->nb[2] + ic * x->nb[1] + iw * x->nb[0];
                            const float input = x->type == GGML_TYPE_F16
                                ? ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(input_ptr))
                                : *reinterpret_cast<const float*>(input_ptr);
                            float* output_row = accum.data() + (size_t)ow * output_count;
                            if (quantized) {
                                axpy_quantized_32_microtile_avx2(w->type, quant_tile, input, output_row);
                            } else {
                                axpy_f32(output_count, input, decoded, output_row);
                            }
                        }
                    }
                }
                for (int ow = 0; ow < L_out; ++ow) {
                    const float* output_row = accum.data() + (size_t)ow * output_count;
                    for (int lane = 0; lane < output_count; ++lane) {
                        const int local_oc = output_start + lane;
                        const int oc = group * C_out_group + local_oc;
                        char* output_ptr = static_cast<char*>(dst->data) +
                            b * dst->nb[2] + oc * dst->nb[1] + ow * dst->nb[0];
                        if (dst->type == GGML_TYPE_F16) {
                            *reinterpret_cast<ggml_fp16_t*>(output_ptr) = ggml_fp32_to_fp16(output_row[lane]);
                        } else {
                            *reinterpret_cast<float*>(output_ptr) = output_row[lane];
                        }
                    }
                }
              }
            }
        }
        return true;
    }

    if (x->type == GGML_TYPE_F16) {
        if (dst->type != GGML_TYPE_F16) return false;
        std::vector<float> x_f32((size_t)L_in * C_in * batch);
        std::vector<float> dst_f32((size_t)L_out * weight_desc.output_channels * batch);
        #pragma omp parallel for collapse(2) num_threads(omp_threads)
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < C_in; ++c) {
                for (int iw = 0; iw < L_in; ++iw) {
                    const char* src = (const char*)x->data + b * x->nb[2] + c * x->nb[1] + iw * x->nb[0];
                    x_f32[iw + (size_t)L_in * (c + C_in * b)] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)src);
                }
            }
        }

        ggml_tensor x_view = *x;
        x_view.type = GGML_TYPE_F32;
        x_view.data = x_f32.data();
        x_view.nb[0] = sizeof(float);
        x_view.nb[1] = L_in * sizeof(float);
        x_view.nb[2] = (size_t)L_in * C_in * sizeof(float);
        x_view.nb[3] = x_view.nb[2] * batch;

        ggml_tensor node_view = *node;
        node_view.type = GGML_TYPE_F32;
        node_view.data = dst_f32.data();
        node_view.nb[0] = sizeof(float);
        node_view.nb[1] = L_out * sizeof(float);
        node_view.nb[2] = (size_t)L_out * weight_desc.output_channels * sizeof(float);
        node_view.nb[3] = node_view.nb[2] * batch;
        node_view.src[1] = &x_view;
        if (!ops_cpu_op_conv_transpose_1d(backend, &node_view)) return false;

        #pragma omp parallel for collapse(2) num_threads(omp_threads)
        for (int b = 0; b < batch; ++b) {
            for (int c = 0; c < weight_desc.output_channels; ++c) {
                for (int ow = 0; ow < L_out; ++ow) {
                    char* out = (char*)dst->data + b * dst->nb[2] + c * dst->nb[1] + ow * dst->nb[0];
                    *(ggml_fp16_t*)out = ggml_fp32_to_fp16(
                        dst_f32[ow + (size_t)L_out * (c + weight_desc.output_channels * b)]);
                }
            }
        }
        return true;
    }

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float* dst_d = (float*)dst->data;
    const float* x_d = (const float*)x->data;

    int64_t nb_dst0 = dst->nb[0];
    int64_t nb_dst1 = dst->nb[1];
    int64_t nb_dst2 = dst->nb[2];

    std::vector<float> bias_vec(C_out_group * groups, 0.0f);
    if (params.bias) {
        const struct ggml_tensor* bias = params.bias;
        for (int oc = 0; oc < C_out_group * groups; ++oc) {
            float val = 0.0f;
            if (bias->type == GGML_TYPE_F32) {
                val = *(const float *)((const char *)bias->data + oc * bias->nb[0]);
            } else if (bias->type == GGML_TYPE_F16) {
                val = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)bias->data + oc * bias->nb[0]));
            }
            bias_vec[oc] = val;
        }
    }

    // Initialize dst with bias
    for (int b = 0; b < batch; ++b) {
        for (int c_out = 0; c_out < C_out_group * groups; ++c_out) {
            float bias_val = bias_vec[c_out];
            float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
            if (nb_dst0 == sizeof(float)) {
                std::fill_n(dst_row, L_out, bias_val);
            } else {
                for (int ow = 0; ow < L_out; ++ow) {
                    float* ptr = (float*)((char*)dst_row + ow * nb_dst0);
                    *ptr = bias_val;
                }
            }
        }
    }

    int64_t nb_x0 = x->nb[0];
    int64_t nb_x1 = x->nb[1];
    int64_t nb_x2 = x->nb[2];

    bool standard_strides = (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float));

    const int gemm_cols = C_out_group * kW;
    const bool use_gemm = C_in_group >= 16 && gemm_cols >= 16 && L_in >= 16;
    std::vector<float> prepared_weights((size_t)C_in * gemm_cols);
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int c_out = 0; c_out < C_out_group; ++c_out) {
            for (int kw = 0; kw < kW; ++kw) {
                const size_t offset = c_in * w->nb[2] + c_out * w->nb[1] + kw * w->nb[0];
                const float val = *reinterpret_cast<const float*>(static_cast<const char*>(w->data) + offset);
                const int m = c_out * kW + kw;
                if (use_gemm) {
                    const int g = c_in / C_in_group;
                    const int ic = c_in % C_in_group;
                    prepared_weights[((size_t)g * gemm_cols + m) * C_in_group + ic] = val;
                } else {
                    prepared_weights[(size_t)c_in * gemm_cols + m] = val;
                }
            }
        }
    }
    const std::vector<float> empty_weights;
    const std::vector<float>& w_gemm = use_gemm ? prepared_weights : empty_weights;
    const std::vector<float>& w_dequant = use_gemm ? empty_weights : prepared_weights;

    if (use_gemm) {
        std::vector<float> x_gemm((size_t)L_in * C_in_group);
        std::vector<float> col((size_t)L_in * gemm_cols);
        for (int b = 0; b < batch; ++b) {
            for (int g = 0; g < groups; ++g) {
                for (int iw = 0; iw < L_in; ++iw) {
                    float* row = x_gemm.data() + (size_t)iw * C_in_group;
                    for (int ic = 0; ic < C_in_group; ++ic) {
                        const int global_ic = g * C_in_group + ic;
                        row[ic] = *(const float*)((const char*)x_d +
                            b * nb_x2 + global_ic * nb_x1 + iw * nb_x0);
                    }
                }

                ops_matmul_f32(
                    L_in, gemm_cols, C_in_group,
                    x_gemm.data(),
                    w_gemm.data() + (size_t)g * gemm_cols * C_in_group,
                    col.data(),
                    omp_threads
                );

                #pragma omp parallel for num_threads(omp_threads)
                for (int c_out_in_group = 0; c_out_in_group < C_out_group; ++c_out_in_group) {
                    const int c_out = g * C_out_group + c_out_in_group;
                    float* dst_row = (float*)((char*)dst_d + b * nb_dst2 + c_out * nb_dst1);
                    for (int ow = 0; ow < L_out; ++ow) {
                        float sum = bias_vec[c_out];
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw_stride = ow + padding - kw * dilation;
                            if (iw_stride >= 0 && iw_stride % stride == 0) {
                                const int iw = iw_stride / stride;
                                if (iw < L_in) {
                                    sum += col[(size_t)iw * gemm_cols + c_out_in_group * kW + kw];
                                }
                            }
                        }
                        if (standard_strides) {
                            dst_row[ow] = sum;
                        } else {
                            *(float*)((char*)dst_row + ow * nb_dst0) = sum;
                        }
                    }
                }
            }
        }
        return true;
    }

    if (batch * groups >= 4) {
        #pragma omp parallel for collapse(2) num_threads(omp_threads)
        for (int b = 0; b < batch; ++b) {
            for (int g = 0; g < groups; ++g) {
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
    } else {
        for (int b = 0; b < batch; ++b) {
            for (int g = 0; g < groups; ++g) {
                #pragma omp parallel for num_threads(omp_threads)
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
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
