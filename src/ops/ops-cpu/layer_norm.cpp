#include "ops/ops.h"
#include "ops/cpu.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

bool ops_cpu_op_layer_norm(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* gamma = node->src[1];
    struct ggml_tensor* beta = node->src[2];
    struct ggml_tensor* dst = node;

    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));

    const void* x_d = x->data;
    const void* gamma_d = gamma->data;
    const void* beta_d = beta->data;
    void* dst_d = dst->data;

    int64_t ne0 = dst->ne[0]; // Columns
    int64_t ne1 = dst->ne[1]; // Rows
    int64_t ne2 = dst->ne[2]; // Batch
    int64_t ne3 = dst->ne[3];

    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    #pragma omp parallel for collapse(3) num_threads(omp_threads)
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                // Stack buffers or dynamic buffers
                float x_buf_stack[4096];
                float* x_val = x_buf_stack;
                std::vector<float> x_buf_dynamic;
                if (ne0 > 4096) {
                    x_buf_dynamic.resize(ne0);
                    x_val = x_buf_dynamic.data();
                }

                // 1. Load x to FP32
                if (x->type == GGML_TYPE_F32) {
                    x_val = (float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                } else if (x->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* px_row = (const ggml_fp16_t*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                    ggml_fp16_to_fp32_row(px_row, x_val, ne0);
                }

                // 2. Load gamma to FP32
                float gamma_buf_stack[4096];
                float* gamma_val = gamma_buf_stack;
                std::vector<float> gamma_buf_dynamic;
                if (ne0 > 4096) {
                    gamma_buf_dynamic.resize(ne0);
                    gamma_val = gamma_buf_dynamic.data();
                }

                if (gamma->type == GGML_TYPE_F32) {
                    if (gamma->nb[0] == sizeof(float)) {
                        gamma_val = (float*)gamma_d;
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            gamma_val[i0] = *(const float*)((const char*)gamma_d + i0*gamma->nb[0]);
                        }
                    }
                } else if (gamma->type == GGML_TYPE_F16) {
                    if (gamma->nb[0] == sizeof(ggml_fp16_t)) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t*)gamma_d, gamma_val, ne0);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            gamma_val[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)((const char*)gamma_d + i0*gamma->nb[0]));
                        }
                    }
                }

                // 3. Load beta to FP32
                float beta_buf_stack[4096];
                float* beta_val = beta_buf_stack;
                std::vector<float> beta_buf_dynamic;
                if (ne0 > 4096) {
                    beta_buf_dynamic.resize(ne0);
                    beta_val = beta_buf_dynamic.data();
                }

                if (beta->type == GGML_TYPE_F32) {
                    if (beta->nb[0] == sizeof(float)) {
                        beta_val = (float*)beta_d;
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            beta_val[i0] = *(const float*)((const char*)beta_d + i0*beta->nb[0]);
                        }
                    }
                } else if (beta->type == GGML_TYPE_F16) {
                    if (beta->nb[0] == sizeof(ggml_fp16_t)) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t*)beta_d, beta_val, ne0);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            beta_val[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)((const char*)beta_d + i0*beta->nb[0]));
                        }
                    }
                }

                // 4. Compute mean and variance (auto-vectorized)
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += x_val[i0];
                }
                float mean = sum / ne0;

                float sum_sq = 0.0f;
                #pragma omp simd reduction(+:sum_sq)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    float diff = x_val[i0] - mean;
                    sum_sq += diff * diff;
                }
                float variance = sum_sq / ne0;
                float inv_std = 1.0f / std::sqrt(variance + eps);

                // 5. Compute result & Write back
                if (dst->type == GGML_TYPE_F32) {
                    float* pdst_row = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        pdst_row[i0] = (x_val[i0] - mean) * inv_std * gamma_val[i0] + beta_val[i0];
                    }
                } else if (dst->type == GGML_TYPE_F16) {
                    float dst_buf_stack[4096];
                    float* dst_val = dst_buf_stack;
                    std::vector<float> dst_buf_dynamic;
                    if (ne0 > 4096) {
                        dst_buf_dynamic.resize(ne0);
                        dst_val = dst_buf_dynamic.data();
                    }

                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_val[i0] = (x_val[i0] - mean) * inv_std * gamma_val[i0] + beta_val[i0];
                    }

                    ggml_fp16_t* pdst_row = (ggml_fp16_t*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    ggml_fp32_to_fp16_row(dst_val, pdst_row, ne0);
                }
            }
        }
    }

    return true;
}

bool ops_cpu_op_ada_ln(ggml_backend_t backend, struct ggml_tensor* node) {
    const int omp_threads = backend_thread_count(backend);

    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* scale = node->src[1];
    struct ggml_tensor* shift = node->src[2];
    struct ggml_tensor* dst = node;

    float eps;
    std::memcpy(&eps, node->op_params, sizeof(float));

    const void* x_d = x->data;
    const void* scale_d = scale->data;
    const void* shift_d = shift->data;
    void* dst_d = dst->data;

    int64_t ne0 = dst->ne[0]; // Columns (channels C)
    int64_t ne1 = dst->ne[1]; // Rows (sequence T)
    int64_t ne2 = dst->ne[2]; // Batch B
    int64_t ne3 = dst->ne[3];

    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    auto get_offset = [](const struct ggml_tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t dst_ne2) -> size_t {
        int64_t s0 = i0;
        int64_t s1 = 0;
        int64_t s2 = 0;
        int64_t s3 = 0;
        if (t->ne[2] > 1) {
            s2 = i2 % t->ne[2];
            s1 = i1 % t->ne[1];
        } else if (t->ne[1] > 1) {
            if (t->ne[1] == dst_ne2) {
                s1 = i2;
            } else {
                s1 = i1 % t->ne[1];
            }
        }
        if (t->ne[3] > 1) {
            s3 = i3 % t->ne[3];
        }
        return s3 * t->nb[3] + s2 * t->nb[2] + s1 * t->nb[1] + s0 * t->nb[0];
    };

    #pragma omp parallel for collapse(3) num_threads(omp_threads)
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                // Stack buffers or dynamic buffers
                float x_buf_stack[4096];
                float* x_val = x_buf_stack;
                std::vector<float> x_buf_dynamic;
                if (ne0 > 4096) {
                    x_buf_dynamic.resize(ne0);
                    x_val = x_buf_dynamic.data();
                }

                // 1. Load x to FP32
                if (x->type == GGML_TYPE_F32) {
                    x_val = (float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                } else if (x->type == GGML_TYPE_F16) {
                    const ggml_fp16_t* px_row = (const ggml_fp16_t*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                    ggml_fp16_to_fp32_row(px_row, x_val, ne0);
                }

                // 2. Load scale to FP32
                float scale_buf_stack[4096];
                float* scale_val = scale_buf_stack;
                std::vector<float> scale_buf_dynamic;
                if (ne0 > 4096) {
                    scale_buf_dynamic.resize(ne0);
                    scale_val = scale_buf_dynamic.data();
                }

                size_t scale_start_offset = get_offset(scale, 0, i1, i2, i3, ne2);
                if (scale->type == GGML_TYPE_F32) {
                    if (scale->ne[0] == ne0 && scale->nb[0] == sizeof(float)) {
                        scale_val = (float*)((const char*)scale_d + scale_start_offset);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            scale_val[i0] = *(const float*)((const char*)scale_d + get_offset(scale, i0, i1, i2, i3, ne2));
                        }
                    }
                } else if (scale->type == GGML_TYPE_F16) {
                    if (scale->ne[0] == ne0 && scale->nb[0] == sizeof(ggml_fp16_t)) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t*)((const char*)scale_d + scale_start_offset), scale_val, ne0);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            scale_val[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)((const char*)scale_d + get_offset(scale, i0, i1, i2, i3, ne2)));
                        }
                    }
                }

                // 3. Load shift to FP32
                float shift_buf_stack[4096];
                float* shift_val = shift_buf_stack;
                std::vector<float> shift_buf_dynamic;
                if (ne0 > 4096) {
                    shift_buf_dynamic.resize(ne0);
                    shift_val = shift_buf_dynamic.data();
                }

                size_t shift_start_offset = get_offset(shift, 0, i1, i2, i3, ne2);
                if (shift->type == GGML_TYPE_F32) {
                    if (shift->ne[0] == ne0 && shift->nb[0] == sizeof(float)) {
                        shift_val = (float*)((const char*)shift_d + shift_start_offset);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            shift_val[i0] = *(const float*)((const char*)shift_d + get_offset(shift, i0, i1, i2, i3, ne2));
                        }
                    }
                } else if (shift->type == GGML_TYPE_F16) {
                    if (shift->ne[0] == ne0 && shift->nb[0] == sizeof(ggml_fp16_t)) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t*)((const char*)shift_d + shift_start_offset), shift_val, ne0);
                    } else {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            shift_val[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)((const char*)shift_d + get_offset(shift, i0, i1, i2, i3, ne2)));
                        }
                    }
                }

                // 4. Compute mean and variance (auto-vectorized)
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += x_val[i0];
                }
                float mean = sum / ne0;

                float sum_sq = 0.0f;
                #pragma omp simd reduction(+:sum_sq)
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    float diff = x_val[i0] - mean;
                    sum_sq += diff * diff;
                }
                float variance = sum_sq / ne0;
                float inv_std = 1.0f / std::sqrt(variance + eps);

                // 5. Compute result & Write back
                if (dst->type == GGML_TYPE_F32) {
                    float* pdst_row = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        pdst_row[i0] = (x_val[i0] - mean) * inv_std * (1.0f + scale_val[i0]) + shift_val[i0];
                    }
                } else if (dst->type == GGML_TYPE_F16) {
                    float dst_buf_stack[4096];
                    float* dst_val = dst_buf_stack;
                    std::vector<float> dst_buf_dynamic;
                    if (ne0 > 4096) {
                        dst_buf_dynamic.resize(ne0);
                        dst_val = dst_buf_dynamic.data();
                    }

                    for (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_val[i0] = (x_val[i0] - mean) * inv_std * (1.0f + scale_val[i0]) + shift_val[i0];
                    }

                    ggml_fp16_t* pdst_row = (ggml_fp16_t*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                    ggml_fp32_to_fp16_row(dst_val, pdst_row, ne0);
                }
            }
        }
    }

    return true;
}

} // namespace cpu
} // namespace ggml_ops_ext
