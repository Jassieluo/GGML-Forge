#include "ops/ops.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

namespace ggml_ops_ext {
namespace cpu {

// Scalar Snake F32 execution
void ggml_vec_ext_snake_f32(const int n, float * y, const float * x, const float alpha) {
    if (std::abs(alpha) < 1e-6f) {
        // Fallback: y = x if alpha is near-zero
        for (int i = 0; i < n; ++i) {
            y[i] = x[i];
        }
        return;
    }
    
    const float inv_alpha = 1.0f / alpha;
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        float sin_val = std::sin(alpha * val);
        y[i] = val + (sin_val * sin_val) * inv_alpha;
    }
}

// Scalar Snake F16 execution optimized with stack-allocated chunked buffers.
// This avoids heap allocation and allows the compiler to vectorize the F16-F32 conversion loops.
void ggml_vec_ext_snake_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float alpha) {
    if (std::abs(alpha) < 1e-6f) {
        for (int i = 0; i < n; ++i) {
            y[i] = x[i];
        }
        return;
    }
    const float inv_alpha = 1.0f / alpha;
    
    alignas(32) float x_buf[1024];
    alignas(32) float y_buf[1024];
    
    for (int i = 0; i < n; i += 1024) {
        int chunk = std::min(1024, n - i);
        
        // Loop 1: F16 -> F32 Conversion (No math: compiler can fully vectorize this via F16C/AVX2)
        for (int j = 0; j < chunk; ++j) {
            x_buf[j] = ggml_fp16_to_fp32(x[i + j]);
        }
        
        // Loop 2: Math computation (std::sin prevents auto-vectorization, but runs entirely in L1 cache)
        for (int j = 0; j < chunk; ++j) {
            float val = x_buf[j];
            float sin_val = std::sin(alpha * val);
            y_buf[j] = val + (sin_val * sin_val) * inv_alpha;
        }
        
        // Loop 3: F32 -> F16 Conversion (No math: compiler can fully vectorize this via F16C/AVX2)
        for (int j = 0; j < chunk; ++j) {
            y[i + j] = ggml_fp32_to_fp16(y_buf[j]);
        }
    }
}

bool ops_cpu_op_snake(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    
    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* dst = node;

    // Extract alpha parameter from op_params
    float* p = (float*)node->op_params;
    float alpha = p[0];

    const float* x_d = (const float*)x->data;
    float* dst_d = (float*)dst->data;

    int64_t nelements = ggml_nelements(dst);

    int64_t ne0 = dst->ne[0];
    int64_t ne1 = dst->ne[1];
    int64_t ne2 = dst->ne[2];
    int64_t ne3 = dst->ne[3];

    size_t nb_x0 = x->nb[0];
    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst0 = dst->nb[0];
    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            #pragma omp parallel for
            for (int64_t i = 0; i < nelements; i += 65536) {
                int64_t chunk = std::min((int64_t)65536, nelements - i);
                ggml_vec_ext_snake_f32((int)chunk, dst_d + i, x_d + i, alpha);
            }
        } else if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
            #pragma omp parallel for collapse(3)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                        float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                        ggml_vec_ext_snake_f32((int)ne0, pdst, px, alpha);
                    }
                }
            }
        } else {
            const float inv_alpha = (std::abs(alpha) < 1e-6f) ? 0.0f : 1.0f / alpha;
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                            float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);
                            float val = *px;
                            if (std::abs(alpha) < 1e-6f) {
                                *pdst = val;
                            } else {
                                float sin_val = std::sin(alpha * val);
                                *pdst = val + (sin_val * sin_val) * inv_alpha;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Contiguous F16 optimized path
        if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(x) && ggml_is_contiguous(dst)) {
            const ggml_fp16_t* px = (const ggml_fp16_t*)x->data;
            ggml_fp16_t* pdst = (ggml_fp16_t*)dst->data;
            #pragma omp parallel for
            for (int64_t i = 0; i < nelements; i += 65536) {
                int64_t chunk = std::min((int64_t)65536, nelements - i);
                ggml_vec_ext_snake_f16((int)chunk, pdst + i, px + i, alpha);
            }
        } else {
            // F16 or mixed type non-contiguous fallback path
            #pragma omp parallel
            {
                std::vector<float> x_buf(ne0);
                std::vector<float> dst_buf(ne0);
                #pragma omp for
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    for (int64_t i2 = 0; i2 < ne2; ++i2) {
                        for (int64_t i1 = 0; i1 < ne1; ++i1) {
                            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                                const void* px = (const char*)x->data + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0;
                                if (x->type == GGML_TYPE_F16) {
                                    x_buf[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)px);
                                } else {
                                    x_buf[i0] = *(const float*)px;
                                }
                            }
                            ggml_vec_ext_snake_f32((int)ne0, dst_buf.data(), x_buf.data(), alpha);
                            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                                void* pdst = (char*)dst->data + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0;
                                if (dst->type == GGML_TYPE_F16) {
                                    *(ggml_fp16_t*)pdst = ggml_fp32_to_fp16(dst_buf[i0]);
                                } else {
                                    *(float*)pdst = dst_buf[i0];
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

// SnakeBeta F32 vector execution
void ggml_vec_ext_snake_beta_f32(const int n, float * y, const float * x, const float alpha, const float beta) {
    if (std::abs(beta) < 1e-6f) {
        for (int i = 0; i < n; ++i) {
            y[i] = x[i];
        }
        return;
    }
    const float inv_beta = 1.0f / beta;
    for (int i = 0; i < n; ++i) {
        float val = x[i];
        float sin_val = std::sin(alpha * val);
        y[i] = val + (sin_val * sin_val) * inv_beta;
    }
}

// SnakeBeta F16 vector execution optimized with chunked buffers
void ggml_vec_ext_snake_beta_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float alpha, const float beta) {
    if (std::abs(beta) < 1e-6f) {
        for (int i = 0; i < n; ++i) {
            y[i] = x[i];
        }
        return;
    }
    const float inv_beta = 1.0f / beta;
    
    alignas(32) float x_buf[1024];
    alignas(32) float y_buf[1024];
    
    for (int i = 0; i < n; i += 1024) {
        int chunk = std::min(1024, n - i);
        
        for (int j = 0; j < chunk; ++j) {
            x_buf[j] = ggml_fp16_to_fp32(x[i + j]);
        }
        
        for (int j = 0; j < chunk; ++j) {
            float val = x_buf[j];
            float sin_val = std::sin(alpha * val);
            y_buf[j] = val + (sin_val * sin_val) * inv_beta;
        }
        
        for (int j = 0; j < chunk; ++j) {
            y[i + j] = ggml_fp32_to_fp16(y_buf[j]);
        }
    }
}

bool ops_cpu_op_snake_beta(ggml_backend_t backend, struct ggml_tensor* node) {
    (void)backend;
    
    struct ggml_tensor* x = node->src[0];
    struct ggml_tensor* alpha_t = node->src[1];
    struct ggml_tensor* beta_t = node->src[2];
    struct ggml_tensor* dst = node;

    int64_t C = x->ne[1];
    std::vector<float> alpha(C, 0.0f);
    std::vector<float> beta(C, 0.0f);

    auto load_vector = [](struct ggml_tensor* t, std::vector<float>& vec) {
        int64_t n = ggml_nelements(t);
        if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t* data = (const ggml_fp16_t*)t->data;
            for (int64_t i = 0; i < n && i < (int64_t)vec.size(); ++i) {
                vec[i] = ggml_fp16_to_fp32(data[i]);
            }
        } else {
            const float* data = (const float*)t->data;
            for (int64_t i = 0; i < n && i < (int64_t)vec.size(); ++i) {
                vec[i] = data[i];
            }
        }
    };

    load_vector(alpha_t, alpha);
    load_vector(beta_t, beta);

    const void* x_d = x->data;
    void* dst_d = dst->data;

    int64_t ne0 = dst->ne[0];
    int64_t ne1 = dst->ne[1];
    int64_t ne2 = dst->ne[2];
    int64_t ne3 = dst->ne[3];

    size_t nb_x0 = x->nb[0];
    size_t nb_x1 = x->nb[1];
    size_t nb_x2 = x->nb[2];
    size_t nb_x3 = x->nb[3];

    size_t nb_dst0 = dst->nb[0];
    size_t nb_dst1 = dst->nb[1];
    size_t nb_dst2 = dst->nb[2];
    size_t nb_dst3 = dst->nb[3];

    if (x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (nb_x0 == sizeof(float) && nb_dst0 == sizeof(float)) {
            #pragma omp parallel for collapse(3)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                        float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                        ggml_vec_ext_snake_beta_f32((int)ne0, pdst, px, alpha[i1], beta[i1]);
                    }
                }
            }
        } else {
            #pragma omp parallel for collapse(3)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        float a = alpha[i1];
                        float b = beta[i1];
                        float inv_b = (std::abs(b) < 1e-6f) ? 0.0f : 1.0f / b;
                        for (int64_t i0 = 0; i0 < ne0; ++i0) {
                            const float* px = (const float*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0);
                            float* pdst = (float*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0);
                            float val = *px;
                            if (std::abs(b) < 1e-6f) {
                                *pdst = val;
                            } else {
                                float sin_val = std::sin(a * val);
                                *pdst = val + (sin_val * sin_val) * inv_b;
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (x->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16 && nb_x0 == sizeof(ggml_fp16_t) && nb_dst0 == sizeof(ggml_fp16_t)) {
            #pragma omp parallel for collapse(3)
            for (int64_t i3 = 0; i3 < ne3; ++i3) {
                for (int64_t i2 = 0; i2 < ne2; ++i2) {
                    for (int64_t i1 = 0; i1 < ne1; ++i1) {
                        const ggml_fp16_t* px = (const ggml_fp16_t*)((const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1);
                        ggml_fp16_t* pdst = (ggml_fp16_t*)((char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1);
                        ggml_vec_ext_snake_beta_f16((int)ne0, pdst, px, alpha[i1], beta[i1]);
                    }
                }
            }
        } else {
            #pragma omp parallel
            {
                std::vector<float> x_buf(ne0);
                std::vector<float> dst_buf(ne0);
                #pragma omp for collapse(3)
                for (int64_t i3 = 0; i3 < ne3; ++i3) {
                    for (int64_t i2 = 0; i2 < ne2; ++i2) {
                        for (int64_t i1 = 0; i1 < ne1; ++i1) {
                            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                                const void* px = (const char*)x_d + i3*nb_x3 + i2*nb_x2 + i1*nb_x1 + i0*nb_x0;
                                if (x->type == GGML_TYPE_F16) {
                                    x_buf[i0] = ggml_fp16_to_fp32(*(const ggml_fp16_t*)px);
                                } else {
                                    x_buf[i0] = *(const float*)px;
                                }
                            }
                            ggml_vec_ext_snake_beta_f32((int)ne0, dst_buf.data(), x_buf.data(), alpha[i1], beta[i1]);
                            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                                void* pdst = (char*)dst_d + i3*nb_dst3 + i2*nb_dst2 + i1*nb_dst1 + i0*nb_dst0;
                                if (dst->type == GGML_TYPE_F16) {
                                    *(ggml_fp16_t*)pdst = ggml_fp32_to_fp16(dst_buf[i0]);
                                } else {
                                    *(float*)pdst = dst_buf[i0];
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
