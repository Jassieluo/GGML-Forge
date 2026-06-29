// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
// ops_matmul_f32 — 完整的 ggml 矩阵乘法，最小改动移植
//
// 源文件：
//   ggml/src/ggml-cpu/vec.cpp          → ggml_vec_dot_f32
//   ggml/src/ggml-cpu/ggml-cpu.c:1155-1443 → 分块 + 线程调度
//   ggml/src/ggml-cpu/simd-mappings.h  → SIMD 宏
//
// 改动说明：
//   GGML_TENSOR_BINARY_OP_LOCALS → 手动维度变量
//   type_traits[].vec_dot         → 直接 ggml_vec_dot_f32
//   threadpool atomics            → std::atomic<int64_t>
//   其他：ggml_barrier→omp barrier, ggml_is_contiguous→true, etc
// ＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

#include "matmul_f32.h"
#include <cstring>
#include <cassert>
#include <atomic>
#include <omp.h>

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

#if defined(__AVX512F__)
#define OPS_F32_STEP 64
#define OPS_F32_EPR  16
#define OPS_F32_ARR  4
#elif defined(__AVX2__) || defined(__AVX__)
#define OPS_F32_STEP 32
#define OPS_F32_EPR  8
#define OPS_F32_ARR  4
#elif defined(__ARM_NEON__)
#define OPS_F32_STEP 16
#define OPS_F32_EPR  4
#define OPS_F32_ARR  4
#else
#define OPS_F32_STEP 1
#define OPS_F32_EPR  1
#define OPS_F32_ARR  1
#endif

// ── ops_vec_dot_f32 ── 完整拷贝自 ggml/src/ggml-cpu/vec.cpp ──
float ops_vec_dot_f32(int n, const float * x, const float * y) {
    float result;
#if defined(__ARM_FEATURE_SVE)
    float sumf = 0.0f;
    const int sve_reg = 256, f32_epr = sve_reg/32, f32_step = 8*f32_epr;
    const int np = (n & ~(f32_step - 1));
    svfloat32_t s1=svdup_n_f32(0),s2=svdup_n_f32(0),s3=svdup_n_f32(0),s4=svdup_n_f32(0);
    svfloat32_t s5=svdup_n_f32(0),s6=svdup_n_f32(0),s7=svdup_n_f32(0),s8=svdup_n_f32(0);
    for(int i=0;i<np;i+=f32_step){
        s1=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i),svld1_f32(svptrue_b32(),y+i),s1);
        s2=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+f32_epr),svld1_f32(svptrue_b32(),y+i+f32_epr),s2);
        s3=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+2*f32_epr),svld1_f32(svptrue_b32(),y+i+2*f32_epr),s3);
        s4=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+3*f32_epr),svld1_f32(svptrue_b32(),y+i+3*f32_epr),s4);
        s5=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+4*f32_epr),svld1_f32(svptrue_b32(),y+i+4*f32_epr),s5);
        s6=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+5*f32_epr),svld1_f32(svptrue_b32(),y+i+5*f32_epr),s6);
        s7=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+6*f32_epr),svld1_f32(svptrue_b32(),y+i+6*f32_epr),s7);
        s8=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i+7*f32_epr),svld1_f32(svptrue_b32(),y+i+7*f32_epr),s8);
    }
    const int np2=(n&~(f32_epr-1));
    for(int i=np;i<np2;i+=f32_epr) s1=svmad_f32_m(svptrue_b32(),svld1_f32(svptrue_b32(),x+i),svld1_f32(svptrue_b32(),y+i),s1);
    if(np2<n){svbool_t pg=svwhilelt_b32(np2,n);s1=svmad_f32_m(pg,svld1_f32(pg,x+np2),svld1_f32(pg,y+np2),s1);}
    s1=svadd_f32_m(svptrue_b32(),s1,s2);s3=svadd_f32_m(svptrue_b32(),s3,s4);
    s5=svadd_f32_m(svptrue_b32(),s5,s6);s7=svadd_f32_m(svptrue_b32(),s7,s8);
    s1=svadd_f32_m(svptrue_b32(),s1,s3);s5=svadd_f32_m(svptrue_b32(),s5,s7);
    s1=svadd_f32_m(svptrue_b32(),s1,s5);sumf=(float)svaddv_f32(svptrue_b32(),s1);result=sumf;
#elif defined(__riscv_v_intrinsic)
    float sumf=0;int vl=__riscv_vsetvlmax_e32m8();
    vfloat32m1_t vs=__riscv_vfmv_v_f_f32m1(0,vl);
    vfloat32m8_t vsum=__riscv_vfmv_v_f_f32m8_tu(vsum,0,vl);
    for(int i=0;i<n;i+=vl){vl=__riscv_vsetvl_e32m8(n-i);
        vfloat32m8_t ax=__riscv_vle32_v_f32m8_tu(ax,&x[i],vl),ay=__riscv_vle32_v_f32m8_tu(ay,&y[i],vl);
        vsum=__riscv_vfmacc_vv_f32m8_tu(vsum,ax,ay,vl);}
    vl=__riscv_vsetvlmax_e32m8();vs=__riscv_vfredusum_vs_f32m8_f32m1(vsum,vs,vl);
    sumf+=__riscv_vfmv_f_s_f32m1_f32(vs);result=sumf;
#elif defined(__AVX512F__)
    float sumf=0;const int np=(n&~(OPS_F32_STEP-1));
    {__m512 s[OPS_F32_ARR]={_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps(),_mm512_setzero_ps()};
    for(int i=0;i<np;i+=OPS_F32_STEP)for(int j=0;j<OPS_F32_ARR;j++)s[j]=_mm512_fmadd_ps(_mm512_loadu_ps(x+i+j*OPS_F32_EPR),_mm512_loadu_ps(y+i+j*OPS_F32_EPR),s[j]);
    int off=OPS_F32_ARR>>1;for(int i=0;i<off;++i)s[i]=_mm512_add_ps(s[i],s[off+i]);
    off>>=1;for(int i=0;i<off;++i)s[i]=_mm512_add_ps(s[i],s[off+i]);off>>=1;for(int i=0;i<off;++i)s[i]=_mm512_add_ps(s[i],s[off+i]);
    sumf=(float)_mm512_reduce_add_ps(s[0]);}
    for(int i=np;i<n;++i)sumf+=x[i]*y[i];result=sumf;
#elif defined(__AVX2__)
    float sumf=0;const int np=(n&~(OPS_F32_STEP-1));
    {__m256 s[OPS_F32_ARR]={_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps(),_mm256_setzero_ps()};
    for(int i=0;i<np;i+=OPS_F32_STEP)for(int j=0;j<OPS_F32_ARR;j++)s[j]=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+j*OPS_F32_EPR),_mm256_loadu_ps(y+i+j*OPS_F32_EPR),s[j]);
    int off=OPS_F32_ARR>>1;for(int i=0;i<off;++i)s[i]=_mm256_add_ps(s[i],s[off+i]);
    off>>=1;for(int i=0;i<off;++i)s[i]=_mm256_add_ps(s[i],s[off+i]);off>>=1;for(int i=0;i<off;++i)s[i]=_mm256_add_ps(s[i],s[off+i]);
    __m128 t0=_mm_add_ps(_mm256_castps256_ps128(s[0]),_mm256_extractf128_ps(s[0],1));
    t0=_mm_hadd_ps(t0,t0);sumf=_mm_cvtss_f32(_mm_hadd_ps(t0,t0));}
    for(int i=np;i<n;++i)sumf+=x[i]*y[i];result=sumf;
#elif defined(__ARM_NEON__)
    float sumf=0;const int np=(n&~(OPS_F32_STEP-1));
    {float32x4_t s[OPS_F32_ARR]={vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0),vdupq_n_f32(0)};
    for(int i=0;i<np;i+=OPS_F32_STEP)for(int j=0;j<OPS_F32_ARR;j++)s[j]=vmlaq_f32(s[j],vld1q_f32(x+i+j*OPS_F32_EPR),vld1q_f32(y+i+j*OPS_F32_EPR));
    float32x4_t t=s[0];for(int i=1;i<OPS_F32_ARR;++i)t=vaddq_f32(t,s[i]);
    float32x2_t hi=vadd_f32(vget_high_f32(t),vget_low_f32(t));sumf=vget_lane_f32(vpadd_f32(hi,hi),0);}
    for(int i=np;i<n;++i)sumf+=x[i]*y[i];result=sumf;
#else
    double sumf=0;for(int i=0;i<n;++i)sumf+=(double)(x[i]*y[i]);result=(float)sumf;
#endif
    return result;
}

// ── one_chunk ── 完整拷贝自 ggml-cpu.c:1155-1243，最小改动 ──
// 改动：GGML_TENSOR_BINARY_OP_LOCALS → 参数；vec_dot → 直接调

static void mul_mat_one_chunk(
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne10, int64_t ne11, int64_t ne12, int64_t ne13,
    int64_t ne0,  int64_t ne1,  int64_t ne2,  int64_t ne3,
    size_t  nb00, size_t  nb01, size_t  nb02, size_t  nb03,
    size_t  nb10, size_t  nb11, size_t  nb12, size_t  nb13,
    size_t  nb0,  size_t  nb1,  size_t  nb2,  size_t  nb3,
    const float * src0_data, const float * src1_data, float * dst_data,
    const float * wdata,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start, const int64_t ir0_end,
    const int64_t ir1_start, const int64_t ir1_end)
{
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) return;

    bool src1_cont = true;
    const void * wd = wdata ? wdata : (const void*)src1_data;
    size_t row_size = sizeof(float) * ne10;
    size_t src1_col_stride = src1_cont ? row_size : nb11;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;
                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0_data + (0 + i02 * nb02 + i03 * nb03);
                const char * src1_col = (const char*)wd +
                    (src1_cont
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                char * dst_col = (char*)dst_data + (i1 * nb1 + i2 * nb2 + i3 * nb3);

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    float dot = ops_vec_dot_f32(
                        (int)ne00,
                        (const float*)(src0_row + ir0 * nb01),
                        (const float*)(src1_col));
                    // 行优先存储：dst_data[ir0][ir1]
                    *(float*)((char*)dst_data + ir0 * nb1 + ir1 * nb0) += dot;
                }
            }
        }
    }
}

// ── ops_matmul_f32 ── 完整拷贝自 ggml-cpu.c:1245-1443，最小改动 ──
// 改动：张量→维度变量；threadpool→atomic；type_traits→直接

void ops_matmul_f32(int64_t mo, int64_t no, int64_t k,
                    const float * A, const float * B, float * C) {
    // ── 维度变量（替代 GGML_TENSOR_BINARY_OP_LOCALS + type_traits）──
    const int64_t ne00 = k,  ne01 = mo, ne02 = 1, ne03 = 1;
    const int64_t ne10 = k,  ne11 = no, ne12 = 1, ne13 = 1;
    const int64_t ne0  = mo, ne1  = no, ne2  = 1, ne3  = 1;
    const size_t  nb00 = sizeof(float);
    const size_t  nb01 = (size_t)k * sizeof(float);
    const size_t  nb02 = (size_t)k * mo * sizeof(float);
    const size_t  nb03 = (size_t)k * mo * sizeof(float);
    const size_t  nb10 = sizeof(float);
    const size_t  nb11 = (size_t)k * sizeof(float);
    const size_t  nb12 = (size_t)k * no * sizeof(float);
    const size_t  nb13 = (size_t)k * no * sizeof(float);
    const size_t  nb0  = sizeof(float);
    const size_t  nb1  = (size_t)no * sizeof(float);
    const size_t  nb2  = (size_t)mo * no * sizeof(float);
    const size_t  nb3  = (size_t)mo * no * sizeof(float);

    const int64_t vec_dot_type   = 0; // GGML_TYPE_F32
    (void)vec_dot_type;
    const int64_t vec_dot_num_rows = 1; // type_traits[F32].nrows

    // ── 原版 GGML_ASSERT ──
    assert(ne0 == ne01); assert(ne1 == ne11);
    assert(ne2 == ne12); assert(ne3 == ne13);
    assert(nb00 == sizeof(float)); assert(nb10 == sizeof(float));
    assert(nb0 == sizeof(float)); assert(nb0 <= nb1); assert(nb1 <= nb2); assert(nb2 <= nb3);

    std::memset(C, 0, (size_t)mo * no * sizeof(float));

    // ── 多线程 ──
    std::atomic<int64_t> g_chunk{0};
    #pragma omp parallel
    {
        int ith = omp_get_thread_num(), nthr = omp_get_num_threads();
        #pragma omp single
        { g_chunk.store(nthr, std::memory_order_relaxed); }
        #pragma omp barrier

        int64_t chunk_size = (ne0 == 1 || ne1 == 1) ? 64 : 16;
        int64_t nchunk0 = (ne0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (ne1 + chunk_size - 1) / chunk_size;
        if (nchunk0 * nchunk1 < nthr * 4) {
            nchunk0 = ne0 > ne1 ? nthr : 1;
            nchunk1 = ne0 > ne1 ? 1 : nthr;
        }
        int64_t dr0 = (ne0 + nchunk0 - 1) / nchunk0;
        int64_t dr1 = (ne1 + nchunk1 - 1) / nchunk1;

        int64_t current_chunk = ith;
        while (current_chunk < nchunk0 * nchunk1) {
            int64_t ith0 = current_chunk % nchunk0;
            int64_t ith1 = current_chunk / nchunk0;
            int64_t ir0_start = dr0 * ith0;
            int64_t ir0_end   = (ir0_start + dr0 < ne0) ? ir0_start + dr0 : ne0;
            int64_t ir1_start = dr1 * ith1;
            int64_t ir1_end   = (ir1_start + dr1 < ne1) ? ir1_start + dr1 : ne1;

            mul_mat_one_chunk(ne00,ne01,ne02,ne03, ne10,ne11,ne12,ne13,
                ne0,ne1,ne2,ne3, nb00,nb01,nb02,nb03, nb10,nb11,nb12,nb13,
                nb0,nb1,nb2,nb3, A, B, C, nullptr,
                1, ir0_start, ir0_end, ir1_start, ir1_end);

            if (nthr >= nchunk0 * nchunk1) break;
            current_chunk = g_chunk.fetch_add(1, std::memory_order_relaxed);
        }
    }
}
