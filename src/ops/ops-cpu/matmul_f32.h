#pragma once
#include <cstdint>

// -----------------------------------------------------------------------
// Public F32 matrix multiplication utility for CPU custom operators.
//
//   C[mo][no] = A[mo][k] × B[k][no]
//
// All matrices are row-major contiguous.  Internally uses the same
// hand-tuned AVX2/AVX512 kernels as ggml's ggml_vec_dot_f32.
// -----------------------------------------------------------------------
void ops_matmul_f32(
    int64_t mo, int64_t no, int64_t k,
    const float * A,
    const float * B,
    float * C,
    int n_threads);

// C[m][n] = A[m][k] x B[k][n], all row-major contiguous.
void ops_matmul_f32_nn(
    int64_t m, int64_t n, int64_t k,
    const float * A,
    const float * B,
    float * C,
    int n_threads);

// C[m][n] = A[m][k] x B[k][n], with a caller-provided output row stride.
void ops_matmul_f32_nn_strided(
    int64_t m, int64_t n, int64_t k,
    const float * A,
    const float * B,
    float * C,
    int64_t ldc,
    int n_threads);

// C[m][n] = A[k][m]^T x B[k][n], all row-major contiguous.
void ops_matmul_f32_tn(
    int64_t m, int64_t n, int64_t k,
    const float * A,
    const float * B,
    float * C,
    int n_threads);

// C = A^T x B + beta * C, row-major contiguous.
void ops_matmul_f32_tn_accumulate(
    int64_t m, int64_t n, int64_t k,
    const float * A,
    const float * B,
    float * C,
    float beta,
    int n_threads);

float ops_vec_dot_f32(int n, const float * x, const float * y);
