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
    float * C);

float ops_vec_dot_f32(int n, const float * x, const float * y);
