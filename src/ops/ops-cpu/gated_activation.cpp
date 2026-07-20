#include "ops/cpu.h"
#include "ops/ops.h"
#include "ops_cpu_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

namespace ggml_ops_ext::cpu {
namespace {

template <int Activation> float activate(float value) {
  if constexpr (Activation == 0) {
    return value / (1.0f + std::exp(-value));
  }
  if constexpr (Activation == 1) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    return 0.5f * value *
           (1.0f + std::tanh(sqrt_two_over_pi * value *
                             (1.0f + 0.044715f * value * value)));
  }
  if constexpr (Activation == 2) {
    return std::max(value, 0.0f);
  }
  return value;
}

template <int Activation, typename T>
void execute(const T *input, T *output, int64_t inner, int64_t half_axis,
             int64_t outer, int threads) {
  if (inner == 1) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
      const T *gate = input + outer_index * 2 * half_axis;
      const T *linear = gate + half_axis;
      T *output_row = output + outer_index * half_axis;
#pragma omp simd
      for (int64_t axis_index = 0; axis_index < half_axis; ++axis_index) {
        write_val(output_row + axis_index,
                  activate<Activation>(read_val(gate + axis_index)) *
                      read_val(linear + axis_index));
      }
    }
    return;
  }
  const int64_t slices = outer * half_axis;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t slice = 0; slice < slices; ++slice) {
    const int64_t outer_index = slice / half_axis;
    const int64_t axis_index = slice % half_axis;
    const int64_t gate_offset =
        (outer_index * 2 * half_axis + axis_index) * inner;
    const T *gate = input + gate_offset;
    const T *linear = gate + half_axis * inner;
    T *output_slice = output + slice * inner;
#pragma omp simd
    for (int64_t inner_index = 0; inner_index < inner; ++inner_index) {
      write_val(output_slice + inner_index,
                activate<Activation>(read_val(gate + inner_index)) *
                    read_val(linear + inner_index));
    }
  }
}

template <int Activation>
void execute_f16_slice(const ggml_fp16_t *gate, const ggml_fp16_t *linear,
                       ggml_fp16_t *output, int64_t length) {
  int64_t index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
  alignas(32) float gate_f32[8];
  alignas(32) float linear_f32[8];
  alignas(32) float result_f32[8];
  const int64_t vector_length = length & ~int64_t(7);
  for (; index < vector_length; index += 8) {
    _mm256_store_ps(gate_f32,
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(gate + index))));
    _mm256_store_ps(linear_f32,
                    _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(linear + index))));
#pragma omp simd
    for (int lane = 0; lane < 8; ++lane) {
      result_f32[lane] =
          activate<Activation>(gate_f32[lane]) * linear_f32[lane];
    }
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(output + index),
        _mm256_cvtps_ph(_mm256_load_ps(result_f32),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
#endif
  for (; index < length; ++index) {
    output[index] =
        ggml_fp32_to_fp16(activate<Activation>(ggml_fp16_to_fp32(gate[index])) *
                          ggml_fp16_to_fp32(linear[index]));
  }
}

template <int Activation>
void execute_f16(const ggml_fp16_t *input, ggml_fp16_t *output, int64_t inner,
                 int64_t half_axis, int64_t outer, int threads) {
  if (inner == 1) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
      const ggml_fp16_t *gate = input + outer_index * 2 * half_axis;
      execute_f16_slice<Activation>(
          gate, gate + half_axis, output + outer_index * half_axis, half_axis);
    }
    return;
  }
  const int64_t slices = outer * half_axis;
#pragma omp parallel for num_threads(threads) schedule(static)
  for (int64_t slice = 0; slice < slices; ++slice) {
    const int64_t outer_index = slice / half_axis;
    const int64_t axis_index = slice % half_axis;
    const ggml_fp16_t *gate =
        input + (outer_index * 2 * half_axis + axis_index) * inner;
    execute_f16_slice<Activation>(gate, gate + half_axis * inner,
                                  output + slice * inner, inner);
  }
}

template <typename T>
bool dispatch_activation(int activation, const T *input, T *output,
                         int64_t inner, int64_t half_axis, int64_t outer,
                         int threads) {
  switch (activation) {
  case 0:
    execute<0>(input, output, inner, half_axis, outer, threads);
    return true;
  case 1:
    execute<1>(input, output, inner, half_axis, outer, threads);
    return true;
  case 2:
    execute<2>(input, output, inner, half_axis, outer, threads);
    return true;
  case 3:
    execute<3>(input, output, inner, half_axis, outer, threads);
    return true;
  default:
    return false;
  }
}

template <>
bool dispatch_activation(int activation, const ggml_fp16_t *input,
                         ggml_fp16_t *output, int64_t inner, int64_t half_axis,
                         int64_t outer, int threads) {
  switch (activation) {
  case 0:
    execute_f16<0>(input, output, inner, half_axis, outer, threads);
    return true;
  case 1:
    execute_f16<1>(input, output, inner, half_axis, outer, threads);
    return true;
  case 2:
    execute_f16<2>(input, output, inner, half_axis, outer, threads);
    return true;
  case 3:
    execute_f16<3>(input, output, inner, half_axis, outer, threads);
    return true;
  default:
    return false;
  }
}

} // namespace

bool ops_cpu_op_gated_activation(ggml_backend_t backend, ggml_tensor *node) {
  if (!node || !node->src[0])
    return false;
  ops_gated_activation_params params;
  std::memcpy(&params, node->op_params, sizeof(params));
  const ggml_tensor *input = node->src[0];
  int64_t inner = 1;
  for (int axis = 0; axis < params.axis; ++axis)
    inner *= input->ne[axis];
  const int64_t half_axis = input->ne[params.axis] / 2;
  const int64_t outer = ggml_nelements(node) / (inner * half_axis);
  const int threads = backend_thread_count(backend);
  if (node->type == GGML_TYPE_F32) {
    return dispatch_activation(
        params.activation, static_cast<const float *>(input->data),
        static_cast<float *>(node->data), inner, half_axis, outer, threads);
  }
  if (node->type == GGML_TYPE_F16) {
    return dispatch_activation(params.activation,
                               static_cast<const ggml_fp16_t *>(input->data),
                               static_cast<ggml_fp16_t *>(node->data), inner,
                               half_axis, outer, threads);
  }
  return false;
}

} // namespace ggml_ops_ext::cpu
