#include "ops_cuda_common.cuh"

namespace ggml_ops_ext::cuda {
namespace {

template <int Activation> __device__ float activate(float value) {
  if constexpr (Activation == 0) {
    return value / (1.0f + expf(-value));
  }
  if constexpr (Activation == 1) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    return 0.5f * value *
           (1.0f + tanhf(sqrt_two_over_pi * value *
                         (1.0f + 0.044715f * value * value)));
  }
  if constexpr (Activation == 2) {
    return fmaxf(value, 0.0f);
  }
  return value;
}

template <int Activation, typename T>
__global__ void gated_activation_kernel(const T *input, T *output,
                                        int64_t elements, int64_t inner,
                                        int64_t half_axis) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  const int64_t inner_index = index % inner;
  const int64_t quotient = index / inner;
  const int64_t axis_index = quotient % half_axis;
  const int64_t outer_index = quotient / half_axis;
  const int64_t gate_offset =
      (outer_index * 2 * half_axis + axis_index) * inner + inner_index;
  const int64_t linear_offset = gate_offset + half_axis * inner;
  output[index] = static_cast<T>(
      activate<Activation>(static_cast<float>(input[gate_offset])) *
      static_cast<float>(input[linear_offset]));
}

template <int Activation>
__global__ void gated_activation_f16_axis0_kernel(const half2 *input,
                                                  half2 *output, int64_t pairs,
                                                  int64_t half_axis_pairs) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs)
    return;
  const int64_t axis_index = index % half_axis_pairs;
  const int64_t outer_index = index / half_axis_pairs;
  const int64_t gate_offset = outer_index * 2 * half_axis_pairs + axis_index;
  const float2 gate = __half22float2(input[gate_offset]);
  const float2 linear = __half22float2(input[gate_offset + half_axis_pairs]);
  output[index] = __floats2half2_rn(activate<Activation>(gate.x) * linear.x,
                                    activate<Activation>(gate.y) * linear.y);
}

template <typename T>
bool launch(cudaStream_t stream, int activation, const T *input, T *output,
            int64_t elements, int64_t inner, int64_t half_axis) {
  constexpr int threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((elements + threads - 1) / threads);
#define LAUNCH_GATED(ACTIVATION)                                               \
  gated_activation_kernel<ACTIVATION><<<blocks, threads, 0, stream>>>(         \
      input, output, elements, inner, half_axis)
  switch (activation) {
  case 0:
    LAUNCH_GATED(0);
    break;
  case 1:
    LAUNCH_GATED(1);
    break;
  case 2:
    LAUNCH_GATED(2);
    break;
  case 3:
    LAUNCH_GATED(3);
    break;
  default:
    return false;
  }
#undef LAUNCH_GATED
  return cudaGetLastError() == cudaSuccess;
}

bool launch_f16(cudaStream_t stream, int activation, const half *input,
                half *output, int64_t elements, int64_t inner,
                int64_t half_axis) {
  if (inner != 1 || half_axis % 2 != 0) {
    return launch(stream, activation, input, output, elements, inner,
                  half_axis);
  }
  constexpr int threads = 256;
  const int64_t pairs = elements / 2;
  const int64_t half_axis_pairs = half_axis / 2;
  const unsigned blocks =
      static_cast<unsigned>((pairs + threads - 1) / threads);
#define LAUNCH_GATED_F16(ACTIVATION)                                           \
  gated_activation_f16_axis0_kernel<ACTIVATION>                                \
      <<<blocks, threads, 0, stream>>>(reinterpret_cast<const half2 *>(input), \
                                       reinterpret_cast<half2 *>(output),      \
                                       pairs, half_axis_pairs)
  switch (activation) {
  case 0:
    LAUNCH_GATED_F16(0);
    break;
  case 1:
    LAUNCH_GATED_F16(1);
    break;
  case 2:
    LAUNCH_GATED_F16(2);
    break;
  case 3:
    LAUNCH_GATED_F16(3);
    break;
  default:
    return false;
  }
#undef LAUNCH_GATED_F16
  return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_gated_activation_entry(ggml_backend_t backend,
                                         ggml_tensor *node) {
  if (!node || !node->src[0])
    return false;
  ops_gated_activation_params params;
  std::memcpy(&params, node->op_params, sizeof(params));
  const ggml_tensor *input = node->src[0];
  int64_t inner = 1;
  for (int axis = 0; axis < params.axis; ++axis)
    inner *= input->ne[axis];
  const int64_t half_axis = input->ne[params.axis] / 2;
  const int64_t elements = ggml_nelements(node);
  const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
  cudaStream_t stream =
      static_cast<cudaStream_t>(ggml_ops_ext_bridge_cuda_get_stream(backend));
  CUDA_CHECK(cudaSetDevice(device));
  if (node->type == GGML_TYPE_F32) {
    return launch(stream, params.activation,
                  static_cast<const float *>(input->data),
                  static_cast<float *>(node->data), elements, inner, half_axis);
  }
  if (node->type == GGML_TYPE_F16) {
    return launch_f16(
        stream, params.activation, static_cast<const half *>(input->data),
        static_cast<half *>(node->data), elements, inner, half_axis);
  }
  return false;
}

} // namespace ggml_ops_ext::cuda
