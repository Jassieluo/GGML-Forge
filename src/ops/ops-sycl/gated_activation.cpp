#include "ops/ops.h"
#include "ops_sycl.h"

#include "common.hpp"
#include "ggml-backend.h"
#include "ggml.h"

namespace ggml_ops_ext::sycl {
namespace {

template <int Activation, typename T> class GatedActivationKernel;

template <int Activation> float activate(float value) {
  if constexpr (Activation == 0) {
    return value / (1.0f + ::sycl::exp(-value));
  }
  if constexpr (Activation == 1) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    return 0.5f * value *
           (1.0f + ::sycl::tanh(sqrt_two_over_pi * value *
                                (1.0f + 0.044715f * value * value)));
  }
  if constexpr (Activation == 2) {
    return ::sycl::fmax(value, 0.0f);
  }
  return value;
}

template <int Activation, typename T>
void launch_kernel(::sycl::queue *queue, const T *input, T *output,
                   int64_t elements, int64_t inner, int64_t half_axis) {
  queue->submit([&](::sycl::handler &handler) {
    handler.parallel_for<GatedActivationKernel<Activation, T>>(
        ::sycl::range<1>(static_cast<size_t>(elements)), [=](::sycl::id<1> id) {
          const int64_t index = id[0];
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
        });
  });
}

template <typename T>
bool launch(::sycl::queue *queue, int activation, const T *input, T *output,
            int64_t elements, int64_t inner, int64_t half_axis) {
  switch (activation) {
  case 0:
    launch_kernel<0>(queue, input, output, elements, inner, half_axis);
    return true;
  case 1:
    launch_kernel<1>(queue, input, output, elements, inner, half_axis);
    return true;
  case 2:
    launch_kernel<2>(queue, input, output, elements, inner, half_axis);
    return true;
  case 3:
    launch_kernel<3>(queue, input, output, elements, inner, half_axis);
    return true;
  default:
    return false;
  }
}

} // namespace

bool ggml_sycl_op_gated_activation_entry(ggml_backend_t backend,
                                         ggml_tensor *node) {
  if (!node || !node->src[0])
    return false;
  auto *queue =
      static_cast<::sycl::queue *>(ggml_ops_ext_bridge_sycl_get_queue(backend));
  if (!queue)
    return false;
  ops_gated_activation_params params;
  std::memcpy(&params, node->op_params, sizeof(params));
  const ggml_tensor *input = node->src[0];
  int64_t inner = 1;
  for (int axis = 0; axis < params.axis; ++axis)
    inner *= input->ne[axis];
  const int64_t half_axis = input->ne[params.axis] / 2;
  const int64_t elements = ggml_nelements(node);
  if (node->type == GGML_TYPE_F32) {
    return launch(queue, params.activation,
                  static_cast<const float *>(input->data),
                  static_cast<float *>(node->data), elements, inner, half_axis);
  }
  if (node->type == GGML_TYPE_F16) {
    return launch(queue, params.activation,
                  static_cast<const ::sycl::half *>(input->data),
                  static_cast<::sycl::half *>(node->data), elements, inner,
                  half_axis);
  }
  return false;
}

} // namespace ggml_ops_ext::sycl
