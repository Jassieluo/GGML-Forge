#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <limits>

namespace ggml_ops_ext::sycl {
namespace {

struct tensor_strides { size_t values[4]; };
template <typename T> class ReduceNDKernel;
template <typename T> class ArgReduceNDKernel;

inline void decode_index(int64_t flat, const int64_t shape[4], int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
}

template <typename T>
bool launch_reduce(::sycl::queue* queue, const ggml_tensor* input, ggml_tensor* output,
                   const ops_reduce_nd_desc& desc) {
    const tensor_strides input_strides = {{input->nb[0], input->nb[1], input->nb[2], input->nb[3]}};
    const tensor_strides output_strides = {{output->nb[0], output->nb[1], output->nb[2], output->nb[3]}};
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);
    constexpr size_t local_size = 256;
    const size_t global_size = static_cast<size_t>((desc.output_count + local_size - 1) / local_size) * local_size;
    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<ReduceNDKernel<T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int64_t output_index = item.get_global_linear_id();
                if (output_index >= desc.output_count) return;
                int64_t output_coord[4];
                int64_t input_coord[4] = {};
                decode_index(output_index, desc.output_shape, output_coord);
                int compact_axis = 0;
                for (int axis = 0; axis < 4; ++axis) {
                    if (!(desc.axis_mask & (1u << axis))) {
                        input_coord[axis] = output_coord[desc.keep_dims ? axis : compact_axis++];
                    }
                }
                float result = desc.mode == ops_reduce_mode::maximum ||
                                       desc.mode == ops_reduce_mode::logsumexp
                                   ? -std::numeric_limits<float>::infinity()
                               : desc.mode == ops_reduce_mode::minimum
                                   ? std::numeric_limits<float>::infinity()
                               : desc.mode == ops_reduce_mode::product
                                   ? 1.0f
                                   : 0.0f;
                float auxiliary = 0.0f;
                for (int64_t reduction_index = 0; reduction_index < desc.reduction_count; ++reduction_index) {
                    int64_t cursor = reduction_index;
                    for (int axis = 0; axis < 4; ++axis) {
                        if (desc.axis_mask & (1u << axis)) {
                            input_coord[axis] = cursor % desc.input_shape[axis];
                            cursor /= desc.input_shape[axis];
                        }
                    }
                    size_t offset = 0;
                    for (int axis = 0; axis < 4; ++axis) offset += input_coord[axis] * input_strides.values[axis];
                    const float value = static_cast<float>(
                        *reinterpret_cast<const T*>(reinterpret_cast<const char*>(source) + offset));
                    if (desc.mode == ops_reduce_mode::maximum) result = ::sycl::fmax(result, value);
                    else if (desc.mode == ops_reduce_mode::minimum) result = ::sycl::fmin(result, value);
                    else if (desc.mode == ops_reduce_mode::product) result *= value;
                    else if (desc.mode == ops_reduce_mode::variance ||
                             desc.mode == ops_reduce_mode::standard_deviation) {
                        const float delta = value - result;
                        result += delta / static_cast<float>(reduction_index + 1);
                        auxiliary += delta * (value - result);
                    } else if (desc.mode == ops_reduce_mode::logsumexp) {
                        if (value <= result) auxiliary += ::sycl::exp(value - result);
                        else {
                            auxiliary = auxiliary * ::sycl::exp(result - value) + 1.0f;
                            result = value;
                        }
                    } else result += value;
                }
                if (desc.mode == ops_reduce_mode::mean) result /= static_cast<float>(desc.reduction_count);
                else if (desc.mode == ops_reduce_mode::variance ||
                         desc.mode == ops_reduce_mode::standard_deviation) {
                    result = auxiliary / static_cast<float>(desc.reduction_count - desc.correction);
                    if (desc.mode == ops_reduce_mode::standard_deviation) result = ::sycl::sqrt(result);
                } else if (desc.mode == ops_reduce_mode::logsumexp) result += ::sycl::log(auxiliary);
                size_t output_offset = 0;
                for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output_strides.values[axis];
                *reinterpret_cast<T*>(reinterpret_cast<char*>(destination) + output_offset) = static_cast<T>(result);
            });
    });
    return true;
}

template <typename T>
bool launch_arg_reduce(::sycl::queue* queue, const ggml_tensor* input, ggml_tensor* output,
                       const ops_arg_reduce_nd_desc& desc) {
    const tensor_strides input_strides = {{input->nb[0], input->nb[1], input->nb[2], input->nb[3]}};
    const tensor_strides output_strides = {{output->nb[0], output->nb[1], output->nb[2], output->nb[3]}};
    const T* source = static_cast<const T*>(input->data);
    int32_t* destination = static_cast<int32_t*>(output->data);
    constexpr size_t local_size = 256;
    const size_t global_size = static_cast<size_t>((desc.output_count + local_size - 1) / local_size) * local_size;
    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<ArgReduceNDKernel<T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int64_t output_index = item.get_global_linear_id();
                if (output_index >= desc.output_count) return;
                int64_t output_coord[4];
                int64_t input_coord[4] = {};
                decode_index(output_index, desc.output_shape, output_coord);
                int compact_axis = 0;
                for (int axis = 0; axis < 4; ++axis) {
                    if (!(desc.axis_mask & (1u << axis))) {
                        input_coord[axis] = output_coord[desc.keep_dims ? axis : compact_axis++];
                    }
                }
                float best = desc.mode == ops_arg_reduce_mode::maximum
                                 ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
                int32_t best_index = 0;
                for (int64_t reduction_index = 0; reduction_index < desc.reduction_count; ++reduction_index) {
                    int64_t cursor = reduction_index;
                    for (int axis = 0; axis < 4; ++axis) {
                        if (desc.axis_mask & (1u << axis)) {
                            input_coord[axis] = cursor % desc.input_shape[axis];
                            cursor /= desc.input_shape[axis];
                        }
                    }
                    size_t offset = 0;
                    for (int axis = 0; axis < 4; ++axis) offset += input_coord[axis] * input_strides.values[axis];
                    const float value = static_cast<float>(
                        *reinterpret_cast<const T*>(reinterpret_cast<const char*>(source) + offset));
                    const bool better = desc.mode == ops_arg_reduce_mode::maximum ? value > best : value < best;
                    if (better) {
                        best = value;
                        best_index = static_cast<int32_t>(reduction_index);
                    }
                }
                size_t output_offset = 0;
                for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output_strides.values[axis];
                *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(destination) + output_offset) = best_index;
            });
    });
    return true;
}

} // namespace

bool ggml_sycl_op_reduce_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources,
                           1, &params, sizeof(params), node};
    ops_reduce_nd_desc desc;
    if (!ops_validate_reduce_nd_contract(request, &desc)) return false;
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return launch_reduce<float>(queue, sources[0], node, desc);
    case GGML_TYPE_F16: return launch_reduce<::sycl::half>(queue, sources[0], node, desc);
    default: return false;
    }
}

bool ggml_sycl_op_arg_reduce_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_arg_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources,
                           1, &params, sizeof(params), node};
    ops_arg_reduce_nd_desc desc;
    if (!ops_validate_arg_reduce_nd_contract(request, &desc)) return false;
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) return false;
    switch (sources[0]->type) {
    case GGML_TYPE_F32: return launch_arg_reduce<float>(queue, sources[0], node, desc);
    case GGML_TYPE_F16: return launch_arg_reduce<::sycl::half>(queue, sources[0], node, desc);
    default: return false;
    }
}

} // namespace ggml_ops_ext::sycl
