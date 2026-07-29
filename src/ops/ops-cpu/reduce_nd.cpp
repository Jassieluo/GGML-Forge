#include "ops/cpu.h"
#include "ops/ops.h"

#include <cmath>
#include <limits>

namespace ggml_ops_ext::cpu {
namespace {

template <ggml_type Type> float load_value(const ggml_tensor* tensor, size_t offset) {
    const char* address = static_cast<const char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) return *reinterpret_cast<const float*>(address);
    if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <ggml_type Type> void store_value(ggml_tensor* tensor, size_t offset, float value) {
    char* address = static_cast<char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) *reinterpret_cast<float*>(address) = value;
    else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(address) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(address) = ggml_fp32_to_bf16(value);
    }
}

inline void decode_index(int64_t flat, const int64_t shape[4], int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
}

template <ggml_type Type>
bool execute_reduce(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                    const ops_reduce_nd_desc& desc) {
    const int thread_count = backend_thread_count(backend);
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int64_t output_index = 0; output_index < desc.output_count; ++output_index) {
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
        for (int64_t reduction_index = 0; reduction_index < desc.reduction_count;
             ++reduction_index) {
            int64_t cursor = reduction_index;
            for (int axis = 0; axis < 4; ++axis) {
                if (desc.axis_mask & (1u << axis)) {
                    input_coord[axis] = cursor % desc.input_shape[axis];
                    cursor /= desc.input_shape[axis];
                }
            }
            size_t input_offset = 0;
            for (int axis = 0; axis < 4; ++axis) input_offset += input_coord[axis] * input->nb[axis];
            const float value = load_value<Type>(input, input_offset);
            if (desc.mode == ops_reduce_mode::maximum) result = value > result ? value : result;
            else if (desc.mode == ops_reduce_mode::minimum) result = value < result ? value : result;
            else if (desc.mode == ops_reduce_mode::product) result *= value;
            else if (desc.mode == ops_reduce_mode::variance ||
                     desc.mode == ops_reduce_mode::standard_deviation) {
                const float delta = value - result;
                result += delta / static_cast<float>(reduction_index + 1);
                auxiliary += delta * (value - result);
            } else if (desc.mode == ops_reduce_mode::logsumexp) {
                if (value <= result) auxiliary += std::exp(value - result);
                else {
                    auxiliary = auxiliary * std::exp(result - value) + 1.0f;
                    result = value;
                }
            } else result += value;
        }
        if (desc.mode == ops_reduce_mode::mean) result /= static_cast<float>(desc.reduction_count);
        else if (desc.mode == ops_reduce_mode::variance ||
                 desc.mode == ops_reduce_mode::standard_deviation) {
            result = auxiliary / static_cast<float>(desc.reduction_count - desc.correction);
            if (desc.mode == ops_reduce_mode::standard_deviation) result = std::sqrt(result);
        } else if (desc.mode == ops_reduce_mode::logsumexp) {
            result += std::log(auxiliary);
        }
        size_t output_offset = 0;
        for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output->nb[axis];
        store_value<Type>(output, output_offset, result);
    }
    return true;
}

template <ggml_type Type>
bool execute_arg_reduce(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                        const ops_arg_reduce_nd_desc& desc) {
    const int thread_count = backend_thread_count(backend);
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int64_t output_index = 0; output_index < desc.output_count; ++output_index) {
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
            size_t input_offset = 0;
            for (int axis = 0; axis < 4; ++axis) input_offset += input_coord[axis] * input->nb[axis];
            const float value = load_value<Type>(input, input_offset);
            const bool better = desc.mode == ops_arg_reduce_mode::maximum ? value > best : value < best;
            if (better) {
                best = value;
                best_index = static_cast<int32_t>(reduction_index);
            }
        }
        size_t output_offset = 0;
        for (int axis = 0; axis < 4; ++axis) output_offset += output_coord[axis] * output->nb[axis];
        *reinterpret_cast<int32_t*>(static_cast<char*>(output->data) + output_offset) = best_index;
    }
    return true;
}

} // namespace

bool ops_cpu_op_reduce_nd(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op),
                           sources, 1, &params, sizeof(params), node};
    ops_reduce_nd_desc desc;
    if (!ops_validate_reduce_nd_contract(request, &desc)) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return execute_reduce<GGML_TYPE_F32>(backend, node, sources[0], desc);
    case GGML_TYPE_F16: return execute_reduce<GGML_TYPE_F16>(backend, node, sources[0], desc);
    case GGML_TYPE_BF16: return execute_reduce<GGML_TYPE_BF16>(backend, node, sources[0], desc);
    default: return false;
    }
}

bool ops_cpu_op_arg_reduce_nd(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    ops_arg_reduce_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op),
                           sources, 1, &params, sizeof(params), node};
    ops_arg_reduce_nd_desc desc;
    if (!ops_validate_arg_reduce_nd_contract(request, &desc)) return false;
    switch (sources[0]->type) {
    case GGML_TYPE_F32: return execute_arg_reduce<GGML_TYPE_F32>(backend, node, sources[0], desc);
    case GGML_TYPE_F16: return execute_arg_reduce<GGML_TYPE_F16>(backend, node, sources[0], desc);
    case GGML_TYPE_BF16: return execute_arg_reduce<GGML_TYPE_BF16>(backend, node, sources[0], desc);
    default: return false;
    }
}

} // namespace ggml_ops_ext::cpu
