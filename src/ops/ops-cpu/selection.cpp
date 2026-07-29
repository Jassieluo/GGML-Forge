#include "ops/cpu.h"
#include "ops/ops.h"

namespace ggml_ops_ext::cpu {
namespace {

inline void decode(int64_t flat, const ggml_tensor* tensor, int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) {
        coord[axis] = flat % tensor->ne[axis];
        flat /= tensor->ne[axis];
    }
}
inline size_t offset(const ggml_tensor* tensor, const int64_t coord[4]) {
    return coord[0] * tensor->nb[0] + coord[1] * tensor->nb[1] +
           coord[2] * tensor->nb[2] + coord[3] * tensor->nb[3];
}

template <ggml_type Type> float load_float(const ggml_tensor* tensor, size_t byte_offset) {
    const char* address = static_cast<const char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) return *reinterpret_cast<const float*>(address);
    if constexpr (Type == GGML_TYPE_F16) return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <typename T> bool compare_value(T lhs, T rhs, ops_compare_mode mode) {
    switch (mode) {
    case ops_compare_mode::equal: return lhs == rhs;
    case ops_compare_mode::not_equal: return lhs != rhs;
    case ops_compare_mode::less: return lhs < rhs;
    case ops_compare_mode::less_equal: return lhs <= rhs;
    case ops_compare_mode::greater: return lhs > rhs;
    case ops_compare_mode::greater_equal: return lhs >= rhs;
    }
    return false;
}

template <ggml_type Type>
bool execute_compare(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* lhs,
                     const ggml_tensor* rhs, ops_compare_mode mode) {
    const int64_t count = ggml_nelements(output);
    const int threads = backend_thread_count(backend);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < count; ++index) {
        int64_t coord[4]; decode(index, output, coord);
        bool result;
        if constexpr (Type == GGML_TYPE_I32) {
            const int32_t a = *reinterpret_cast<const int32_t*>(static_cast<const char*>(lhs->data) + offset(lhs, coord));
            const int32_t b = *reinterpret_cast<const int32_t*>(static_cast<const char*>(rhs->data) + offset(rhs, coord));
            result = compare_value(a, b, mode);
        } else {
            result = compare_value(load_float<Type>(lhs, offset(lhs, coord)),
                                   load_float<Type>(rhs, offset(rhs, coord)), mode);
        }
        *reinterpret_cast<int32_t*>(static_cast<char*>(output->data) + offset(output, coord)) = result ? 1 : 0;
    }
    return true;
}

bool execute_logical(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* lhs,
                     const ggml_tensor* rhs, ops_logical_mode mode) {
    const int64_t count = ggml_nelements(output);
    const int threads = backend_thread_count(backend);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < count; ++index) {
        int64_t coord[4]; decode(index, output, coord);
        const bool a = *reinterpret_cast<const int32_t*>(static_cast<const char*>(lhs->data) + offset(lhs, coord)) != 0;
        bool result = !a;
        if (mode != ops_logical_mode::logical_not) {
            const bool b = *reinterpret_cast<const int32_t*>(static_cast<const char*>(rhs->data) + offset(rhs, coord)) != 0;
            result = mode == ops_logical_mode::logical_and ? a && b
                   : mode == ops_logical_mode::logical_or ? a || b : a != b;
        }
        *reinterpret_cast<int32_t*>(static_cast<char*>(output->data) + offset(output, coord)) = result ? 1 : 0;
    }
    return true;
}

template <typename T>
bool execute_where(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* condition,
                   const ggml_tensor* when_true, const ggml_tensor* when_false) {
    const int64_t count = ggml_nelements(output);
    const int threads = backend_thread_count(backend);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < count; ++index) {
        int64_t coord[4]; decode(index, output, coord);
        const bool choose_true = *reinterpret_cast<const int32_t*>(
            static_cast<const char*>(condition->data) + offset(condition, coord)) != 0;
        const ggml_tensor* source = choose_true ? when_true : when_false;
        *reinterpret_cast<T*>(static_cast<char*>(output->data) + offset(output, coord)) =
            *reinterpret_cast<const T*>(static_cast<const char*>(source->data) + offset(source, coord));
    }
    return true;
}

} // namespace

bool ops_cpu_op_selection(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    if (node->op == GGML_OP_OPS_VIRT_COMPARE) {
        ops_selection_params params{}; std::memcpy(&params, node->op_params, sizeof(params));
        ggml_tensor* sources[] = {node->src[0], node->src[1]};
        ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources, 2,
                               &params, sizeof(params), node};
        if (!ops_validate_compare(request)) return false;
        const auto mode = static_cast<ops_compare_mode>(params.mode);
        switch (sources[0]->type) {
        case GGML_TYPE_F32: return execute_compare<GGML_TYPE_F32>(backend, node, sources[0], sources[1], mode);
        case GGML_TYPE_F16: return execute_compare<GGML_TYPE_F16>(backend, node, sources[0], sources[1], mode);
        case GGML_TYPE_BF16: return execute_compare<GGML_TYPE_BF16>(backend, node, sources[0], sources[1], mode);
        case GGML_TYPE_I32: return execute_compare<GGML_TYPE_I32>(backend, node, sources[0], sources[1], mode);
        default: return false;
        }
    }
    if (node->op == GGML_OP_OPS_VIRT_LOGICAL) {
        ops_selection_params params{}; std::memcpy(&params, node->op_params, sizeof(params));
        const auto mode = static_cast<ops_logical_mode>(params.mode);
        ggml_tensor* sources[] = {node->src[0], node->src[1]};
        const int count = mode == ops_logical_mode::logical_not ? 1 : 2;
        ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources, count,
                               &params, sizeof(params), node};
        return ops_validate_logical(request) && execute_logical(backend, node, sources[0], sources[1], mode);
    }
    ggml_tensor* sources[] = {node->src[0], node->src[1], node->src[2]};
    ops_request request = {ggml_backend_get_device(backend), static_cast<int>(node->op), sources, 3,
                           nullptr, 0, node};
    if (!ops_validate_where(request)) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return execute_where<float>(backend, node, sources[0], sources[1], sources[2]);
    case GGML_TYPE_F16: return execute_where<ggml_fp16_t>(backend, node, sources[0], sources[1], sources[2]);
    case GGML_TYPE_BF16: return execute_where<ggml_bf16_t>(backend, node, sources[0], sources[1], sources[2]);
    case GGML_TYPE_I32: return execute_where<int32_t>(backend, node, sources[0], sources[1], sources[2]);
    default: return false;
    }
}

} // namespace ggml_ops_ext::cpu
