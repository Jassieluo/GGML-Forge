#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <type_traits>

namespace ggml_ops_ext::sycl {
namespace {

struct strides4 { size_t v[4]; };
struct shape4 { int64_t v[4]; };
template <typename T> class CompareKernel;
class LogicalKernel;
template <typename T> class WhereKernel;

inline void decode(int64_t flat, shape4 shape, int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) { coord[axis] = flat % shape.v[axis]; flat /= shape.v[axis]; }
}
inline size_t offset(strides4 strides, const int64_t coord[4]) {
    return coord[0] * strides.v[0] + coord[1] * strides.v[1] + coord[2] * strides.v[2] + coord[3] * strides.v[3];
}
shape4 tensor_shape(const ggml_tensor* t) { return {{t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}; }
strides4 tensor_strides(const ggml_tensor* t) { return {{t->nb[0], t->nb[1], t->nb[2], t->nb[3]}}; }

template <typename T>
bool launch_compare(::sycl::queue* queue, ggml_tensor* output, const ggml_tensor* lhs,
                    const ggml_tensor* rhs, ops_compare_mode mode) {
    const int64_t count = ggml_nelements(output);
    const shape4 shape = tensor_shape(output);
    const strides4 lhs_s = tensor_strides(lhs), rhs_s = tensor_strides(rhs), out_s = tensor_strides(output);
    const T* a_ptr = static_cast<const T*>(lhs->data);
    const T* b_ptr = static_cast<const T*>(rhs->data);
    int32_t* out_ptr = static_cast<int32_t*>(output->data);
    queue->submit([&](::sycl::handler& h) {
        h.parallel_for<CompareKernel<T>>(::sycl::range<1>(static_cast<size_t>(count)), [=](::sycl::id<1> id) {
            const int64_t index = id[0]; int64_t coord[4]; decode(index, shape, coord);
            const T av = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(a_ptr) + offset(lhs_s, coord));
            const T bv = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(b_ptr) + offset(rhs_s, coord));
            bool result;
            if constexpr (std::is_same_v<T, int32_t>) {
                result = mode == ops_compare_mode::equal ? av == bv
                       : mode == ops_compare_mode::not_equal ? av != bv
                       : mode == ops_compare_mode::less ? av < bv
                       : mode == ops_compare_mode::less_equal ? av <= bv
                       : mode == ops_compare_mode::greater ? av > bv : av >= bv;
            } else {
                const float a = static_cast<float>(av), b = static_cast<float>(bv);
                result = mode == ops_compare_mode::equal ? a == b
                       : mode == ops_compare_mode::not_equal ? a != b
                       : mode == ops_compare_mode::less ? a < b
                       : mode == ops_compare_mode::less_equal ? a <= b
                       : mode == ops_compare_mode::greater ? a > b : a >= b;
            }
            *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(out_ptr) + offset(out_s, coord)) = result;
        });
    });
    return true;
}

bool launch_logical(::sycl::queue* queue, ggml_tensor* output, const ggml_tensor* lhs,
                    const ggml_tensor* rhs, ops_logical_mode mode) {
    const int64_t count = ggml_nelements(output); const shape4 shape = tensor_shape(output);
    const strides4 lhs_s = tensor_strides(lhs), rhs_s = rhs ? tensor_strides(rhs) : strides4{},
                   out_s = tensor_strides(output);
    const int32_t* a_ptr = static_cast<const int32_t*>(lhs->data);
    const int32_t* b_ptr = rhs ? static_cast<const int32_t*>(rhs->data) : nullptr;
    int32_t* out_ptr = static_cast<int32_t*>(output->data);
    queue->submit([&](::sycl::handler& h) {
        h.parallel_for<LogicalKernel>(::sycl::range<1>(static_cast<size_t>(count)), [=](::sycl::id<1> id) {
            const int64_t index = id[0]; int64_t coord[4]; decode(index, shape, coord);
            const bool a = *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(a_ptr) + offset(lhs_s, coord)) != 0;
            bool result = !a;
            if (mode != ops_logical_mode::logical_not) {
                const bool b = *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(b_ptr) + offset(rhs_s, coord)) != 0;
                result = mode == ops_logical_mode::logical_and ? a && b
                       : mode == ops_logical_mode::logical_or ? a || b : a != b;
            }
            *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(out_ptr) + offset(out_s, coord)) = result;
        });
    });
    return true;
}

template <typename T>
bool launch_where(::sycl::queue* queue, ggml_tensor* output, const ggml_tensor* condition,
                  const ggml_tensor* when_true, const ggml_tensor* when_false) {
    const int64_t count = ggml_nelements(output); const shape4 shape = tensor_shape(output);
    const strides4 cond_s = tensor_strides(condition), true_s = tensor_strides(when_true),
                   false_s = tensor_strides(when_false), out_s = tensor_strides(output);
    const int32_t* cond_ptr = static_cast<const int32_t*>(condition->data);
    const T* true_ptr = static_cast<const T*>(when_true->data);
    const T* false_ptr = static_cast<const T*>(when_false->data);
    T* out_ptr = static_cast<T*>(output->data);
    queue->submit([&](::sycl::handler& h) {
        h.parallel_for<WhereKernel<T>>(::sycl::range<1>(static_cast<size_t>(count)), [=](::sycl::id<1> id) {
            const int64_t index = id[0]; int64_t coord[4]; decode(index, shape, coord);
            const bool choose = *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(cond_ptr) + offset(cond_s, coord)) != 0;
            const char* source = reinterpret_cast<const char*>(choose ? true_ptr : false_ptr);
            const strides4 selected = choose ? true_s : false_s;
            *reinterpret_cast<T*>(reinterpret_cast<char*>(out_ptr) + offset(out_s, coord)) =
                *reinterpret_cast<const T*>(source + offset(selected, coord));
        });
    });
    return true;
}

} // namespace

bool ggml_sycl_op_selection_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) return false;
    if (node->op == GGML_OP_OPS_VIRT_COMPARE) {
        ops_selection_params p{}; std::memcpy(&p, node->op_params, sizeof(p));
        ggml_tensor* s[] = {node->src[0], node->src[1]};
        ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, 2, &p, sizeof(p), node};
        if (!ops_validate_compare(r)) return false; const auto mode = static_cast<ops_compare_mode>(p.mode);
        switch (s[0]->type) {
        case GGML_TYPE_F32: return launch_compare<float>(queue, node, s[0], s[1], mode);
        case GGML_TYPE_F16: return launch_compare<::sycl::half>(queue, node, s[0], s[1], mode);
        case GGML_TYPE_I32: return launch_compare<int32_t>(queue, node, s[0], s[1], mode);
        default: return false;
        }
    }
    if (node->op == GGML_OP_OPS_VIRT_LOGICAL) {
        ops_selection_params p{}; std::memcpy(&p, node->op_params, sizeof(p));
        const auto mode = static_cast<ops_logical_mode>(p.mode); ggml_tensor* s[] = {node->src[0], node->src[1]};
        const int n = mode == ops_logical_mode::logical_not ? 1 : 2;
        ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, n, &p, sizeof(p), node};
        return ops_validate_logical(r) && launch_logical(queue, node, s[0], s[1], mode);
    }
    ggml_tensor* s[] = {node->src[0], node->src[1], node->src[2]};
    ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, 3, nullptr, 0, node};
    if (!ops_validate_where(r)) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return launch_where<float>(queue, node, s[0], s[1], s[2]);
    case GGML_TYPE_F16: return launch_where<::sycl::half>(queue, node, s[0], s[1], s[2]);
    case GGML_TYPE_I32: return launch_where<int32_t>(queue, node, s[0], s[1], s[2]);
    default: return false;
    }
}

} // namespace ggml_ops_ext::sycl
