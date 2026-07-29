#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>
#include <type_traits>

namespace ggml_ops_ext::cuda {
namespace {

struct strides4 { size_t v[4]; };
struct shape4 { int64_t v[4]; };

__device__ void decode(int64_t flat, shape4 shape, int64_t coord[4]) {
    for (int axis = 0; axis < 4; ++axis) { coord[axis] = flat % shape.v[axis]; flat /= shape.v[axis]; }
}
__device__ size_t offset(strides4 strides, const int64_t coord[4]) {
    return coord[0] * strides.v[0] + coord[1] * strides.v[1] + coord[2] * strides.v[2] + coord[3] * strides.v[3];
}
template <typename T> __device__ float as_float(T value) { return static_cast<float>(value); }
template <> __device__ float as_float<half>(half value) { return __half2float(value); }
template <> __device__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
__device__ bool compare_values(T lhs, T rhs, ops_compare_mode mode) {
    if constexpr (!std::is_same_v<T, int32_t>) {
        const float a = as_float(lhs), b = as_float(rhs);
        switch (mode) {
        case ops_compare_mode::equal: return a == b; case ops_compare_mode::not_equal: return a != b;
        case ops_compare_mode::less: return a < b; case ops_compare_mode::less_equal: return a <= b;
        case ops_compare_mode::greater: return a > b; case ops_compare_mode::greater_equal: return a >= b;
        }
    } else {
        switch (mode) {
        case ops_compare_mode::equal: return lhs == rhs; case ops_compare_mode::not_equal: return lhs != rhs;
        case ops_compare_mode::less: return lhs < rhs; case ops_compare_mode::less_equal: return lhs <= rhs;
        case ops_compare_mode::greater: return lhs > rhs; case ops_compare_mode::greater_equal: return lhs >= rhs;
        }
    }
    return false;
}

template <typename T>
__global__ void compare_kernel(const T* lhs, const T* rhs, int32_t* output, int64_t count,
                               shape4 shape, strides4 lhs_s, strides4 rhs_s, strides4 out_s,
                               ops_compare_mode mode) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t coord[4]; decode(index, shape, coord);
    const T a = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(lhs) + offset(lhs_s, coord));
    const T b = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(rhs) + offset(rhs_s, coord));
    *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(output) + offset(out_s, coord)) = compare_values(a, b, mode);
}

__global__ void logical_kernel(const int32_t* lhs, const int32_t* rhs, int32_t* output, int64_t count,
                               shape4 shape, strides4 lhs_s, strides4 rhs_s, strides4 out_s,
                               ops_logical_mode mode) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t coord[4]; decode(index, shape, coord);
    const bool a = *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(lhs) + offset(lhs_s, coord)) != 0;
    bool result = !a;
    if (mode != ops_logical_mode::logical_not) {
        const bool b = *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(rhs) + offset(rhs_s, coord)) != 0;
        result = mode == ops_logical_mode::logical_and ? a && b
               : mode == ops_logical_mode::logical_or ? a || b : a != b;
    }
    *reinterpret_cast<int32_t*>(reinterpret_cast<char*>(output) + offset(out_s, coord)) = result;
}

template <typename T>
__global__ void where_kernel(const int32_t* condition, const T* when_true, const T* when_false, T* output,
                             int64_t count, shape4 shape, strides4 cond_s, strides4 true_s,
                             strides4 false_s, strides4 out_s) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    int64_t coord[4]; decode(index, shape, coord);
    const bool choose = *reinterpret_cast<const int32_t*>(
        reinterpret_cast<const char*>(condition) + offset(cond_s, coord)) != 0;
    const char* source = reinterpret_cast<const char*>(choose ? when_true : when_false);
    const strides4 selected = choose ? true_s : false_s;
    *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + offset(out_s, coord)) =
        *reinterpret_cast<const T*>(source + offset(selected, coord));
}

shape4 tensor_shape(const ggml_tensor* t) { return {{t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}; }
strides4 tensor_strides(const ggml_tensor* t) { return {{t->nb[0], t->nb[1], t->nb[2], t->nb[3]}}; }

template <typename T>
bool launch_compare(cudaStream_t stream, ggml_tensor* output, const ggml_tensor* lhs, const ggml_tensor* rhs,
                    ops_compare_mode mode) {
    const int64_t count = ggml_nelements(output);
    compare_kernel<T><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(lhs->data), static_cast<const T*>(rhs->data), static_cast<int32_t*>(output->data),
        count, tensor_shape(output), tensor_strides(lhs), tensor_strides(rhs), tensor_strides(output), mode);
    return cudaGetLastError() == cudaSuccess;
}

template <typename T>
bool launch_where(cudaStream_t stream, ggml_tensor* output, const ggml_tensor* condition,
                  const ggml_tensor* when_true, const ggml_tensor* when_false) {
    const int64_t count = ggml_nelements(output);
    where_kernel<T><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const int32_t*>(condition->data), static_cast<const T*>(when_true->data),
        static_cast<const T*>(when_false->data), static_cast<T*>(output->data), count, tensor_shape(output),
        tensor_strides(condition), tensor_strides(when_true), tensor_strides(when_false), tensor_strides(output));
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_selection_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) return false;
    if (node->op == GGML_OP_OPS_VIRT_COMPARE) {
        ops_selection_params params{}; std::memcpy(&params, node->op_params, sizeof(params));
        ggml_tensor* s[] = {node->src[0], node->src[1]};
        ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, 2, &params, sizeof(params), node};
        if (!ops_validate_compare(r)) return false;
        const auto mode = static_cast<ops_compare_mode>(params.mode);
        switch (s[0]->type) {
        case GGML_TYPE_F32: return launch_compare<float>(stream, node, s[0], s[1], mode);
        case GGML_TYPE_F16: return launch_compare<half>(stream, node, s[0], s[1], mode);
        case GGML_TYPE_BF16: return launch_compare<__nv_bfloat16>(stream, node, s[0], s[1], mode);
        case GGML_TYPE_I32: return launch_compare<int32_t>(stream, node, s[0], s[1], mode);
        default: return false;
        }
    }
    if (node->op == GGML_OP_OPS_VIRT_LOGICAL) {
        ops_selection_params params{}; std::memcpy(&params, node->op_params, sizeof(params));
        const auto mode = static_cast<ops_logical_mode>(params.mode);
        ggml_tensor* s[] = {node->src[0], node->src[1]};
        const int n = mode == ops_logical_mode::logical_not ? 1 : 2;
        ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, n, &params, sizeof(params), node};
        if (!ops_validate_logical(r)) return false;
        const int64_t count = ggml_nelements(node);
        logical_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const int32_t*>(s[0]->data), static_cast<const int32_t*>(s[1] ? s[1]->data : nullptr),
            static_cast<int32_t*>(node->data), count, tensor_shape(node), tensor_strides(s[0]),
            s[1] ? tensor_strides(s[1]) : strides4{}, tensor_strides(node), mode);
        return cudaGetLastError() == cudaSuccess;
    }
    ggml_tensor* s[] = {node->src[0], node->src[1], node->src[2]};
    ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, 3, nullptr, 0, node};
    if (!ops_validate_where(r)) return false;
    switch (node->type) {
    case GGML_TYPE_F32: return launch_where<float>(stream, node, s[0], s[1], s[2]);
    case GGML_TYPE_F16: return launch_where<half>(stream, node, s[0], s[1], s[2]);
    case GGML_TYPE_BF16: return launch_where<__nv_bfloat16>(stream, node, s[0], s[1], s[2]);
    case GGML_TYPE_I32: return launch_where<int32_t>(stream, node, s[0], s[1], s[2]);
    default: return false;
    }
}

} // namespace ggml_ops_ext::cuda
