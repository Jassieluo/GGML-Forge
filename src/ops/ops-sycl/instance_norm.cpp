#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

template <typename T>
class InstanceNormSYCLKernel;

template <typename T>
static void launch_instance_norm(
    ::sycl::queue* queue, const T* x, const void* gamma, int gamma_type,
    const void* beta, int beta_type, T* dst, int64_t T_len, int64_t C, float eps
) {
    constexpr int group_size = 256;
    queue->submit([&](::sycl::handler &cgh) {
        ::sycl::local_accessor<float, 1> local_mem(::sycl::range<1>(group_size), cgh);
        cgh.parallel_for<InstanceNormSYCLKernel<T>>(
            ::sycl::nd_range<1>(C * group_size, group_size),
            [=](::sycl::nd_item<1> item) {
                int64_t c = item.get_group(0);
                int thread_id = item.get_local_id(0);
                float local_sum = 0.0f;
                for (int64_t t = thread_id; t < T_len; t += group_size) local_sum += (float)x[c * T_len + t];
                local_mem[thread_id] = local_sum;
                item.barrier(::sycl::access::fence_space::local_space);
                for (int offset = group_size / 2; offset > 0; offset /= 2) {
                    if (thread_id < offset) local_mem[thread_id] += local_mem[thread_id + offset];
                    item.barrier(::sycl::access::fence_space::local_space);
                }
                float mean = local_mem[0] / T_len;
                float local_var_sum = 0.0f;
                for (int64_t t = thread_id; t < T_len; t += group_size) {
                    float diff = (float)x[c * T_len + t] - mean;
                    local_var_sum += diff * diff;
                }
                local_mem[thread_id] = local_var_sum;
                item.barrier(::sycl::access::fence_space::local_space);
                for (int offset = group_size / 2; offset > 0; offset /= 2) {
                    if (thread_id < offset) local_mem[thread_id] += local_mem[thread_id + offset];
                    item.barrier(::sycl::access::fence_space::local_space);
                }
                float inv_std = 1.0f / ::sycl::sqrt(local_mem[0] / T_len + eps);
                float g = !gamma ? 1.0f : gamma_type == 0
                    ? ((const float*)gamma)[c] : (float)((const ::sycl::half*)gamma)[c];
                float b = !beta ? 0.0f : beta_type == 0
                    ? ((const float*)beta)[c] : (float)((const ::sycl::half*)beta)[c];
                for (int64_t t = thread_id; t < T_len; t += group_size) {
                    dst[c * T_len + t] = (T)(((float)x[c * T_len + t] - mean) * inv_std * g + b);
                }
            }
        );
    });
}

bool ggml_sycl_op_instance_norm(
    ggml_backend_t backend,
    struct ggml_tensor* node
) {
    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params)) return false;

    ::sycl::queue* q_queue = (::sycl::queue*)ggml_ops_ext_bridge_sycl_get_queue(backend);
    if (!q_queue) return false;

    struct ggml_tensor* x = params.x;
    struct ggml_tensor* gamma = params.gamma;
    struct ggml_tensor* beta = params.beta;
    struct ggml_tensor* dst = node;
    float eps = params.eps;

    int64_t T = x->ne[0];
    int64_t C = x->ne[1];

    int gamma_type = gamma && gamma->type == GGML_TYPE_F16 ? 1 : 0;
    int beta_type = beta && beta->type == GGML_TYPE_F16 ? 1 : 0;
    if (x->type == GGML_TYPE_F32) {
        launch_instance_norm(q_queue, (const float*)x->data,
            gamma ? gamma->data : nullptr, gamma_type, beta ? beta->data : nullptr, beta_type,
            (float*)dst->data, T, C, eps);
    } else if (x->type == GGML_TYPE_F16) {
        launch_instance_norm(q_queue, (const ::sycl::half*)x->data,
            gamma ? gamma->data : nullptr, gamma_type, beta ? beta->data : nullptr, beta_type,
            (::sycl::half*)dst->data, T, C, eps);
    } else {
        return false;
    }
    return true;
}

bool ggml_sycl_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_instance_norm(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
