#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "common.hpp"

namespace ggml_ops_ext {
namespace sycl {

class InstanceNormSYCLKernelF32;

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

    if (x->type == GGML_TYPE_F32) {
        const float* x_d = (const float*)x->data;
        const float* gamma_d = gamma ? (const float*)gamma->data : nullptr;
        const float* beta_d = beta ? (const float*)beta->data : nullptr;
        float* dst_d = (float*)dst->data;

        int group_size = 256;

        q_queue->submit([&](::sycl::handler &cgh) {
            // local_mem: used for group reductions
            ::sycl::local_accessor<float, 1> local_mem(::sycl::range<1>(group_size), cgh);

            cgh.parallel_for<InstanceNormSYCLKernelF32>(
                ::sycl::nd_range<1>(C * group_size, group_size),
                [=](::sycl::nd_item<1> item) {
                    int64_t c = item.get_group(0);
                    int thread_id = item.get_local_id(0);

                    // 1. Load sequence and compute mean
                    float local_sum = 0.0f;
                    for (int64_t t = thread_id; t < T; t += group_size) {
                        local_sum += x_d[c * T + t];
                    }
                    local_mem[thread_id] = local_sum;
                    item.barrier(::sycl::access::fence_space::local_space);

                    for (int offset = group_size / 2; offset > 0; offset /= 2) {
                        if (thread_id < offset) {
                            local_mem[thread_id] += local_mem[thread_id + offset];
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                    float mean = local_mem[0] / T;

                    // 2. Compute variance using global memory reads
                    float local_var_sum = 0.0f;
                    for (int64_t t = thread_id; t < T; t += group_size) {
                        float diff = x_d[c * T + t] - mean;
                        local_var_sum += diff * diff;
                    }
                    local_mem[thread_id] = local_var_sum;
                    item.barrier(::sycl::access::fence_space::local_space);

                    for (int offset = group_size / 2; offset > 0; offset /= 2) {
                        if (thread_id < offset) {
                            local_mem[thread_id] += local_mem[thread_id + offset];
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                    float var = local_mem[0] / T;
                    float inv_std = 1.0f / ::sycl::sqrt(var + eps);

                    float g = gamma_d ? gamma_d[c] : 1.0f;
                    float b = beta_d ? beta_d[c] : 0.0f;

                    // 3. Write out directly reading from global memory
                    for (int64_t t = thread_id; t < T; t += group_size) {
                        dst_d[c * T + t] = (x_d[c * T + t] - mean) * inv_std * g + b;
                    }
                }
            );
        });
    } else {
        return false;
    }
    q_queue->wait();
    return true;
}

bool ggml_sycl_op_instance_norm_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    return ggml_sycl_op_instance_norm(backend, node);
}

} // namespace sycl
} // namespace ggml_ops_ext
