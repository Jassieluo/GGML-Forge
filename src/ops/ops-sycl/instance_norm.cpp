#include "ops/ops.h"
#include "ops_sycl.h"

#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"

namespace ggml_ops_ext::sycl {
namespace {

template <typename T> class InstanceNormSubgroupKernel;

template <typename T>
void launch(::sycl::queue* queue, const T* input, const void* gamma, int gamma_type, const void* beta, int beta_type,
            T* output, int64_t length, int64_t channels, int64_t groups, float eps) {
    constexpr size_t subgroup_size = 32;
    constexpr size_t workgroup_size = 256;
    constexpr size_t groups_per_workgroup = workgroup_size / subgroup_size;
    const size_t workgroups = (static_cast<size_t>(groups) + groups_per_workgroup - 1) / groups_per_workgroup;
    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<InstanceNormSubgroupKernel<T>>(
            ::sycl::nd_range<1>(workgroups * workgroup_size, workgroup_size),
            [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(subgroup_size)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t group = static_cast<int64_t>(item.get_group_linear_id()) * groups_per_workgroup +
                                      static_cast<int64_t>(subgroup.get_group_linear_id());
                if (group >= groups) {
                    return;
                }
                const int64_t lane = subgroup.get_local_linear_id();
                const T* input_group = input + group * length;
                float local_sum = 0.0f;
                float local_squared_sum = 0.0f;
                for (int64_t index = lane; index < length; index += subgroup_size) {
                    const float value = static_cast<float>(input_group[index]);
                    local_sum += value;
                    local_squared_sum += value * value;
                }
                const float sum = ::sycl::reduce_over_group(subgroup, local_sum, ::sycl::plus<float>());
                const float squared_sum = ::sycl::reduce_over_group(subgroup, local_squared_sum, ::sycl::plus<float>());
                const float mean = sum / static_cast<float>(length);
                const float variance = ::sycl::fmax(squared_sum / static_cast<float>(length) - mean * mean, 0.0f);
                const float inverse_std = ::sycl::rsqrt(variance + eps);
                const int64_t channel = group % channels;
                const float scale = !gamma ? 1.0f
                                    : gamma_type == 0
                                        ? static_cast<const float*>(gamma)[channel]
                                        : static_cast<float>(static_cast<const ::sycl::half*>(gamma)[channel]);
                const float shift = !beta ? 0.0f
                                    : beta_type == 0
                                        ? static_cast<const float*>(beta)[channel]
                                        : static_cast<float>(static_cast<const ::sycl::half*>(beta)[channel]);
                T* output_group = output + group * length;
                for (int64_t index = lane; index < length; index += subgroup_size) {
                    output_group[index] =
                        static_cast<T>((static_cast<float>(input_group[index]) - mean) * inverse_std * scale + shift);
                }
            });
    });
}

} // namespace

bool ggml_sycl_op_instance_norm(ggml_backend_t backend, ggml_tensor* node) {
    ops_instance_norm_params params;
    if (!ops_extract_instance_norm_params(node, params)) {
        return false;
    }
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }
    const int64_t length = params.x->ne[0];
    const int64_t channels = params.x->ne[1];
    const int64_t groups = ggml_nelements(params.x) / length;
    const int gamma_type = params.gamma && params.gamma->type == GGML_TYPE_F16 ? 1 : 0;
    const int beta_type = params.beta && params.beta->type == GGML_TYPE_F16 ? 1 : 0;
    if (params.x->type == GGML_TYPE_F32) {
        launch(queue, static_cast<const float*>(params.x->data), params.gamma ? params.gamma->data : nullptr,
               gamma_type, params.beta ? params.beta->data : nullptr, beta_type, static_cast<float*>(node->data),
               length, channels, groups, params.eps);
        return true;
    }
    if (params.x->type == GGML_TYPE_F16) {
        launch(queue, static_cast<const ::sycl::half*>(params.x->data), params.gamma ? params.gamma->data : nullptr,
               gamma_type, params.beta ? params.beta->data : nullptr, beta_type, static_cast<::sycl::half*>(node->data),
               length, channels, groups, params.eps);
        return true;
    }
    return false;
}

bool ggml_sycl_op_instance_norm_entry(ggml_backend_t backend, ggml_tensor* node) {
    return ggml_sycl_op_instance_norm(backend, node);
}

} // namespace ggml_ops_ext::sycl
