#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace ggml_ops_ext::cpu {
namespace {

template <ggml_type Type> float load_value(const ggml_tensor* tensor, size_t offset) {
    const char* address = static_cast<const char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(address);
    }
    if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <ggml_type Type> void store_value(ggml_tensor* tensor, size_t offset, float value) {
    char* address = static_cast<char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(address) = value;
    } else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(address) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(address) = ggml_fp32_to_bf16(value);
    }
}

template <int Dims>
size_t spatial_offset(const ggml_tensor* tensor, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                      const int64_t spatial_size[3]) {
    if constexpr (Dims == 1) {
        return batch * tensor->nb[2] + channel * tensor->nb[1] + x * tensor->nb[0];
    }
    if constexpr (Dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    const int64_t flattened_spatial = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * tensor->nb[2] + channel * tensor->nb[1] + flattened_spatial * tensor->nb[0];
}

struct axis_window {
    int32_t first_kernel;
    int32_t valid_count;
    int32_t padded_count;
};

std::vector<axis_window> build_axis_windows(int64_t output_size, int64_t input_size, int64_t kernel_size,
                                            int64_t stride, int64_t dilation, int64_t padding_before,
                                            int64_t padding_after) {
    std::vector<axis_window> windows(static_cast<size_t>(output_size));
    for (int64_t output = 0; output < output_size; ++output) {
        const int64_t base = output * stride - padding_before;
        int32_t first = static_cast<int32_t>(kernel_size);
        int32_t last = -1;
        int32_t padded = 0;
        for (int32_t kernel = 0; kernel < kernel_size; ++kernel) {
            const int64_t coordinate = base + int64_t(kernel) * dilation;
            if (coordinate >= -padding_before && coordinate < input_size + padding_after) ++padded;
            if (coordinate >= 0 && coordinate < input_size) {
                first = std::min(first, kernel);
                last = kernel;
            }
        }
        windows[static_cast<size_t>(output)] = {first, last >= first ? last - first + 1 : 0, padded};
    }
    return windows;
}

template <int Dims, ggml_type Type, ops_pool_mode Mode>
bool execute_pool(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                  const ops_pool_nd_encoded_params& params, const ops_pool_nd_desc& desc) {
    std::vector<axis_window> windows[3];
    for (int axis = 0; axis < 3; ++axis) {
        windows[axis] = build_axis_windows(desc.output_size[axis], desc.input_size[axis], desc.kernel_size[axis],
                                           params.stride[axis], params.dilation[axis], params.padding_before[axis],
                                           params.padding_after[axis]);
    }
    const int threads = backend_thread_count(backend);
#pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
    for (int64_t batch = 0; batch < desc.batch; ++batch) {
        for (int64_t channel = 0; channel < desc.channels; ++channel) {
            for (int64_t output_z = 0; output_z < desc.output_size[2]; ++output_z) {
                const axis_window wz = windows[2][static_cast<size_t>(output_z)];
                const int64_t input_z_base = output_z * params.stride[2] - params.padding_before[2] +
                                             int64_t(wz.first_kernel) * params.dilation[2];
                for (int64_t output_y = 0; output_y < desc.output_size[1]; ++output_y) {
                    const axis_window wy = windows[1][static_cast<size_t>(output_y)];
                    const int64_t input_y_base = output_y * params.stride[1] - params.padding_before[1] +
                                                 int64_t(wy.first_kernel) * params.dilation[1];
                    for (int64_t output_x = 0; output_x < desc.output_size[0]; ++output_x) {
                        const axis_window wx = windows[0][static_cast<size_t>(output_x)];
                        const int64_t input_x_base = output_x * params.stride[0] - params.padding_before[0] +
                                                     int64_t(wx.first_kernel) * params.dilation[0];
                        float accumulator =
                            Mode == ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                        for (int32_t kernel_z = 0; kernel_z < wz.valid_count; ++kernel_z) {
                            const int64_t input_z = input_z_base + int64_t(kernel_z) * params.dilation[2];
                            for (int32_t kernel_y = 0; kernel_y < wy.valid_count; ++kernel_y) {
                                const int64_t input_y = input_y_base + int64_t(kernel_y) * params.dilation[1];
                                for (int32_t kernel_x = 0; kernel_x < wx.valid_count; ++kernel_x) {
                                    const int64_t input_x = input_x_base + int64_t(kernel_x) * params.dilation[0];
                                    const float value =
                                        load_value<Type>(input, spatial_offset<Dims>(input, input_x, input_y, input_z,
                                                                                     channel, batch, desc.input_size));
                                    if constexpr (Mode == ops_pool_mode::maximum) {
                                        accumulator = value > accumulator ? value : accumulator;
                                    } else {
                                        accumulator += value;
                                    }
                                }
                            }
                        }
                        if constexpr (Mode == ops_pool_mode::average) {
                            const int64_t valid_elements = int64_t(wx.valid_count) * wy.valid_count * wz.valid_count;
                            const int64_t padded_elements =
                                int64_t(wx.padded_count) * wy.padded_count * wz.padded_count;
                            const int64_t denominator = desc.count_include_pad ? padded_elements : valid_elements;
                            accumulator = denominator > 0 ? accumulator / static_cast<float>(denominator) : 0.0f;
                        }
                        store_value<Type>(output,
                                          spatial_offset<Dims>(output, output_x, output_y, output_z, channel, batch,
                                                               desc.output_size),
                                          accumulator);
                    }
                }
            }
        }
    }
    return true;
}

template <int Dims, ggml_type Type>
bool dispatch_mode(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                   const ops_pool_nd_encoded_params& params, const ops_pool_nd_desc& desc) {
    if (desc.mode == ops_pool_mode::maximum) {
        return execute_pool<Dims, Type, ops_pool_mode::maximum>(backend, output, input, params, desc);
    }
    return execute_pool<Dims, Type, ops_pool_mode::average>(backend, output, input, params, desc);
}

template <ggml_type Type>
bool dispatch_dims(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                   const ops_pool_nd_encoded_params& params, const ops_pool_nd_desc& desc) {
    if (desc.spatial_dims == 1) return dispatch_mode<1, Type>(backend, output, input, params, desc);
    if (desc.spatial_dims == 2) return dispatch_mode<2, Type>(backend, output, input, params, desc);
    return dispatch_mode<3, Type>(backend, output, input, params, desc);
}

} // namespace

bool ops_cpu_op_pool_nd(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_POOL_1D ? 1 : op == GGML_OP_OPS_VIRT_POOL_2D ? 2 : 3;
    ops_pool_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node};
    ops_pool_nd_desc desc;
    if (!ops_validate_pool_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    switch (node->src[0]->type) {
    case GGML_TYPE_F32:
        return dispatch_dims<GGML_TYPE_F32>(backend, node, node->src[0], params, desc);
    case GGML_TYPE_F16:
        return dispatch_dims<GGML_TYPE_F16>(backend, node, node->src[0], params, desc);
    case GGML_TYPE_BF16:
        return dispatch_dims<GGML_TYPE_BF16>(backend, node, node->src[0], params, desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::cpu
