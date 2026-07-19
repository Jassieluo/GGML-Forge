#include "ops/cpu.h"
#include "ops/ops.h"

#include <limits>

namespace ggml_ops_ext::cpu {
namespace {

struct adaptive_window {
    int64_t begin;
    int64_t end;
};

adaptive_window make_adaptive_window(int64_t output_index, int64_t input_size, int64_t output_size) {
    return {
        output_index * input_size / output_size,
        ((output_index + 1) * input_size + output_size - 1) / output_size,
    };
}

size_t spatial_offset(const ggml_tensor* tensor, int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel,
                      int64_t batch, const int64_t spatial_size[3]) {
    if (spatial_dims == 1) {
        return batch * tensor->nb[2] + channel * tensor->nb[1] + x * tensor->nb[0];
    }
    if (spatial_dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    const int64_t spatial_index = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * tensor->nb[2] + channel * tensor->nb[1] + spatial_index * tensor->nb[0];
}

template <ggml_type Type> float load_value(const ggml_tensor* tensor, size_t byte_offset) {
    const char* address = static_cast<const char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) return *reinterpret_cast<const float*>(address);
    if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <ggml_type Type> void store_value(ggml_tensor* tensor, size_t byte_offset, float value) {
    char* address = static_cast<char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32)
        *reinterpret_cast<float*>(address) = value;
    else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(address) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(address) = ggml_fp32_to_bf16(value);
    }
}

template <ggml_type Type>
bool execute_adaptive_pool(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
                           const ops_adaptive_pool_nd_desc& desc) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const int thread_count = backend_thread_count(backend);
#pragma omp parallel for num_threads(thread_count) schedule(static)
    for (int64_t element_index = 0; element_index < total_elements; ++element_index) {
        const int64_t output_spatial = element_index % output_volume;
        const int64_t channel = (element_index / output_volume) % desc.channels;
        const int64_t batch = element_index / (output_volume * desc.channels);
        const int64_t output_x = output_spatial % desc.output_size[0];
        const int64_t output_remainder = output_spatial / desc.output_size[0];
        const int64_t output_y = output_remainder % desc.output_size[1];
        const int64_t output_z = output_remainder / desc.output_size[1];
        const adaptive_window x_window = make_adaptive_window(output_x, desc.input_size[0], desc.output_size[0]);
        const adaptive_window y_window = make_adaptive_window(output_y, desc.input_size[1], desc.output_size[1]);
        const adaptive_window z_window = make_adaptive_window(output_z, desc.input_size[2], desc.output_size[2]);
        float result = desc.mode == ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
        for (int64_t z = z_window.begin; z < z_window.end; ++z) {
            for (int64_t y = y_window.begin; y < y_window.end; ++y) {
                for (int64_t x = x_window.begin; x < x_window.end; ++x) {
                    const float sample = load_value<Type>(
                        input, spatial_offset(input, desc.spatial_dims, x, y, z, channel, batch, desc.input_size));
                    result =
                        desc.mode == ops_pool_mode::maximum ? (sample > result ? sample : result) : result + sample;
                }
            }
        }
        if (desc.mode == ops_pool_mode::average) {
            const int64_t sample_count =
                (x_window.end - x_window.begin) * (y_window.end - y_window.begin) * (z_window.end - z_window.begin);
            result /= static_cast<float>(sample_count);
        }
        store_value<Type>(
            output,
            spatial_offset(output, desc.spatial_dims, output_x, output_y, output_z, channel, batch, desc.output_size),
            result);
    }
    return true;
}

} // namespace

bool ops_cpu_op_adaptive_pool_nd(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) return false;
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                             : op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                       : 3;
    ops_adaptive_pool_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node};
    ops_adaptive_pool_nd_desc desc;
    if (!ops_validate_adaptive_pool_nd_contract(request, spatial_dims, &desc)) return false;
    switch (node->type) {
    case GGML_TYPE_F32:
        return execute_adaptive_pool<GGML_TYPE_F32>(backend, node, sources[0], desc);
    case GGML_TYPE_F16:
        return execute_adaptive_pool<GGML_TYPE_F16>(backend, node, sources[0], desc);
    case GGML_TYPE_BF16:
        return execute_adaptive_pool<GGML_TYPE_BF16>(backend, node, sources[0], desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::cpu
