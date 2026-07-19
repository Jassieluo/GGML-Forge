#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <limits>

namespace ggml_ops_ext::sycl {
namespace {

struct tensor_strides {
    size_t values[4];
};

struct adaptive_window {
    int64_t begin;
    int64_t end;
};

template <typename T> class AdaptivePoolNDKernel;

template <typename T, ops_pool_mode Mode> class AdaptiveGlobalPoolNDKernel;

template <typename T, ops_pool_mode Mode>
void launch_global_adaptive_pool(::sycl::queue* queue, const T* source, T* destination, int64_t input_volume,
                                 int64_t batch_channels) {
    constexpr size_t local_size = 256;
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> partial(::sycl::range<1>(local_size), handler);
        handler.parallel_for<AdaptiveGlobalPoolNDKernel<T, Mode>>(
            ::sycl::nd_range<1>(::sycl::range<1>(batch_channels * local_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int lane = static_cast<int>(item.get_local_id(0));
                const int64_t batch_channel = item.get_group(0);
                const T* row = source + batch_channel * input_volume;
                float value = Mode == ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                for (int64_t index = lane; index < input_volume; index += local_size) {
                    const float sample = static_cast<float>(row[index]);
                    if constexpr (Mode == ops_pool_mode::maximum)
                        value = ::sycl::fmax(value, sample);
                    else
                        value += sample;
                }
                partial[lane] = value;
                item.barrier(::sycl::access::fence_space::local_space);
                for (int offset = local_size / 2; offset > 0; offset >>= 1) {
                    if (lane < offset) {
                        if constexpr (Mode == ops_pool_mode::maximum) {
                            partial[lane] = ::sycl::fmax(partial[lane], partial[lane + offset]);
                        } else {
                            partial[lane] += partial[lane + offset];
                        }
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }
                if (lane == 0) {
                    float result = partial[0];
                    if constexpr (Mode == ops_pool_mode::average) result /= static_cast<float>(input_volume);
                    destination[batch_channel] = static_cast<T>(result);
                }
            });
    });
}

inline adaptive_window make_adaptive_window(int64_t output_index, int64_t input_size, int64_t output_size) {
    return {
        output_index * input_size / output_size,
        ((output_index + 1) * input_size + output_size - 1) / output_size,
    };
}

inline size_t spatial_offset(int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                             const int64_t spatial_size[3], tensor_strides strides) {
    if (spatial_dims == 1) {
        return batch * strides.values[2] + channel * strides.values[1] + x * strides.values[0];
    }
    if (spatial_dims == 2) {
        return batch * strides.values[3] + channel * strides.values[2] + y * strides.values[1] + x * strides.values[0];
    }
    const int64_t spatial_index = x + spatial_size[0] * (y + spatial_size[1] * z);
    return batch * strides.values[2] + channel * strides.values[1] + spatial_index * strides.values[0];
}

template <typename T>
bool launch_adaptive_pool(::sycl::queue* queue, ggml_tensor* input, ggml_tensor* output,
                          const ops_adaptive_pool_nd_desc& desc) {
    const int64_t input_volume = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    if (output_volume == 1 && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        const int64_t batch_channels = desc.batch * desc.channels;
        const T* source = static_cast<const T*>(input->data);
        T* destination = static_cast<T*>(output->data);
        if (desc.mode == ops_pool_mode::maximum) {
            launch_global_adaptive_pool<T, ops_pool_mode::maximum>(queue, source, destination, input_volume,
                                                                   batch_channels);
        } else {
            launch_global_adaptive_pool<T, ops_pool_mode::average>(queue, source, destination, input_volume,
                                                                   batch_channels);
        }
        return true;
    }
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const size_t local_size = 256;
    const size_t global_size = static_cast<size_t>((total_elements + local_size - 1) / local_size) * local_size;
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);
    const tensor_strides input_strides = {{
        input->nb[0],
        input->nb[1],
        input->nb[2],
        input->nb[3],
    }};
    const tensor_strides output_strides = {{
        output->nb[0],
        output->nb[1],
        output->nb[2],
        output->nb[3],
    }};

    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<AdaptivePoolNDKernel<T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int64_t element_index = item.get_global_linear_id();
                if (element_index >= total_elements) {
                    return;
                }
                const int64_t output_spatial = element_index % output_volume;
                const int64_t channel = (element_index / output_volume) % desc.channels;
                const int64_t batch = element_index / (output_volume * desc.channels);
                const int64_t output_x = output_spatial % desc.output_size[0];
                const int64_t output_remainder = output_spatial / desc.output_size[0];
                const int64_t output_y = output_remainder % desc.output_size[1];
                const int64_t output_z = output_remainder / desc.output_size[1];
                const adaptive_window x_window =
                    make_adaptive_window(output_x, desc.input_size[0], desc.output_size[0]);
                const adaptive_window y_window =
                    make_adaptive_window(output_y, desc.input_size[1], desc.output_size[1]);
                const adaptive_window z_window =
                    make_adaptive_window(output_z, desc.input_size[2], desc.output_size[2]);

                float result = desc.mode == ops_pool_mode::maximum ? -std::numeric_limits<float>::infinity() : 0.0f;
                for (int64_t input_z = z_window.begin; input_z < z_window.end; ++input_z) {
                    for (int64_t input_y = y_window.begin; input_y < y_window.end; ++input_y) {
                        for (int64_t input_x = x_window.begin; input_x < x_window.end; ++input_x) {
                            const char* address = reinterpret_cast<const char*>(source) +
                                                  spatial_offset(desc.spatial_dims, input_x, input_y, input_z, channel,
                                                                 batch, desc.input_size, input_strides);
                            const float sample = static_cast<float>(*reinterpret_cast<const T*>(address));
                            result =
                                desc.mode == ops_pool_mode::maximum ? ::sycl::fmax(result, sample) : result + sample;
                        }
                    }
                }
                if (desc.mode == ops_pool_mode::average) {
                    const int64_t sample_count = (x_window.end - x_window.begin) * (y_window.end - y_window.begin) *
                                                 (z_window.end - z_window.begin);
                    result /= static_cast<float>(sample_count);
                }
                char* output_address = reinterpret_cast<char*>(destination) +
                                       spatial_offset(desc.spatial_dims, output_x, output_y, output_z, channel, batch,
                                                      desc.output_size, output_strides);
                *reinterpret_cast<T*>(output_address) = static_cast<T>(result);
            });
    });
    return true;
}

} // namespace

bool ggml_sycl_op_adaptive_pool_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_1D   ? 1
                             : op == GGML_OP_OPS_VIRT_ADAPTIVE_POOL_2D ? 2
                                                                       : 3;
    ops_adaptive_pool_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {
        ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node,
    };
    ops_adaptive_pool_nd_desc desc;
    if (!ops_validate_adaptive_pool_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }
    switch (node->type) {
    case GGML_TYPE_F32:
        return launch_adaptive_pool<float>(queue, sources[0], node, desc);
    case GGML_TYPE_F16:
        return launch_adaptive_pool<::sycl::half>(queue, sources[0], node, desc);
    default:
        return false;
    }
}

} // namespace ggml_ops_ext::sycl
