#include "common.hpp"
#include "ops/ops.h"
#include "ops_sycl.h"

namespace ggml_ops_ext::sycl {

template <typename T> class PadNDKernel;

inline int64_t map_coordinate(int64_t coordinate, int64_t size, ops_pad_mode mode, bool& valid) {
    if (coordinate >= 0 && coordinate < size) {
        return coordinate;
    }
    if (mode == ops_pad_mode::constant) {
        valid = false;
        return 0;
    }
    if (mode == ops_pad_mode::replicate) {
        return coordinate < 0 ? 0 : size - 1;
    }
    if (mode == ops_pad_mode::circular) {
        return (coordinate % size + size) % size;
    }
    return coordinate < 0 ? -coordinate : 2 * size - 2 - coordinate;
}

struct tensor_strides {
    size_t x;
    size_t y;
    size_t channel_or_batch;
    size_t batch;
};

inline size_t spatial_offset(int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
                             const int64_t spatial_size[3], tensor_strides strides) {
    if (spatial_dims == 1) {
        return batch * strides.channel_or_batch + channel * strides.y + x * strides.x;
    }
    if (spatial_dims == 2) {
        return batch * strides.batch + channel * strides.channel_or_batch + y * strides.y + x * strides.x;
    }
    return batch * strides.channel_or_batch + channel * strides.y +
           (x + spatial_size[0] * (y + spatial_size[1] * z)) * strides.x;
}

template <typename T>
bool launch_pad_sycl(::sycl::queue* queue, ggml_tensor* input, ggml_tensor* output,
                     const ops_pad_nd_encoded_params& params, const ops_pad_nd_desc& desc) {
    bool zero_padding = true;
    for (int axis = 0; axis < 3; ++axis) {
        zero_padding &= params.padding_before[axis] == 0 && params.padding_after[axis] == 0;
    }
    if (zero_padding && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        queue->memcpy(output->data, input->data, ggml_nbytes(input));
        return true;
    }
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const size_t local = 256;
    const size_t global = static_cast<size_t>((total_elements + local - 1) / local) * local;
    const T* source = static_cast<const T*>(input->data);
    T* destination = static_cast<T*>(output->data);
    const tensor_strides input_strides = {input->nb[0], input->nb[1], input->nb[2], input->nb[3]};
    const tensor_strides output_strides = {output->nb[0], output->nb[1], output->nb[2], output->nb[3]};
    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<PadNDKernel<T>>(
            ::sycl::nd_range<1>(::sycl::range<1>(global), ::sycl::range<1>(local)), [=](::sycl::nd_item<1> item) {
                const int64_t index = item.get_global_linear_id();
                if (index >= total_elements) {
                    return;
                }
                const int64_t output_spatial = index % output_volume;
                const int64_t channel = (index / output_volume) % desc.channels;
                const int64_t batch = index / (output_volume * desc.channels);
                const int64_t output_x = output_spatial % desc.output_size[0];
                const int64_t output_remainder = output_spatial / desc.output_size[0];
                const int64_t output_y = output_remainder % desc.output_size[1];
                const int64_t output_z = output_remainder / desc.output_size[1];
                bool valid = true;
                const int64_t input_x =
                    map_coordinate(output_x - params.padding_before[0], desc.input_size[0], desc.mode, valid);
                const int64_t input_y =
                    map_coordinate(output_y - params.padding_before[1], desc.input_size[1], desc.mode, valid);
                const int64_t input_z =
                    map_coordinate(output_z - params.padding_before[2], desc.input_size[2], desc.mode, valid);
                T value = static_cast<T>(desc.value);
                if (valid) {
                    const char* address = reinterpret_cast<const char*>(source) +
                                          spatial_offset(desc.spatial_dims, input_x, input_y, input_z, channel, batch,
                                                         desc.input_size, input_strides);
                    value = *reinterpret_cast<const T*>(address);
                }
                char* address = reinterpret_cast<char*>(destination) +
                                spatial_offset(desc.spatial_dims, output_x, output_y, output_z, channel, batch,
                                               desc.output_size, output_strides);
                *reinterpret_cast<T*>(address) = value;
            });
    });
    return true;
}

bool ggml_sycl_op_pad_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0]) {
        return false;
    }
    const int op = static_cast<int>(node->op);
    const int spatial_dims = op == GGML_OP_OPS_VIRT_PAD_1D ? 1 : op == GGML_OP_OPS_VIRT_PAD_2D ? 2 : 3;
    ops_pad_nd_encoded_params params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    ggml_tensor* sources[] = {node->src[0]};
    ops_request request = {ggml_backend_get_device(backend), op, sources, 1, &params, sizeof(params), node};
    ops_pad_nd_desc desc;
    if (!ops_validate_pad_nd_contract(request, spatial_dims, &desc)) {
        return false;
    }
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) {
        return false;
    }
    if (node->type == GGML_TYPE_F32) {
        return launch_pad_sycl<float>(queue, node->src[0], node, params, desc);
    }
    if (node->type == GGML_TYPE_F16) {
        return launch_pad_sycl<::sycl::half>(queue, node->src[0], node, params, desc);
    }
    return false;
}

} // namespace ggml_ops_ext::sycl
