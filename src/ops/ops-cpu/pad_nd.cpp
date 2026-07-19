#include "ops/cpu.h"
#include "ops/ops.h"

#include <cstring>

namespace ggml_ops_ext::cpu {
namespace {

int64_t map_coordinate(int64_t coordinate, int64_t size, ops_pad_mode mode, bool& valid) {
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

size_t spatial_offset(const ggml_tensor* tensor, int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel,
                      int64_t batch, const int64_t spatial_size[3]) {
    if (spatial_dims == 1) {
        return batch * tensor->nb[2] + channel * tensor->nb[1] + x * tensor->nb[0];
    }
    if (spatial_dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    return batch * tensor->nb[2] + channel * tensor->nb[1] +
           (x + spatial_size[0] * (y + spatial_size[1] * z)) * tensor->nb[0];
}

template <ggml_type Type> float load(const ggml_tensor* tensor, size_t byte_offset) {
    const char* address = static_cast<const char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(address);
    }
    if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(address));
    }
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(address));
}

template <ggml_type Type> void store(ggml_tensor* tensor, size_t byte_offset, float value) {
    char* address = static_cast<char*>(tensor->data) + byte_offset;
    if constexpr (Type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(address) = value;
    } else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(address) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(address) = ggml_fp32_to_bf16(value);
    }
}

template <ggml_type Type>
bool execute(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
             const ops_pad_nd_encoded_params& params, const ops_pad_nd_desc& desc) {
    bool zero_padding = true;
    for (int axis = 0; axis < 3; ++axis) {
        zero_padding &= params.padding_before[axis] == 0 && params.padding_after[axis] == 0;
    }
    if (zero_padding && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        std::memcpy(output->data, input->data, ggml_nbytes(input));
        return true;
    }
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
    const int threads = backend_thread_count(backend);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < total_elements; ++index) {
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
        const float value = valid ? load<Type>(input, spatial_offset(input, desc.spatial_dims, input_x, input_y,
                                                                     input_z, channel, batch, desc.input_size))
                                  : desc.value;
        store<Type>(
            output,
            spatial_offset(output, desc.spatial_dims, output_x, output_y, output_z, channel, batch, desc.output_size),
            value);
    }
    return true;
}

} // namespace

bool ops_cpu_op_pad_nd(ggml_backend_t backend, ggml_tensor* node) {
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
    if (node->type == GGML_TYPE_F32) {
        return execute<GGML_TYPE_F32>(backend, node, node->src[0], params, desc);
    }
    if (node->type == GGML_TYPE_F16) {
        return execute<GGML_TYPE_F16>(backend, node, node->src[0], params, desc);
    }
    if (node->type == GGML_TYPE_BF16) {
        return execute<GGML_TYPE_BF16>(backend, node, node->src[0], params, desc);
    }
    return false;
}

} // namespace ggml_ops_ext::cpu
