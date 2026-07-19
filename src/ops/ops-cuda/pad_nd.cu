#include "ops_cuda_common.cuh"

#include <cuda_bf16.h>

namespace ggml_ops_ext::cuda {
namespace {

template <typename T> __device__ T pad_from_float(float value) { return static_cast<T>(value); }
template <> __device__ half pad_from_float<half>(float value) { return __float2half(value); }
template <> __device__ __nv_bfloat16 pad_from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

__device__ int64_t pad_map(int64_t coordinate, int64_t size, ops_pad_mode mode, bool& valid) {
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

__device__ size_t spatial_offset(int spatial_dims, int64_t x, int64_t y, int64_t z, int64_t channel, int64_t batch,
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
__global__ void pad_nd_kernel(const T* input, T* output, ops_pad_nd_encoded_params params, ops_pad_nd_desc desc,
                              tensor_strides input_strides, tensor_strides output_strides) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_elements = desc.batch * desc.channels * output_volume;
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
    const int64_t input_x = pad_map(output_x - params.padding_before[0], desc.input_size[0], desc.mode, valid);
    const int64_t input_y = pad_map(output_y - params.padding_before[1], desc.input_size[1], desc.mode, valid);
    const int64_t input_z = pad_map(output_z - params.padding_before[2], desc.input_size[2], desc.mode, valid);
    T value = pad_from_float<T>(desc.value);
    if (valid) {
        const char* address =
            reinterpret_cast<const char*>(input) + spatial_offset(desc.spatial_dims, input_x, input_y, input_z, channel,
                                                                  batch, desc.input_size, input_strides);
        value = *reinterpret_cast<const T*>(address);
    }
    char* address = reinterpret_cast<char*>(output) + spatial_offset(desc.spatial_dims, output_x, output_y, output_z,
                                                                     channel, batch, desc.output_size, output_strides);
    *reinterpret_cast<T*>(address) = value;
}

template <typename T>
bool launch_pad(cudaStream_t stream, ggml_tensor* input, ggml_tensor* output, const ops_pad_nd_encoded_params& params,
                const ops_pad_nd_desc& desc) {
    bool zero_padding = true;
    for (int axis = 0; axis < 3; ++axis) {
        zero_padding &= params.padding_before[axis] == 0 && params.padding_after[axis] == 0;
    }
    if (zero_padding && ggml_is_contiguous(input) && ggml_is_contiguous(output)) {
        return cudaMemcpyAsync(output->data, input->data, ggml_nbytes(input), cudaMemcpyDeviceToDevice, stream) ==
               cudaSuccess;
    }
    const int64_t total_elements =
        desc.batch * desc.channels * desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const tensor_strides input_strides = {input->nb[0], input->nb[1], input->nb[2], input->nb[3]};
    const tensor_strides output_strides = {output->nb[0], output->nb[1], output->nb[2], output->nb[3]};
    pad_nd_kernel<T><<<static_cast<unsigned>((total_elements + 255) / 256), 256, 0, stream>>>(
        static_cast<const T*>(input->data), static_cast<T*>(output->data), params, desc, input_strides, output_strides);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool ggml_cuda_op_pad_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
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
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) {
        return false;
    }
    if (node->type == GGML_TYPE_F32) {
        return launch_pad<float>(stream, node->src[0], node, params, desc);
    }
    if (node->type == GGML_TYPE_F16) {
        return launch_pad<half>(stream, node->src[0], node, params, desc);
    }
    if (node->type == GGML_TYPE_BF16) {
        return launch_pad<__nv_bfloat16>(stream, node->src[0], node, params, desc);
    }
    return false;
}

} // namespace ggml_ops_ext::cuda
