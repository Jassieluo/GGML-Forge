#pragma once

#include "ggml-common.h"
#include <cuda_fp16.h>

namespace ggml_ops_ext {
namespace cuda {

template <ggml_type Type>
__device__ inline float load_quantized_row_value(const void* data, int64_t row, int64_t column, int64_t row_elements) {
    if constexpr (Type == GGML_TYPE_Q4_0) {
        const int64_t blocks_per_row = row_elements / QK4_0;
        const block_q4_0& block = static_cast<const block_q4_0*>(data)[row * blocks_per_row + column / QK4_0];
        const int index = static_cast<int>(column % QK4_0);
        const uint8_t packed = block.qs[index % (QK4_0 / 2)];
        const int quant = ((index < QK4_0 / 2 ? packed : packed >> 4) & 0x0f) - 8;
        return __half2float(block.d) * quant;
    } else if constexpr (Type == GGML_TYPE_Q8_0) {
        const int64_t blocks_per_row = row_elements / QK8_0;
        const block_q8_0& block = static_cast<const block_q8_0*>(data)[row * blocks_per_row + column / QK8_0];
        return __half2float(block.d) * block.qs[column % QK8_0];
    } else {
        const int64_t blocks_per_row = row_elements / QK_K;
        const block_q4_K& block = static_cast<const block_q4_K*>(data)[row * blocks_per_row + column / QK_K];
        const int index = static_cast<int>(column % QK_K);
        const int group = index / 32;
        uint8_t scale;
        uint8_t minimum;
        if (group < 4) {
            scale = block.scales[group] & 63;
            minimum = block.scales[group + 4] & 63;
        } else {
            scale = (block.scales[group + 4] & 0x0f) | ((block.scales[group - 4] >> 6) << 4);
            minimum = (block.scales[group + 4] >> 4) | ((block.scales[group] >> 6) << 4);
        }
        const int group64 = index / 64;
        const int within64 = index % 64;
        const uint8_t packed = block.qs[group64 * 32 + within64 % 32];
        const int quant = within64 < 32 ? packed & 0x0f : packed >> 4;
        return __half2float(block.dm.x) * scale * quant -
               __half2float(block.dm.y) * minimum;
    }
}

template <ggml_type WeightType, typename T>
__global__ void quantized_conv_1d_direct_kernel(
    const void* weight, const T* input, const void* bias, int bias_type, T* output,
    int64_t input_length, int64_t output_length, int64_t input_channels,
    int64_t output_channels, int64_t kernel, int64_t batch,
    int stride, int padding, int dilation, int groups,
    size_t input_nb0, size_t input_nb1, size_t input_nb2,
    size_t output_nb0, size_t output_nb1, size_t output_nb2
) {
    constexpr int warp_size = 32;
    const int lane = threadIdx.x & (warp_size - 1);
    const int64_t linear = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    const int64_t total = batch * output_channels * output_length;
    if (linear >= total) return;
    const int64_t ow = linear % output_length;
    const int64_t oc = (linear / output_length) % output_channels;
    const int64_t n = linear / (output_length * output_channels);
    const int64_t input_per_group = input_channels / groups;
    const int64_t output_per_group = output_channels / groups;
    const int64_t group = oc / output_per_group;
    float sum = 0.0f;
    for (int64_t kw = 0; kw < kernel; ++kw) {
        const int64_t iw = ow * stride - padding + kw * dilation;
        if (iw < 0 || iw >= input_length) continue;
        for (int64_t local_ic = lane; local_ic < input_per_group; local_ic += warp_size) {
            const int64_t ic = group * input_per_group + local_ic;
            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
            const int64_t row = oc * kernel + kw;
            sum += static_cast<float>(value) *
                   load_quantized_row_value<WeightType>(weight, row, local_ic, input_per_group);
        }
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        sum += bias ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                     : __half2float(static_cast<const half*>(bias)[oc])) : 0.0f;
        *reinterpret_cast<T*>(reinterpret_cast<char*>(output) +
            n * output_nb2 + oc * output_nb1 + ow * output_nb0) = static_cast<T>(sum);
    }
}

template <ggml_type WeightType, typename T>
__global__ void quantized_conv_transpose_1d_direct_kernel(
    const void* weight, const T* input, const void* bias, int bias_type, T* output,
    int64_t input_length, int64_t output_length, int64_t input_channels,
    int64_t output_channels_per_group, int64_t kernel, int64_t batch,
    int stride, int padding, int dilation, int groups,
    size_t input_nb0, size_t input_nb1, size_t input_nb2,
    size_t output_nb0, size_t output_nb1, size_t output_nb2
) {
    const int64_t output_channels = output_channels_per_group * groups;
    constexpr int warp_size = 32;
    const int lane = threadIdx.x & (warp_size - 1);
    const int64_t linear = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / warp_size;
    const int64_t total = batch * output_channels * output_length;
    if (linear >= total) return;
    const int64_t ow = linear % output_length;
    const int64_t oc = (linear / output_length) % output_channels;
    const int64_t n = linear / (output_length * output_channels);
    const int64_t input_per_group = input_channels / groups;
    const int64_t group = oc / output_channels_per_group;
    const int64_t local_oc = oc % output_channels_per_group;
    float sum = 0.0f;
    for (int64_t local_ic = lane; local_ic < input_per_group; local_ic += warp_size) {
        const int64_t ic = group * input_per_group + local_ic;
        for (int64_t kw = 0; kw < kernel; ++kw) {
            const int64_t numerator = ow + padding - kw * dilation;
            if (numerator < 0 || numerator % stride != 0) continue;
            const int64_t iw = numerator / stride;
            if (iw >= input_length) continue;
            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
            const int64_t row = ic * kernel + kw;
            sum += static_cast<float>(value) *
                   load_quantized_row_value<WeightType>(weight, row, local_oc, output_channels_per_group);
        }
    }
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) {
        sum += bias ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                     : __half2float(static_cast<const half*>(bias)[oc])) : 0.0f;
        *reinterpret_cast<T*>(reinterpret_cast<char*>(output) +
            n * output_nb2 + oc * output_nb1 + ow * output_nb0) = static_cast<T>(sum);
    }
}

template <ggml_type WeightType, typename T>
__global__ void quantized_conv_transpose_1d_scalar_kernel(
    const void* weight, const T* input, const void* bias, int bias_type, T* output,
    int64_t input_length, int64_t output_length, int64_t input_channels,
    int64_t output_channels_per_group, int64_t kernel, int64_t batch,
    int stride, int padding, int dilation, int groups,
    size_t input_nb0, size_t input_nb1, size_t input_nb2,
    size_t output_nb0, size_t output_nb1, size_t output_nb2
) {
    const int64_t output_channels = output_channels_per_group * groups;
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * output_channels * output_length;
    if (linear >= total) return;
    const int64_t ow = linear % output_length;
    const int64_t oc = (linear / output_length) % output_channels;
    const int64_t n = linear / (output_length * output_channels);
    const int64_t input_per_group = input_channels / groups;
    const int64_t group = oc / output_channels_per_group;
    const int64_t local_oc = oc % output_channels_per_group;
    float sum = bias ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                       : __half2float(static_cast<const half*>(bias)[oc])) : 0.0f;
    for (int64_t local_ic = 0; local_ic < input_per_group; ++local_ic) {
        const int64_t ic = group * input_per_group + local_ic;
        for (int64_t kw = 0; kw < kernel; ++kw) {
            const int64_t numerator = ow + padding - kw * dilation;
            if (numerator < 0 || numerator % stride != 0) continue;
            const int64_t iw = numerator / stride;
            if (iw >= input_length) continue;
            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
            sum += static_cast<float>(value) * load_quantized_row_value<WeightType>(
                weight, ic * kernel + kw, local_oc, output_channels_per_group);
        }
    }
    *reinterpret_cast<T*>(reinterpret_cast<char*>(output) +
        n * output_nb2 + oc * output_nb1 + ow * output_nb0) = static_cast<T>(sum);
}

} // namespace cuda
} // namespace ggml_ops_ext
