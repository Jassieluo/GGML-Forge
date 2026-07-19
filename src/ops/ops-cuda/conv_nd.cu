#include "ops_cuda_common.cuh"
#include "quantized_conv.cuh"

#include <cuda_bf16.h>
#include <mma.h>
#include <atomic>
#include <limits>
#include <type_traits>

namespace ggml_ops_ext::cuda {

static bool conv_nd_tensor_cores_available(int device) {
    static std::atomic<int> cached[64]{};
    if (device < 0 || device >= 64) return false;
    int value = cached[device].load(std::memory_order_acquire);
    if (value == 0) {
        int major = 0;
        const bool available = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess && major >= 7;
        value = available ? 2 : 1;
        cached[device].store(value, std::memory_order_release);
    }
    return value == 2;
}

template <typename T>
__device__ float conv_nd_to_float(T value) { return static_cast<float>(value); }
template <>
__device__ float conv_nd_to_float<half>(half value) { return __half2float(value); }
template <>
__device__ float conv_nd_to_float<__nv_bfloat16>(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
__device__ T conv_nd_from_float(float value) { return static_cast<T>(value); }
template <>
__device__ half conv_nd_from_float<half>(float value) { return __float2half(value); }
template <>
__device__ __nv_bfloat16 conv_nd_from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <int QuantType, typename WeightT>
__device__ float conv_nd_weight(
    const void* weight, int64_t kernel, int64_t outer, int64_t inner,
    int64_t row_elements, int64_t kernel_volume, ops_weight_layout layout,
    size_t nb0, size_t nb1, size_t nb2
) {
    if constexpr (QuantType >= 0) {
        const int64_t row = layout == ops_weight_layout::flattened_rows ? outer : outer * kernel_volume + kernel;
        const int64_t column = layout == ops_weight_layout::flattened_rows ? inner * kernel_volume + kernel : inner;
        return load_quantized_row_value<static_cast<ggml_type>(QuantType)>(
            weight, row, column, row_elements);
    } else {
        const char* address = layout == ops_weight_layout::flattened_rows
            ? static_cast<const char*>(weight) + outer * nb1 + (inner * kernel_volume + kernel) * nb0
            : static_cast<const char*>(weight) + outer * nb2 + kernel * nb1 + inner * nb0;
        return conv_nd_to_float(*reinterpret_cast<const WeightT*>(address));
    }
}

template <int QuantType, typename WeightT, typename ValueT, bool Transposed>
__global__ void conv_nd_kernel(
    const void* weight, const ValueT* input, const void* bias, int bias_type, ValueT* output,
    ops_conv_nd_encoded_params params, ops_conv_nd_desc desc,
    size_t weight_nb0, size_t weight_nb1, size_t weight_nb2,
    size_t input_nb0, size_t input_nb1, size_t input_nb2, size_t input_nb3,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3
) {
    __shared__ float weight_tile[256];
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t spatial = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t oc = blockIdx.y;
    const int64_t batch = blockIdx.z;
    const bool active = spatial < output_volume;
    const int64_t ox = spatial % desc.output_size[0];
    const int64_t rem = spatial / desc.output_size[0];
    const int64_t oy = rem % desc.output_size[1];
    const int64_t oz = rem / desc.output_size[1];
    const int64_t group = oc / desc.output_channels_per_group;
    const int64_t local_oc = oc % desc.output_channels_per_group;
    float sum = 0.0f;
    if (bias) {
        sum = bias_type == 0 ? static_cast<const float*>(bias)[oc]
            : bias_type == 1 ? __half2float(static_cast<const half*>(bias)[oc])
                             : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[oc]);
    }
    if constexpr (Transposed) {
        const bool phase_path = params.dilation[0] == 1 && params.dilation[1] == 1 &&
            params.dilation[2] == 1 &&
            (params.stride[0] > 1 || params.stride[1] > 1 || params.stride[2] > 1);
        if (phase_path) {
            if (active) {
                const int64_t kx_start = (ox + params.padding_before[0]) % params.stride[0];
                const int64_t ky_start = (oy + params.padding_before[1]) % params.stride[1];
                const int64_t kz_start = (oz + params.padding_before[2]) % params.stride[2];
                for (int64_t kz = kz_start; kz < desc.kernel_size[2]; kz += params.stride[2]) {
                    const int64_t sz = oz + params.padding_before[2] - kz;
                    if (sz < 0) continue;
                    const int64_t iz = sz / params.stride[2];
                    if (iz >= desc.input_size[2]) continue;
                    for (int64_t ky = ky_start; ky < desc.kernel_size[1]; ky += params.stride[1]) {
                        const int64_t sy = oy + params.padding_before[1] - ky;
                        if (sy < 0) continue;
                        const int64_t iy = sy / params.stride[1];
                        if (iy >= desc.input_size[1]) continue;
                        for (int64_t kx = kx_start; kx < desc.kernel_size[0]; kx += params.stride[0]) {
                            const int64_t sx = ox + params.padding_before[0] - kx;
                            if (sx < 0) continue;
                            const int64_t ix = sx / params.stride[0];
                            if (ix >= desc.input_size[0]) continue;
                            const int64_t kernel = kx + desc.kernel_size[0] * (ky + desc.kernel_size[1] * kz);
                            const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                            for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                                const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                                    ? desc.output_channels_per_group * desc.kernel_volume
                                    : desc.output_channels_per_group;
                                const char* input_address = desc.spatial_dims == 2
                                    ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                                    : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                                sum += conv_nd_to_float(*reinterpret_cast<const ValueT*>(input_address)) *
                                    conv_nd_weight<QuantType, WeightT>(weight, kernel, ic, local_oc, row_elements,
                                        desc.kernel_volume, desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                            }
                        }
                    }
                }
                const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
                char* output_address = desc.spatial_dims == 2
                    ? reinterpret_cast<char*>(output) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
                    : reinterpret_cast<char*>(output) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
                *reinterpret_cast<ValueT*>(output_address) = conv_nd_from_float<ValueT>(sum);
            }
            return;
        }
    }
    for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
        const int64_t kx = kernel % desc.kernel_size[0];
        const int64_t krem = kernel / desc.kernel_size[0];
        const int64_t ky = krem % desc.kernel_size[1];
        const int64_t kz = krem / desc.kernel_size[1];
        int64_t ix = -1, iy = -1, iz = -1;
        bool input_valid = active;
        if constexpr (!Transposed) {
            ix = ox * params.stride[0] - params.padding_before[0] + kx * params.dilation[0];
            iy = oy * params.stride[1] - params.padding_before[1] + ky * params.dilation[1];
            iz = oz * params.stride[2] - params.padding_before[2] + kz * params.dilation[2];
        } else {
            const int64_t sx = ox + params.padding_before[0] - kx * params.dilation[0];
            const int64_t sy = oy + params.padding_before[1] - ky * params.dilation[1];
            const int64_t sz = oz + params.padding_before[2] - kz * params.dilation[2];
            input_valid = input_valid && sx >= 0 && sy >= 0 && sz >= 0 &&
                sx % params.stride[0] == 0 && sy % params.stride[1] == 0 && sz % params.stride[2] == 0;
            if (input_valid) {
                ix = sx / params.stride[0];
                iy = sy / params.stride[1];
                iz = sz / params.stride[2];
            }
        }
        input_valid = input_valid && ix >= 0 && ix < desc.input_size[0] &&
            iy >= 0 && iy < desc.input_size[1] && iz >= 0 && iz < desc.input_size[2];
        if (desc.input_channels_per_group <= 4) {
            if (input_valid) {
                const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
                    const int64_t outer = Transposed ? ic : oc;
                    const int64_t inner = Transposed ? local_oc : local_ic;
                    const int64_t row_channels = Transposed ? desc.output_channels_per_group : desc.input_channels_per_group;
                    const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                        ? row_channels * desc.kernel_volume : row_channels;
                    const char* input_address = desc.spatial_dims == 2
                        ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                        : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                    sum += conv_nd_to_float(*reinterpret_cast<const ValueT*>(input_address)) *
                        conv_nd_weight<QuantType, WeightT>(weight, kernel, outer, inner, row_elements,
                            desc.kernel_volume, desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                }
            }
            continue;
        }
        for (int64_t channel_base = 0; channel_base < desc.input_channels_per_group; channel_base += blockDim.x) {
            const int64_t loaded_ic = channel_base + threadIdx.x;
            if (loaded_ic < desc.input_channels_per_group) {
                const int64_t loaded_channel = group * desc.input_channels_per_group + loaded_ic;
                const int64_t outer = Transposed ? loaded_channel : oc;
                const int64_t inner = Transposed ? local_oc : loaded_ic;
                const int64_t row_channels = Transposed ? desc.output_channels_per_group : desc.input_channels_per_group;
                const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                    ? row_channels * desc.kernel_volume : row_channels;
                weight_tile[threadIdx.x] = conv_nd_weight<QuantType, WeightT>(
                    weight, kernel, outer, inner, row_elements, desc.kernel_volume,
                    desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
            }
            __syncthreads();
            if (input_valid) {
                const int64_t remaining = desc.input_channels_per_group - channel_base;
                const int64_t tile_count = remaining < blockDim.x ? remaining : blockDim.x;
                const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                for (int64_t tile_ic = 0; tile_ic < tile_count; ++tile_ic) {
                    const int64_t ic = group * desc.input_channels_per_group + channel_base + tile_ic;
                    const char* input_address = desc.spatial_dims == 2
                        ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                        : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                    sum += conv_nd_to_float(*reinterpret_cast<const ValueT*>(input_address)) * weight_tile[tile_ic];
                }
            }
            __syncthreads();
        }
    }
    if (!active) return;
    const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
    char* output_address = desc.spatial_dims == 2
        ? reinterpret_cast<char*>(output) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
        : reinterpret_cast<char*>(output) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
    *reinterpret_cast<ValueT*>(output_address) = conv_nd_from_float<ValueT>(sum);
}

template <int QuantType, typename WeightT, typename ValueT>
__global__ void conv_nd_implicit_gemm_kernel(
    const void* weight, const ValueT* input, const void* bias, int bias_type, ValueT* output,
    ops_conv_nd_encoded_params params, ops_conv_nd_desc desc,
    size_t weight_nb0, size_t weight_nb1, size_t weight_nb2,
    size_t input_nb0, size_t input_nb1, size_t input_nb2, size_t input_nb3,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3
) {
    constexpr int tile = 16;
    __shared__ float input_tile[tile][tile + 1];
    __shared__ float weight_tile[tile][tile + 1];
    const int lane = threadIdx.x;
    const int spatial_lane = lane & (tile - 1);
    const int output_channel_lane = lane / tile;
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
    const int64_t channel_tiles = (desc.output_channels_per_group + tile - 1) / tile;
    const int64_t group = blockIdx.y / channel_tiles;
    const int64_t local_oc = (blockIdx.y % channel_tiles) * tile + output_channel_lane;
    const int64_t oc = group * desc.output_channels_per_group + local_oc;
    const int64_t batch = blockIdx.z;
    const bool spatial_active = spatial < output_volume;
    const bool channel_active = local_oc < desc.output_channels_per_group;
    const int64_t ox = spatial % desc.output_size[0];
    const int64_t rem = spatial / desc.output_size[0];
    const int64_t oy = rem % desc.output_size[1];
    const int64_t oz = rem / desc.output_size[1];
    float sum = 0.0f;
    if (spatial_active && channel_active && bias) {
        sum = bias_type == 0 ? static_cast<const float*>(bias)[oc]
            : bias_type == 1 ? __half2float(static_cast<const half*>(bias)[oc])
                             : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[oc]);
    }
    const int64_t reduction = desc.kernel_volume * desc.input_channels_per_group;
    const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
        ? reduction : desc.input_channels_per_group;
    const bool pointwise = desc.kernel_volume == 1 && params.stride[0] == 1 &&
        params.stride[1] == 1 && params.stride[2] == 1 &&
        params.padding_before[0] == 0 && params.padding_before[1] == 0 && params.padding_before[2] == 0;
    const bool common_3x3 = desc.kernel_size[0] == 3 && desc.kernel_size[1] == 3 &&
        (desc.spatial_dims == 2 || desc.kernel_size[2] == 3) &&
        desc.input_channels_per_group % tile == 0 &&
        params.stride[0] == 1 && params.stride[1] == 1 &&
        params.dilation[0] == 1 && params.dilation[1] == 1 &&
        (desc.spatial_dims == 2 || (params.stride[2] == 1 && params.dilation[2] == 1));
    for (int64_t reduction_base = 0; reduction_base < reduction; reduction_base += tile) {
        const int64_t tile_kernel = pointwise ? 0
            : (common_3x3 ? reduction_base / desc.input_channels_per_group : -1);
        const int64_t tile_channel_base = (pointwise || common_3x3)
            ? reduction_base - tile_kernel * desc.input_channels_per_group : -1;
        const int64_t input_reduction = reduction_base + output_channel_lane;
        float input_value = 0.0f;
        if (spatial_active && input_reduction < reduction) {
            const int64_t kernel = (pointwise || common_3x3)
                ? tile_kernel : input_reduction / desc.input_channels_per_group;
            const int64_t local_ic = (pointwise || common_3x3)
                ? tile_channel_base + output_channel_lane : input_reduction % desc.input_channels_per_group;
            const int64_t kx = pointwise ? 0 : kernel % desc.kernel_size[0];
            const int64_t krem = pointwise ? 0 : kernel / desc.kernel_size[0];
            const int64_t ky = pointwise ? 0 : krem % desc.kernel_size[1];
            const int64_t kz = pointwise ? 0 : krem / desc.kernel_size[1];
            const int64_t ix = pointwise ? ox : ox * params.stride[0] - params.padding_before[0] + kx * params.dilation[0];
            const int64_t iy = pointwise ? oy : oy * params.stride[1] - params.padding_before[1] + ky * params.dilation[1];
            const int64_t iz = pointwise ? oz : oz * params.stride[2] - params.padding_before[2] + kz * params.dilation[2];
            if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1] &&
                iz >= 0 && iz < desc.input_size[2]) {
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                const char* input_address = desc.spatial_dims == 2
                    ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                    : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                input_value = conv_nd_to_float(*reinterpret_cast<const ValueT*>(input_address));
            }
        }
        input_tile[spatial_lane][output_channel_lane] = input_value;

        const int64_t weight_reduction = reduction_base + spatial_lane;
        float weight_value = 0.0f;
        if (channel_active && weight_reduction < reduction) {
            const int64_t kernel = (pointwise || common_3x3)
                ? tile_kernel : weight_reduction / desc.input_channels_per_group;
            const int64_t local_ic = (pointwise || common_3x3)
                ? tile_channel_base + spatial_lane : weight_reduction % desc.input_channels_per_group;
            weight_value = conv_nd_weight<QuantType, WeightT>(
                weight, kernel, oc, local_ic, row_elements, desc.kernel_volume,
                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
        }
        weight_tile[output_channel_lane][spatial_lane] = weight_value;
        __syncthreads();
        if (spatial_active && channel_active) {
            const int64_t remaining = reduction - reduction_base;
            const int count = remaining < tile ? static_cast<int>(remaining) : tile;
            #pragma unroll
            for (int k = 0; k < tile; ++k) {
                if (k < count) sum += input_tile[spatial_lane][k] * weight_tile[output_channel_lane][k];
            }
        }
        __syncthreads();
    }
    if (!spatial_active || !channel_active) return;
    const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
    char* output_address = desc.spatial_dims == 2
        ? reinterpret_cast<char*>(output) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
        : reinterpret_cast<char*>(output) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
    *reinterpret_cast<ValueT*>(output_address) = conv_nd_from_float<ValueT>(sum);
}

template <int QuantType, typename WeightT, typename ValueT>
__global__ void conv_transpose_2d_phase_gemm_kernel(
    const void* weight, const ValueT* input, const void* bias, int bias_type, ValueT* output,
    ops_conv_nd_encoded_params params, ops_conv_nd_desc desc,
    size_t weight_nb0, size_t weight_nb1, size_t weight_nb2,
    size_t input_nb0, size_t input_nb1, size_t input_nb2, size_t input_nb3,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3
) {
    constexpr int tile = 16;
    __shared__ float input_tile[tile][tile + 1];
    __shared__ float weight_tile[tile][tile + 1];
    const int lane = threadIdx.x;
    const int spatial_lane = lane & (tile - 1);
    const int output_channel_lane = lane / tile;
    const int64_t channel_tiles = (desc.output_channels_per_group + tile - 1) / tile;
    const int64_t phase_count = static_cast<int64_t>(params.stride[0]) * params.stride[1];
    const int64_t phase_and_group = blockIdx.y / channel_tiles;
    const int64_t channel_tile = blockIdx.y % channel_tiles;
    const int64_t phase = phase_and_group % phase_count;
    const int64_t group = phase_and_group / phase_count;
    const int64_t phase_x = phase % params.stride[0];
    const int64_t phase_y = phase / params.stride[0];
    const int64_t padding_x_mod = params.padding_before[0] % params.stride[0];
    const int64_t padding_y_mod = params.padding_before[1] % params.stride[1];
    const int64_t first_x = (phase_x - padding_x_mod + params.stride[0]) % params.stride[0];
    const int64_t first_y = (phase_y - padding_y_mod + params.stride[1]) % params.stride[1];
    const int64_t phase_width = first_x < desc.output_size[0]
        ? (desc.output_size[0] - 1 - first_x) / params.stride[0] + 1 : 0;
    const int64_t phase_height = first_y < desc.output_size[1]
        ? (desc.output_size[1] - 1 - first_y) / params.stride[1] + 1 : 0;
    const int64_t phase_volume = phase_width * phase_height;
    const int64_t packed_spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
    const bool spatial_active = packed_spatial < phase_volume;
    const int64_t packed_x = phase_width > 0 ? packed_spatial % phase_width : 0;
    const int64_t packed_y = phase_width > 0 ? packed_spatial / phase_width : 0;
    const int64_t ox = first_x + packed_x * params.stride[0];
    const int64_t oy = first_y + packed_y * params.stride[1];
    const int64_t local_oc = channel_tile * tile + output_channel_lane;
    const int64_t oc = group * desc.output_channels_per_group + local_oc;
    const bool channel_active = local_oc < desc.output_channels_per_group;
    const int64_t batch = blockIdx.z;
    float sum = 0.0f;
    if (spatial_active && channel_active && bias) {
        sum = bias_type == 0 ? static_cast<const float*>(bias)[oc]
            : bias_type == 1 ? __half2float(static_cast<const half*>(bias)[oc])
                             : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[oc]);
    }

    const int64_t kernel_x_count = phase_x < desc.kernel_size[0]
        ? (desc.kernel_size[0] - 1 - phase_x) / params.stride[0] + 1 : 0;
    const int64_t kernel_y_count = phase_y < desc.kernel_size[1]
        ? (desc.kernel_size[1] - 1 - phase_y) / params.stride[1] + 1 : 0;
    const int64_t phase_kernel_count = kernel_x_count * kernel_y_count;
    const int64_t reduction = phase_kernel_count * desc.input_channels_per_group;
    const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
        ? desc.output_channels_per_group * desc.kernel_volume : desc.output_channels_per_group;
    for (int64_t reduction_base = 0; reduction_base < reduction; reduction_base += tile) {
        const int64_t input_reduction = reduction_base + output_channel_lane;
        float input_value = 0.0f;
        if (spatial_active && input_reduction < reduction) {
            const int64_t phase_kernel = input_reduction / desc.input_channels_per_group;
            const int64_t local_ic = input_reduction % desc.input_channels_per_group;
            const int64_t kx = phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
            const int64_t ky = phase_y + (phase_kernel / kernel_x_count) * params.stride[1];
            const int64_t ix = (ox + params.padding_before[0] - kx) / params.stride[0];
            const int64_t iy = (oy + params.padding_before[1] - ky) / params.stride[1];
            if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1]) {
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                const char* address = reinterpret_cast<const char*>(input) + batch * input_nb3 +
                    ic * input_nb2 + iy * input_nb1 + ix * input_nb0;
                input_value = conv_nd_to_float(*reinterpret_cast<const ValueT*>(address));
            }
        }
        input_tile[spatial_lane][output_channel_lane] = input_value;

        const int64_t weight_reduction = reduction_base + spatial_lane;
        float weight_value = 0.0f;
        if (channel_active && weight_reduction < reduction) {
            const int64_t phase_kernel = weight_reduction / desc.input_channels_per_group;
            const int64_t local_ic = weight_reduction % desc.input_channels_per_group;
            const int64_t kx = phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
            const int64_t ky = phase_y + (phase_kernel / kernel_x_count) * params.stride[1];
            const int64_t kernel = kx + desc.kernel_size[0] * ky;
            const int64_t ic = group * desc.input_channels_per_group + local_ic;
            weight_value = conv_nd_weight<QuantType, WeightT>(
                weight, kernel, ic, local_oc, row_elements, desc.kernel_volume,
                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
        }
        weight_tile[output_channel_lane][spatial_lane] = weight_value;
        __syncthreads();
        if (spatial_active && channel_active) {
            const int64_t remaining = reduction - reduction_base;
            const int count = remaining < tile ? static_cast<int>(remaining) : tile;
            #pragma unroll
            for (int k = 0; k < tile; ++k) {
                if (k < count) sum += input_tile[spatial_lane][k] * weight_tile[output_channel_lane][k];
            }
        }
        __syncthreads();
    }
    if (!spatial_active || !channel_active) return;
    char* address = reinterpret_cast<char*>(output) + batch * output_nb3 +
        oc * output_nb2 + oy * output_nb1 + ox * output_nb0;
    *reinterpret_cast<ValueT*>(address) = conv_nd_from_float<ValueT>(sum);
}

template <int QuantType, typename WeightT, typename ValueT>
__global__ void conv_transpose_2d_phase_wmma_kernel(
    const void* weight, const ValueT* input, const void* bias, int bias_type, ValueT* output,
    ops_conv_nd_encoded_params params, ops_conv_nd_desc desc,
    size_t weight_nb0, size_t weight_nb1, size_t weight_nb2,
    size_t input_nb0, size_t input_nb1, size_t input_nb2, size_t input_nb3,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3
) {
    using namespace nvcuda;
    constexpr int tile = 16;
    constexpr int warps_per_block = 8;
    __shared__ half input_tile[tile * tile];
    __shared__ half weight_tiles[warps_per_block][tile * tile];
    __shared__ float output_tiles[warps_per_block][tile * tile];

    const int thread = threadIdx.x;
    const int warp = thread / warpSize;
    const int lane = thread % warpSize;
    const int64_t channel_tiles = (desc.output_channels_per_group + tile - 1) / tile;
    const int64_t channel_blocks = (channel_tiles + warps_per_block - 1) / warps_per_block;
    const int64_t phase_count = static_cast<int64_t>(params.stride[0]) * params.stride[1];
    const int64_t phase_and_group = blockIdx.y / channel_blocks;
    const int64_t channel_block = blockIdx.y % channel_blocks;
    const int64_t phase = phase_and_group % phase_count;
    const int64_t group = phase_and_group / phase_count;
    const int64_t channel_tile = channel_block * warps_per_block + warp;
    const int64_t phase_x = phase % params.stride[0];
    const int64_t phase_y = phase / params.stride[0];
    const int64_t padding_x_mod = params.padding_before[0] % params.stride[0];
    const int64_t padding_y_mod = params.padding_before[1] % params.stride[1];
    const int64_t first_x = (phase_x - padding_x_mod + params.stride[0]) % params.stride[0];
    const int64_t first_y = (phase_y - padding_y_mod + params.stride[1]) % params.stride[1];
    const int64_t phase_width = first_x < desc.output_size[0]
        ? (desc.output_size[0] - 1 - first_x) / params.stride[0] + 1 : 0;
    const int64_t phase_height = first_y < desc.output_size[1]
        ? (desc.output_size[1] - 1 - first_y) / params.stride[1] + 1 : 0;
    const int64_t phase_volume = phase_width * phase_height;
    const int64_t batch = blockIdx.z;
    const int64_t kernel_x_count = phase_x < desc.kernel_size[0]
        ? (desc.kernel_size[0] - 1 - phase_x) / params.stride[0] + 1 : 0;
    const int64_t kernel_y_count = phase_y < desc.kernel_size[1]
        ? (desc.kernel_size[1] - 1 - phase_y) / params.stride[1] + 1 : 0;
    const int64_t reduction = kernel_x_count * kernel_y_count * desc.input_channels_per_group;
    const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
        ? desc.output_channels_per_group * desc.kernel_volume : desc.output_channels_per_group;

    wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int64_t reduction_base = 0; reduction_base < reduction; reduction_base += tile) {
        for (int index = thread; index < tile * tile; index += blockDim.x) {
            const int spatial_lane = index / tile;
            const int reduction_lane = index % tile;
            const int64_t packed_spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
            const int64_t r = reduction_base + reduction_lane;
            float value = 0.0f;
            if (packed_spatial < phase_volume && r < reduction) {
                const int64_t packed_x = packed_spatial % phase_width;
                const int64_t packed_y = packed_spatial / phase_width;
                const int64_t ox = first_x + packed_x * params.stride[0];
                const int64_t oy = first_y + packed_y * params.stride[1];
                const int64_t phase_kernel = r / desc.input_channels_per_group;
                const int64_t local_ic = r % desc.input_channels_per_group;
                const int64_t kx = phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
                const int64_t ky = phase_y + (phase_kernel / kernel_x_count) * params.stride[1];
                const int64_t ix = (ox + params.padding_before[0] - kx) / params.stride[0];
                const int64_t iy = (oy + params.padding_before[1] - ky) / params.stride[1];
                if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1]) {
                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
                    const char* address = reinterpret_cast<const char*>(input) + batch * input_nb3 +
                        ic * input_nb2 + iy * input_nb1 + ix * input_nb0;
                    value = conv_nd_to_float(*reinterpret_cast<const ValueT*>(address));
                }
            }
            input_tile[index] = __float2half(value);
        }

        for (int index = lane; index < tile * tile; index += warpSize) {
            const int output_channel_lane = index / tile;
            const int reduction_lane = index % tile;
            const int64_t local_oc = channel_tile * tile + output_channel_lane;
            const int64_t r = reduction_base + reduction_lane;
            float value = 0.0f;
            if (channel_tile < channel_tiles && local_oc < desc.output_channels_per_group && r < reduction) {
                const int64_t phase_kernel = r / desc.input_channels_per_group;
                const int64_t local_ic = r % desc.input_channels_per_group;
                const int64_t kx = phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
                const int64_t ky = phase_y + (phase_kernel / kernel_x_count) * params.stride[1];
                const int64_t kernel = kx + desc.kernel_size[0] * ky;
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                value = conv_nd_weight<QuantType, WeightT>(
                    weight, kernel, ic, local_oc, row_elements, desc.kernel_volume,
                    desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
            }
            weight_tiles[warp][reduction_lane + output_channel_lane * tile] = __float2half(value);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, tile, tile, tile, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, tile, tile, tile, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, input_tile, tile);
        wmma::load_matrix_sync(b, weight_tiles[warp], tile);
        wmma::mma_sync(accumulator, a, b, accumulator);
        __syncthreads();
    }

    wmma::store_matrix_sync(output_tiles[warp], accumulator, tile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < tile * tile; index += warpSize) {
        const int spatial_lane = index / tile;
        const int output_channel_lane = index % tile;
        const int64_t packed_spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
        const int64_t local_oc = channel_tile * tile + output_channel_lane;
        if (channel_tile >= channel_tiles || packed_spatial >= phase_volume ||
            local_oc >= desc.output_channels_per_group) continue;

        const int64_t packed_x = packed_spatial % phase_width;
        const int64_t packed_y = packed_spatial / phase_width;
        const int64_t ox = first_x + packed_x * params.stride[0];
        const int64_t oy = first_y + packed_y * params.stride[1];
        const int64_t oc = group * desc.output_channels_per_group + local_oc;
        float value = output_tiles[warp][index];
        if (bias) {
            value += bias_type == 0 ? static_cast<const float*>(bias)[oc]
                : bias_type == 1 ? __half2float(static_cast<const half*>(bias)[oc])
                                 : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[oc]);
        }
        char* address = reinterpret_cast<char*>(output) + batch * output_nb3 +
            oc * output_nb2 + oy * output_nb1 + ox * output_nb0;
        *reinterpret_cast<ValueT*>(address) = conv_nd_from_float<ValueT>(value);
    }
}

template <int QuantType, typename WeightT, typename ValueT, bool Transposed = false>
__global__ void conv_nd_wmma_kernel(
    const void* weight, const ValueT* input, const void* bias, int bias_type, ValueT* output,
    ops_conv_nd_encoded_params params, ops_conv_nd_desc desc,
    size_t weight_nb0, size_t weight_nb1, size_t weight_nb2,
    size_t input_nb0, size_t input_nb1, size_t input_nb2, size_t input_nb3,
    size_t output_nb0, size_t output_nb1, size_t output_nb2, size_t output_nb3
) {
    using namespace nvcuda;
    constexpr int tile = 16;
    constexpr int warps_per_block = 8;
    __shared__ half input_tile[tile * tile];
    __shared__ half weight_tiles[warps_per_block][tile * tile];
    __shared__ float output_tiles[warps_per_block][tile * tile];
    const int thread = threadIdx.x;
    const int warp = thread / warpSize;
    const int lane = thread % warpSize;
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t channel_tiles = (desc.output_channels_per_group + tile - 1) / tile;
    const int64_t channel_blocks = (channel_tiles + warps_per_block - 1) / warps_per_block;
    const int64_t group = blockIdx.y / channel_blocks;
    const int64_t channel_block = blockIdx.y % channel_blocks;
    const int64_t channel_tile = channel_block * warps_per_block + warp;
    const int64_t batch = blockIdx.z;
    const int64_t reduction = desc.kernel_volume * desc.input_channels_per_group;
    const int64_t row_channels = Transposed
        ? desc.output_channels_per_group : desc.input_channels_per_group;
    const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
        ? row_channels * desc.kernel_volume : row_channels;
    const bool pointwise = desc.kernel_volume == 1 && params.stride[0] == 1 &&
        params.stride[1] == 1 && params.stride[2] == 1 &&
        params.padding_before[0] == 0 && params.padding_before[1] == 0 && params.padding_before[2] == 0;
    const bool common_3x3 = desc.kernel_size[0] == 3 && desc.kernel_size[1] == 3 &&
        (desc.spatial_dims == 2 || desc.kernel_size[2] == 3) &&
        desc.input_channels_per_group % tile == 0 &&
        params.stride[0] == 1 && params.stride[1] == 1 &&
        params.dilation[0] == 1 && params.dilation[1] == 1 &&
        (desc.spatial_dims == 2 || (params.stride[2] == 1 && params.dilation[2] == 1));
    wmma::fragment<wmma::accumulator, tile, tile, tile, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int64_t reduction_base = 0; reduction_base < reduction; reduction_base += tile) {
        const int64_t tile_kernel = pointwise ? 0
            : (common_3x3 ? reduction_base / desc.input_channels_per_group : -1);
        const int64_t tile_channel_base = (pointwise || common_3x3)
            ? reduction_base - tile_kernel * desc.input_channels_per_group : -1;
        for (int index = thread; index < tile * tile; index += blockDim.x) {
            const int spatial_lane = index / tile;
            const int reduction_lane = index % tile;
            const int64_t spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
            const int64_t r = reduction_base + reduction_lane;
            float value = 0.0f;
            if (spatial < output_volume && r < reduction) {
                const int64_t ox = spatial % desc.output_size[0];
                const int64_t rem = spatial / desc.output_size[0];
                const int64_t oy = rem % desc.output_size[1];
                const int64_t oz = rem / desc.output_size[1];
                const int64_t kernel = (pointwise || common_3x3)
                    ? tile_kernel : r / desc.input_channels_per_group;
                const int64_t local_ic = (pointwise || common_3x3)
                    ? tile_channel_base + reduction_lane : r % desc.input_channels_per_group;
                const int64_t kx = pointwise ? 0 : kernel % desc.kernel_size[0];
                const int64_t krem = pointwise ? 0 : kernel / desc.kernel_size[0];
                const int64_t ky = pointwise ? 0 : krem % desc.kernel_size[1];
                const int64_t kz = pointwise ? 0 : krem / desc.kernel_size[1];
                const int64_t ix = pointwise ? ox : ox * params.stride[0] - params.padding_before[0] + kx * params.dilation[0];
                const int64_t iy = pointwise ? oy : oy * params.stride[1] - params.padding_before[1] + ky * params.dilation[1];
                const int64_t iz = pointwise ? oz : oz * params.stride[2] - params.padding_before[2] + kz * params.dilation[2];
                if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1] &&
                    iz >= 0 && iz < desc.input_size[2]) {
                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
                    const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                    const char* address = desc.spatial_dims == 2
                        ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                        : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                    value = conv_nd_to_float(*reinterpret_cast<const ValueT*>(address));
                }
            }
            input_tile[index] = __float2half(value);
        }
        for (int index = lane; index < tile * tile; index += warpSize) {
            const int output_channel_lane = index / tile;
            const int weight_reduction_lane = index % tile;
            const int64_t local_oc = channel_tile * tile + output_channel_lane;
            const int64_t oc = group * desc.output_channels_per_group + local_oc;
            const int64_t wr = reduction_base + weight_reduction_lane;
            float weight_value = 0.0f;
            if (channel_tile < channel_tiles && local_oc < desc.output_channels_per_group && wr < reduction) {
                const int64_t kernel = (pointwise || common_3x3)
                    ? tile_kernel : wr / desc.input_channels_per_group;
                const int64_t local_ic = (pointwise || common_3x3)
                    ? tile_channel_base + weight_reduction_lane : wr % desc.input_channels_per_group;
                const int64_t weight_outer = Transposed
                    ? group * desc.input_channels_per_group + local_ic : oc;
                const int64_t weight_inner = Transposed ? local_oc : local_ic;
                weight_value = conv_nd_weight<QuantType, WeightT>(
                    weight, kernel, weight_outer, weight_inner, row_elements, desc.kernel_volume,
                    desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
            }
            // WMMA B is column-major: [K, N].
            weight_tiles[warp][weight_reduction_lane + output_channel_lane * tile] = __float2half(weight_value);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, tile, tile, tile, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, tile, tile, tile, half, wmma::col_major> b;
        wmma::load_matrix_sync(a, input_tile, tile);
        wmma::load_matrix_sync(b, weight_tiles[warp], tile);
        wmma::mma_sync(accumulator, a, b, accumulator);
        __syncthreads();
    }
    wmma::store_matrix_sync(output_tiles[warp], accumulator, tile, wmma::mem_row_major);
    __syncwarp();
    for (int index = lane; index < tile * tile; index += warpSize) {
        const int spatial_lane = index / tile;
        const int output_channel_lane = index % tile;
        const int64_t spatial = static_cast<int64_t>(blockIdx.x) * tile + spatial_lane;
        const int64_t local_oc = channel_tile * tile + output_channel_lane;
        if (channel_tile >= channel_tiles || spatial >= output_volume ||
            local_oc >= desc.output_channels_per_group) continue;
        const int64_t oc = group * desc.output_channels_per_group + local_oc;
        float value = output_tiles[warp][index];
        if (bias) {
            value += bias_type == 0 ? static_cast<const float*>(bias)[oc]
                : bias_type == 1 ? __half2float(static_cast<const half*>(bias)[oc])
                                 : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[oc]);
        }
        const int64_t ox = spatial % desc.output_size[0];
        const int64_t rem = spatial / desc.output_size[0];
        const int64_t oy = rem % desc.output_size[1];
        const int64_t oz = rem / desc.output_size[1];
        const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
        char* address = desc.spatial_dims == 2
            ? reinterpret_cast<char*>(output) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
            : reinterpret_cast<char*>(output) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
        *reinterpret_cast<ValueT*>(address) = conv_nd_from_float<ValueT>(value);
    }
}

template <typename T>
constexpr cudaDataType_t conv_nd_cuda_data_type();

template <>
constexpr cudaDataType_t conv_nd_cuda_data_type<float>() { return CUDA_R_32F; }

template <>
constexpr cudaDataType_t conv_nd_cuda_data_type<half>() { return CUDA_R_16F; }

template <>
constexpr cudaDataType_t conv_nd_cuda_data_type<__nv_bfloat16>() { return CUDA_R_16BF; }

template <typename ValueT>
__global__ void conv_nd_add_bias_kernel(
    ValueT* output, int64_t total, int64_t output_volume, int64_t output_channels,
    const void* bias, int bias_type
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;

    const int64_t output_channel = (index / output_volume) % output_channels;
    const float bias_value = bias_type == 0
        ? static_cast<const float*>(bias)[output_channel]
        : bias_type == 1
            ? __half2float(static_cast<const half*>(bias)[output_channel])
            : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[output_channel]);
    output[index] = conv_nd_from_float<ValueT>(conv_nd_to_float(output[index]) + bias_value);
}

template <typename WeightT, typename ValueT>
void launch_pointwise_conv_nd_cublas(
    cublasHandle_t cublas, cudaStream_t stream, bool transposed,
    const ops_conv_nd_node_params& params, const ops_conv_nd_desc& desc,
    ggml_tensor* output, int bias_type
) {
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    const int m = static_cast<int>(desc.output_size[0] * desc.output_size[1] * desc.output_size[2]);
    const int n = static_cast<int>(desc.output_channels_per_group);
    const int k = static_cast<int>(desc.input_channels_per_group);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cudaDataType_t weight_type = conv_nd_cuda_data_type<WeightT>();
    const cudaDataType_t value_type = conv_nd_cuda_data_type<ValueT>();
    const cublasOperation_t weight_operation = transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int weight_leading_dimension = transposed ? n : k;

    const char* weight_base = static_cast<const char*>(params.weight->data);
    const char* input_base = static_cast<const char*>(params.input->data);
    char* output_base = static_cast<char*>(output->data);
    const size_t weight_outer_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? params.weight->nb[1] : params.weight->nb[2];
    const size_t input_channel_stride = desc.spatial_dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = desc.spatial_dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = desc.spatial_dims == 2 ? output->nb[2] : output->nb[1];
    const size_t output_batch_stride = desc.spatial_dims == 2 ? output->nb[3] : output->nb[2];
    for (int64_t batch = 0; batch < desc.batch; ++batch) {
        for (int64_t group = 0; group < desc.groups; ++group) {
            const int64_t weight_outer = group * (transposed
                ? desc.input_channels_per_group : desc.output_channels_per_group);
            const void* weight = weight_base + weight_outer * weight_outer_stride;
            const void* input = input_base + batch * input_batch_stride +
                group * desc.input_channels_per_group * input_channel_stride;
            void* destination = output_base + batch * output_batch_stride +
                group * desc.output_channels_per_group * output_channel_stride;
            CUBLAS_CHECK(cublasGemmEx(
                cublas,
                CUBLAS_OP_N,
                weight_operation,
                m,
                n,
                k,
                &alpha,
                input,
                value_type,
                m,
                weight,
                weight_type,
                weight_leading_dimension,
                &beta,
                destination,
                value_type,
                m,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    if (params.bias) {
        const int64_t total = ggml_nelements(output);
        constexpr int block_size = 256;
        conv_nd_add_bias_kernel<<<
            static_cast<unsigned>((total + block_size - 1) / block_size), block_size, 0, stream>>>(
                static_cast<ValueT*>(output->data), total, m, desc.output_channels,
                params.bias->data, bias_type);
    }
}

template <int QuantType, typename WeightT, typename ValueT>
void launch_conv_nd(
    cublasHandle_t cublas, cudaStream_t stream, bool transposed, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc, ggml_tensor* output, bool tensor_cores
) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int block = output_volume >= 128 ? 256 : 32;
    const dim3 grid(static_cast<unsigned>((output_volume + block - 1) / block),
                    static_cast<unsigned>(desc.output_channels), static_cast<unsigned>(desc.batch));
    const int bias_type = !params.bias || params.bias->type == GGML_TYPE_F32 ? 0
        : params.bias->type == GGML_TYPE_F16 ? 1 : 2;
    const bool pointwise = desc.kernel_volume == 1 &&
        params.encoded.stride[0] == 1 && params.encoded.stride[1] == 1 && params.encoded.stride[2] == 1 &&
        params.encoded.padding_before[0] == 0 && params.encoded.padding_before[1] == 0 &&
        params.encoded.padding_before[2] == 0 && params.encoded.padding_after[0] == 0 &&
        params.encoded.padding_after[1] == 0 && params.encoded.padding_after[2] == 0 &&
        output_volume <= std::numeric_limits<int>::max() &&
        desc.input_channels_per_group <= std::numeric_limits<int>::max() &&
        desc.output_channels_per_group <= std::numeric_limits<int>::max() &&
        ggml_is_contiguous(params.weight) && ggml_is_contiguous(params.input) && ggml_is_contiguous(output);
    if constexpr (QuantType < 0 && std::is_same_v<WeightT, ValueT>) {
        if (pointwise && cublas) {
            launch_pointwise_conv_nd_cublas<WeightT, ValueT>(
                cublas, stream, transposed, params, desc, output, bias_type);
            return;
        }
    }
    [[maybe_unused]] const int64_t wmma_channel_tiles = (desc.output_channels_per_group + 15) / 16;
    [[maybe_unused]] const int wmma_warps = static_cast<int>(wmma_channel_tiles < 8 ? wmma_channel_tiles : 8);
    [[maybe_unused]] const int wmma_block = wmma_warps * 32;
    [[maybe_unused]] const int64_t wmma_channel_blocks = (wmma_channel_tiles + 7) / 8;
    const int64_t max_phase_volume =
        ((desc.output_size[0] + params.encoded.stride[0] - 1) / params.encoded.stride[0]) *
        ((desc.output_size[1] + params.encoded.stride[1] - 1) / params.encoded.stride[1]);
    const int64_t transpose_phase_grid_y = static_cast<int64_t>(desc.groups) *
        params.encoded.stride[0] * params.encoded.stride[1] *
        ((desc.output_channels_per_group + 15) / 16);
#define LAUNCH_IMPLICIT() conv_nd_implicit_gemm_kernel<QuantType, WeightT, ValueT><<< \
        dim3(static_cast<unsigned>((output_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * ((desc.output_channels_per_group + 15) / 16)), \
             static_cast<unsigned>(desc.batch)), 256, 0, stream>>>( \
        params.weight->data, static_cast<const ValueT*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<ValueT*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_WMMA_F16() conv_nd_wmma_kernel<-1, half, half><<< \
        dim3(static_cast<unsigned>((output_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * wmma_channel_blocks), \
             static_cast<unsigned>(desc.batch)), wmma_block, 0, stream>>>( \
        params.weight->data, static_cast<const half*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<half*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_WMMA_QUANT() conv_nd_wmma_kernel<QuantType, void, float><<< \
        dim3(static_cast<unsigned>((output_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * wmma_channel_blocks), \
             static_cast<unsigned>(desc.batch)), wmma_block, 0, stream>>>( \
        params.weight->data, static_cast<const float*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<float*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_TRANSPOSE_POINTWISE_WMMA_QUANT() \
    conv_nd_wmma_kernel<QuantType, void, float, true><<< \
        dim3(static_cast<unsigned>((output_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * wmma_channel_blocks), \
             static_cast<unsigned>(desc.batch)), wmma_block, 0, stream>>>( \
        params.weight->data, static_cast<const float*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<float*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_TRANSPOSE_WMMA_F16() conv_transpose_2d_phase_wmma_kernel<-1, half, half><<< \
        dim3(static_cast<unsigned>((max_phase_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * params.encoded.stride[0] * \
                                   params.encoded.stride[1] * wmma_channel_blocks), \
             static_cast<unsigned>(desc.batch)), wmma_block, 0, stream>>>( \
        params.weight->data, static_cast<const half*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<half*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_TRANSPOSE_WMMA_QUANT() conv_transpose_2d_phase_wmma_kernel<QuantType, void, float><<< \
        dim3(static_cast<unsigned>((max_phase_volume + 15) / 16), \
             static_cast<unsigned>(desc.groups * params.encoded.stride[0] * \
                                   params.encoded.stride[1] * wmma_channel_blocks), \
             static_cast<unsigned>(desc.batch)), wmma_block, 0, stream>>>( \
        params.weight->data, static_cast<const float*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<float*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH_TRANSPOSE_IMPLICIT() conv_transpose_2d_phase_gemm_kernel<QuantType, WeightT, ValueT><<< \
        dim3(static_cast<unsigned>((max_phase_volume + 15) / 16), \
             static_cast<unsigned>(transpose_phase_grid_y), \
             static_cast<unsigned>(desc.batch)), 256, 0, stream>>>( \
        params.weight->data, static_cast<const ValueT*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<ValueT*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
#define LAUNCH(TRANSPOSED) conv_nd_kernel<QuantType, WeightT, ValueT, TRANSPOSED><<<grid, block, 0, stream>>>( \
        params.weight->data, static_cast<const ValueT*>(params.input->data), \
        params.bias ? params.bias->data : nullptr, bias_type, static_cast<ValueT*>(output->data), \
        params.encoded, desc, params.weight->nb[0], params.weight->nb[1], params.weight->nb[2], \
        params.input->nb[0], params.input->nb[1], params.input->nb[2], params.input->nb[3], \
        output->nb[0], output->nb[1], output->nb[2], output->nb[3])
    const bool use_implicit = !transposed && output_volume >= 64 &&
        desc.input_channels_per_group >= 16 && desc.output_channels_per_group >= 16;
    [[maybe_unused]] const bool use_transpose_pointwise = transposed && desc.kernel_volume == 1 &&
        output_volume >= 64 && desc.input_channels_per_group >= 16 &&
        desc.output_channels_per_group >= 16 &&
        params.encoded.stride[0] == 1 && params.encoded.stride[1] == 1 && params.encoded.stride[2] == 1 &&
        params.encoded.padding_before[0] == 0 && params.encoded.padding_before[1] == 0 &&
        params.encoded.padding_before[2] == 0 && params.encoded.padding_after[0] == 0 &&
        params.encoded.padding_after[1] == 0 && params.encoded.padding_after[2] == 0;
    [[maybe_unused]] const bool use_transpose_implicit = transposed && desc.spatial_dims == 2 &&
        params.encoded.dilation[0] == 1 && params.encoded.dilation[1] == 1 &&
        params.encoded.stride[2] == 1 &&
        (params.encoded.stride[0] > 1 || params.encoded.stride[1] > 1) &&
        max_phase_volume >= 16 && desc.input_channels_per_group >= 16 &&
        desc.output_channels_per_group >= 16 && transpose_phase_grid_y <= 65535;
    [[maybe_unused]] constexpr bool wmma_quant_type = QuantType == GGML_TYPE_Q4_0 || QuantType == GGML_TYPE_Q4_1 ||
        QuantType == GGML_TYPE_Q5_0 || QuantType == GGML_TYPE_Q5_1 || QuantType == GGML_TYPE_Q8_0 ||
        QuantType == GGML_TYPE_Q2_K || QuantType == GGML_TYPE_Q3_K || QuantType == GGML_TYPE_Q4_K ||
        QuantType == GGML_TYPE_Q5_K || QuantType == GGML_TYPE_Q6_K ||
        QuantType == GGML_TYPE_IQ4_NL || QuantType == GGML_TYPE_IQ4_XS ||
        QuantType == GGML_TYPE_MXFP4;
    if constexpr (QuantType < 0 && std::is_same_v<WeightT, half> && std::is_same_v<ValueT, half>) {
        if (use_implicit && tensor_cores) LAUNCH_WMMA_F16();
        else if (use_implicit) LAUNCH_IMPLICIT();
        else if (use_transpose_implicit && tensor_cores) LAUNCH_TRANSPOSE_WMMA_F16();
        else if (use_transpose_implicit) LAUNCH_TRANSPOSE_IMPLICIT();
        else if (transposed) LAUNCH(true);
        else LAUNCH(false);
    } else if constexpr (wmma_quant_type && std::is_same_v<ValueT, float>) {
        if (use_implicit && tensor_cores) LAUNCH_WMMA_QUANT();
        else if (use_implicit) LAUNCH_IMPLICIT();
        else if (use_transpose_pointwise && tensor_cores) LAUNCH_TRANSPOSE_POINTWISE_WMMA_QUANT();
        else if (use_transpose_implicit && tensor_cores) LAUNCH_TRANSPOSE_WMMA_QUANT();
        else if (use_transpose_implicit) LAUNCH_TRANSPOSE_IMPLICIT();
        else if (transposed) LAUNCH(true);
        else LAUNCH(false);
    } else {
        if (use_implicit) LAUNCH_IMPLICIT();
        else if constexpr (QuantType >= 0) {
            if (use_transpose_implicit) LAUNCH_TRANSPOSE_IMPLICIT();
            else if (transposed) LAUNCH(true);
            else LAUNCH(false);
        }
        else if (transposed) LAUNCH(true);
        else LAUNCH(false);
    }
#undef LAUNCH
#undef LAUNCH_IMPLICIT
#undef LAUNCH_WMMA_F16
#undef LAUNCH_WMMA_QUANT
#undef LAUNCH_TRANSPOSE_POINTWISE_WMMA_QUANT
#undef LAUNCH_TRANSPOSE_WMMA_F16
#undef LAUNCH_TRANSPOSE_WMMA_QUANT
#undef LAUNCH_TRANSPOSE_IMPLICIT
}

template <typename ValueT>
bool dispatch_weight(cudaStream_t stream, bool transposed, const ops_conv_nd_node_params& params,
                     const ops_conv_nd_desc& desc, ggml_tensor* output, bool tensor_cores,
                     cublasHandle_t cublas) {
    switch (params.weight->type) {
#define DISPATCH(Q, W) launch_conv_nd<Q, W, ValueT>(cublas, stream, transposed, params, desc, output, tensor_cores)
        case GGML_TYPE_F32: DISPATCH(-1, float); break;
        case GGML_TYPE_F16: DISPATCH(-1, half); break;
        case GGML_TYPE_BF16: DISPATCH(-1, __nv_bfloat16); break;
        case GGML_TYPE_Q4_0: DISPATCH(GGML_TYPE_Q4_0, void); break;
        case GGML_TYPE_Q4_1: DISPATCH(GGML_TYPE_Q4_1, void); break;
        case GGML_TYPE_Q5_0: DISPATCH(GGML_TYPE_Q5_0, void); break;
        case GGML_TYPE_Q5_1: DISPATCH(GGML_TYPE_Q5_1, void); break;
        case GGML_TYPE_Q8_0: DISPATCH(GGML_TYPE_Q8_0, void); break;
        case GGML_TYPE_Q2_K: DISPATCH(GGML_TYPE_Q2_K, void); break;
        case GGML_TYPE_Q3_K: DISPATCH(GGML_TYPE_Q3_K, void); break;
        case GGML_TYPE_Q4_K: DISPATCH(GGML_TYPE_Q4_K, void); break;
        case GGML_TYPE_Q5_K: DISPATCH(GGML_TYPE_Q5_K, void); break;
        case GGML_TYPE_Q6_K: DISPATCH(GGML_TYPE_Q6_K, void); break;
        case GGML_TYPE_IQ4_NL: DISPATCH(GGML_TYPE_IQ4_NL, void); break;
        case GGML_TYPE_IQ4_XS: DISPATCH(GGML_TYPE_IQ4_XS, void); break;
        case GGML_TYPE_MXFP4: DISPATCH(GGML_TYPE_MXFP4, void); break;
        default: return false;
    }
#undef DISPATCH
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_conv_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
    ops_conv_nd_node_params params;
    if (!ops_extract_conv_nd_params(node, params)) return false;
    const int op = static_cast<int>(node->op);
    const int dims = op == GGML_OP_OPS_VIRT_CONV_2D || op == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D ? 2 : 3;
    const bool transposed = op == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_2D || op == GGML_OP_OPS_VIRT_CONV_TRANSPOSE_3D;
    ggml_tensor* srcs[] = { params.weight, params.input, params.bias };
    ops_request request = { ggml_backend_get_device(backend), op, srcs, params.bias ? 3 : 2,
                            &params.encoded, sizeof(params.encoded), node };
    ops_conv_nd_desc desc;
    if (!ops_validate_conv_nd_contract(request, dims, transposed, &desc)) return false;
    cudaStream_t stream = ggml_ops_ext_bridge_cuda_get_stream(backend);
    if (!stream) return false;
    const int device = ggml_ops_ext_bridge_cuda_get_device(backend);
    const bool tensor_cores = conv_nd_tensor_cores_available(device);
    cublasHandle_t cublas = ggml_ops_ext_bridge_cuda_get_cublas(backend);
    switch (params.input->type) {
        case GGML_TYPE_F32: return dispatch_weight<float>(stream, transposed, params, desc, node, tensor_cores, cublas);
        case GGML_TYPE_F16: return dispatch_weight<half>(stream, transposed, params, desc, node, tensor_cores, cublas);
        case GGML_TYPE_BF16: return dispatch_weight<__nv_bfloat16>(stream, transposed, params, desc, node, tensor_cores, cublas);
        default: return false;
    }
}

} // namespace ggml_ops_ext::cuda
