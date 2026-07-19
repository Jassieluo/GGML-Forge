#include "ops/ops.h"
#include "ops_sycl.h"
#include "common.hpp"
#include "quantized_conv.h"

#include <vector>

namespace ggml_ops_ext::sycl {

template <int QuantType, typename WeightT, typename ValueT, bool Transposed>
class ConvNDKernel;

template <int QuantType, typename WeightT, typename ValueT, bool Transposed>
class ConvNDImplicitGemmKernel;

template <int QuantType, typename WeightT, typename ValueT, int Dims>
class ConvTransposeNDPhaseGemmKernel;

template <typename ValueT>
class ConvNDBiasKernel;

template <int QuantType, typename WeightT>
inline float conv_nd_weight_sycl(
    const void* weight, int64_t kernel, int64_t outer, int64_t inner,
    int64_t row_elements, int64_t kernel_volume, ops_weight_layout layout,
    size_t nb0, size_t nb1, size_t nb2
) {
    if constexpr (QuantType >= 0) {
        const int64_t row = layout == ops_weight_layout::flattened_rows ? outer : outer * kernel_volume + kernel;
        const int64_t column = layout == ops_weight_layout::flattened_rows ? inner * kernel_volume + kernel : inner;
        return load_quantized_row_value_sycl<QuantType>(
            weight, row, column, row_elements);
    } else {
        const char* address = layout == ops_weight_layout::flattened_rows
            ? static_cast<const char*>(weight) + outer * nb1 + (inner * kernel_volume + kernel) * nb0
            : static_cast<const char*>(weight) + outer * nb2 + kernel * nb1 + inner * nb0;
        return static_cast<float>(*reinterpret_cast<const WeightT*>(address));
    }
}

template <typename ValueT>
bool launch_pointwise_conv_nd_mkl_sycl(
    ::sycl::queue* queue, bool transposed, const ops_conv_nd_node_params& values,
    const ops_conv_nd_desc& desc, ggml_tensor* output
) {
    if (desc.kernel_volume != 1 || values.weight->type != values.input->type ||
        output->type != values.input->type || !ggml_is_contiguous(values.weight) ||
        !ggml_is_contiguous(values.input) || !ggml_is_contiguous(output)) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (values.encoded.stride[axis] != 1 || values.encoded.dilation[axis] != 1 ||
            values.encoded.padding_before[axis] != 0 || values.encoded.padding_after[axis] != 0 ||
            values.encoded.output_padding[axis] != 0) return false;
    }

    const int64_t spatial = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    if (spatial < 16 || desc.input_channels_per_group < 16 ||
        desc.output_channels_per_group < 16) return false;
    const size_t weight_outer_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? values.weight->nb[1] : values.weight->nb[2];
    const size_t expected_weight_stride = static_cast<size_t>(transposed
        ? desc.output_channels_per_group : desc.input_channels_per_group) * sizeof(ValueT);
    if (weight_outer_stride != expected_weight_stride) return false;

    const size_t input_channel_stride = desc.spatial_dims == 2 ? values.input->nb[2] : values.input->nb[1];
    const size_t input_batch_stride = desc.spatial_dims == 2 ? values.input->nb[3] : values.input->nb[2];
    const size_t output_channel_stride = desc.spatial_dims == 2 ? output->nb[2] : output->nb[1];
    const size_t output_batch_stride = desc.spatial_dims == 2 ? output->nb[3] : output->nb[2];
    const char* weight_base = static_cast<const char*>(values.weight->data);
    const char* input_base = static_cast<const char*>(values.input->data);
    char* output_base = static_cast<char*>(output->data);
    std::vector<::sycl::event> events;
    events.reserve(static_cast<size_t>(desc.batch * desc.groups));
    try {
        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t group = 0; group < desc.groups; ++group) {
                const int64_t weight_outer = group * (transposed
                    ? desc.input_channels_per_group : desc.output_channels_per_group);
                const ValueT* weight = reinterpret_cast<const ValueT*>(
                    weight_base + weight_outer * weight_outer_stride);
                const ValueT* input = reinterpret_cast<const ValueT*>(
                    input_base + batch * input_batch_stride +
                    group * desc.input_channels_per_group * input_channel_stride);
                ValueT* destination = reinterpret_cast<ValueT*>(
                    output_base + batch * output_batch_stride +
                    group * desc.output_channels_per_group * output_channel_stride);
                events.push_back(oneapi::mkl::blas::column_major::gemm(
                    *queue,
                    oneapi::mkl::transpose::nontrans,
                    transposed ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans,
                    spatial,
                    desc.output_channels_per_group,
                    desc.input_channels_per_group,
                    ValueT(1),
                    input,
                    spatial,
                    weight,
                    transposed ? desc.output_channels_per_group : desc.input_channels_per_group,
                    ValueT(0),
                    destination,
                    spatial));
            }
        }
    } catch (const std::exception&) {
        return false;
    }

    if (values.bias) {
        const int64_t total = ggml_nelements(output);
        const int64_t output_channels = desc.output_channels;
        const void* bias = values.bias->data;
        const int bias_type = values.bias->type == GGML_TYPE_F32 ? 0 : 1;
        ValueT* destination = static_cast<ValueT*>(output->data);
        queue->submit([&](::sycl::handler& handler) {
            handler.depends_on(events);
            handler.parallel_for<ConvNDBiasKernel<ValueT>>(
                ::sycl::range<1>(static_cast<size_t>(total)), [=](::sycl::id<1> id) {
                    const int64_t index = static_cast<int64_t>(id[0]);
                    const int64_t output_channel = (index / spatial) % output_channels;
                    const float bias_value = bias_type == 0
                        ? static_cast<const float*>(bias)[output_channel]
                        : static_cast<float>(static_cast<const ::sycl::half*>(bias)[output_channel]);
                    destination[index] = static_cast<ValueT>(
                        static_cast<float>(destination[index]) + bias_value);
                });
        });
    }
    return true;
}

template <int QuantType, typename WeightT, typename ValueT, bool Transposed>
void launch_conv_nd_sycl(
    ::sycl::queue* queue, const ops_conv_nd_node_params& values,
    const ops_conv_nd_desc& desc, ggml_tensor* output
) {
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const size_t local_size = output_volume >= 128 ? 256 : 32;
    const int64_t spatial_groups = (output_volume + local_size - 1) / local_size;
    const int64_t total_groups = desc.batch * desc.output_channels * spatial_groups;
    const void* weight = values.weight->data;
    const ValueT* input = static_cast<const ValueT*>(values.input->data);
    const void* bias = values.bias ? values.bias->data : nullptr;
    ValueT* destination = static_cast<ValueT*>(output->data);
    const int bias_type = !values.bias || values.bias->type == GGML_TYPE_F32 ? 0 : 1;
    const auto params = values.encoded;
    const size_t weight_nb0 = values.weight->nb[0], weight_nb1 = values.weight->nb[1], weight_nb2 = values.weight->nb[2];
    const size_t input_nb0 = values.input->nb[0], input_nb1 = values.input->nb[1];
    const size_t input_nb2 = values.input->nb[2], input_nb3 = values.input->nb[3];
    const size_t output_nb0 = output->nb[0], output_nb1 = output->nb[1];
    const size_t output_nb2 = output->nb[2], output_nb3 = output->nb[3];
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> weight_tile(::sycl::range<1>(local_size), handler);
        handler.parallel_for<ConvNDKernel<QuantType, WeightT, ValueT, Transposed>>(
            ::sycl::nd_range<1>(::sycl::range<1>(total_groups * local_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) {
                const int64_t local_id = item.get_local_linear_id();
                const int64_t group_id = item.get_group_linear_id();
                const int64_t spatial_group = group_id % spatial_groups;
                const int64_t oc = (group_id / spatial_groups) % desc.output_channels;
                const int64_t batch = group_id / (spatial_groups * desc.output_channels);
                const int64_t spatial = spatial_group * local_size + local_id;
                const bool active = spatial < output_volume;
                const int64_t ox = spatial % desc.output_size[0];
                const int64_t rem = spatial / desc.output_size[0];
                const int64_t oy = rem % desc.output_size[1];
                const int64_t oz = rem / desc.output_size[1];
                const int64_t group = oc / desc.output_channels_per_group;
                const int64_t local_oc = oc % desc.output_channels_per_group;
                float sum = 0.0f;
                if (bias) sum = bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                                : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc]);
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
                                            sum += static_cast<float>(*reinterpret_cast<const ValueT*>(input_address)) *
                                                conv_nd_weight_sycl<QuantType, WeightT>(weight, kernel, ic, local_oc,
                                                    row_elements, desc.kernel_volume, desc.weight_layout,
                                                    weight_nb0, weight_nb1, weight_nb2);
                                        }
                                    }
                                }
                            }
                            const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
                            char* output_address = desc.spatial_dims == 2
                                ? reinterpret_cast<char*>(destination) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
                                : reinterpret_cast<char*>(destination) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
                            *reinterpret_cast<ValueT*>(output_address) = static_cast<ValueT>(sum);
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
                            ix = sx / params.stride[0]; iy = sy / params.stride[1]; iz = sz / params.stride[2];
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
                                sum += static_cast<float>(*reinterpret_cast<const ValueT*>(input_address)) *
                                    conv_nd_weight_sycl<QuantType, WeightT>(weight, kernel, outer, inner, row_elements,
                                        desc.kernel_volume, desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                            }
                        }
                        continue;
                    }
                    for (int64_t channel_base = 0; channel_base < desc.input_channels_per_group; channel_base += local_size) {
                        const int64_t loaded_ic = channel_base + local_id;
                        if (loaded_ic < desc.input_channels_per_group) {
                            const int64_t loaded_channel = group * desc.input_channels_per_group + loaded_ic;
                            const int64_t outer = Transposed ? loaded_channel : oc;
                            const int64_t inner = Transposed ? local_oc : loaded_ic;
                            const int64_t row_channels = Transposed ? desc.output_channels_per_group : desc.input_channels_per_group;
                            const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                                ? row_channels * desc.kernel_volume : row_channels;
                            weight_tile[local_id] = conv_nd_weight_sycl<QuantType, WeightT>(
                                weight, kernel, outer, inner, row_elements, desc.kernel_volume,
                                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                        if (input_valid) {
                            const int64_t remaining = desc.input_channels_per_group - channel_base;
                            const int64_t tile_count = remaining < static_cast<int64_t>(local_size)
                                ? remaining : static_cast<int64_t>(local_size);
                            const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                            for (int64_t tile_ic = 0; tile_ic < tile_count; ++tile_ic) {
                                const int64_t ic = group * desc.input_channels_per_group + channel_base + tile_ic;
                                const char* input_address = desc.spatial_dims == 2
                                    ? reinterpret_cast<const char*>(input) + batch * input_nb3 + ic * input_nb2 + iy * input_nb1 + ix * input_nb0
                                    : reinterpret_cast<const char*>(input) + batch * input_nb2 + ic * input_nb1 + input_flat * input_nb0;
                                sum += static_cast<float>(*reinterpret_cast<const ValueT*>(input_address)) * weight_tile[tile_ic];
                            }
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                }
                if (!active) return;
                const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
                char* output_address = desc.spatial_dims == 2
                    ? reinterpret_cast<char*>(destination) + batch * output_nb3 + oc * output_nb2 + oy * output_nb1 + ox * output_nb0
                    : reinterpret_cast<char*>(destination) + batch * output_nb2 + oc * output_nb1 + output_flat * output_nb0;
                *reinterpret_cast<ValueT*>(output_address) = static_cast<ValueT>(sum);
            });
    });
}

template <int QuantType, typename WeightT, typename ValueT, bool Transposed>
void launch_conv_nd_implicit_gemm_sycl(
    ::sycl::queue* queue, const ops_conv_nd_node_params& values,
    const ops_conv_nd_desc& desc, ggml_tensor* output
) {
    constexpr int tile = 16;
    constexpr int channel_block = 2 * tile;
    constexpr int stride = tile + 1;
    constexpr size_t local_size = tile * tile;
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const int64_t spatial_tiles = (output_volume + tile - 1) / tile;
    const int64_t channel_tiles = (desc.output_channels_per_group + channel_block - 1) / channel_block;
    const int64_t total_groups = desc.batch * desc.groups * channel_tiles * spatial_tiles;
    const void* weight = values.weight->data;
    const ValueT* input = static_cast<const ValueT*>(values.input->data);
    const void* bias = values.bias ? values.bias->data : nullptr;
    ValueT* destination = static_cast<ValueT*>(output->data);
    const int bias_type = !values.bias || values.bias->type == GGML_TYPE_F32 ? 0 : 1;
    const auto params = values.encoded;
    const size_t weight_nb0 = values.weight->nb[0], weight_nb1 = values.weight->nb[1], weight_nb2 = values.weight->nb[2];
    const size_t input_nb0 = values.input->nb[0], input_nb1 = values.input->nb[1];
    const size_t input_nb2 = values.input->nb[2], input_nb3 = values.input->nb[3];
    const size_t output_nb0 = output->nb[0], output_nb1 = output->nb[1];
    const size_t output_nb2 = output->nb[2], output_nb3 = output->nb[3];
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> input_tile(::sycl::range<1>(tile * stride), handler);
        ::sycl::local_accessor<float, 1> weight_tile(::sycl::range<1>(2 * tile * stride), handler);
        handler.parallel_for<ConvNDImplicitGemmKernel<QuantType, WeightT, ValueT, Transposed>>(
            ::sycl::nd_range<1>(::sycl::range<1>(total_groups * local_size), ::sycl::range<1>(local_size)),
            [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const int lane = static_cast<int>(item.get_local_linear_id());
                const int spatial_lane = lane & (tile - 1);
                const int output_channel_lane = lane / tile;
                int64_t group_linear = item.get_group_linear_id();
                const int64_t spatial_tile_id = group_linear % spatial_tiles;
                group_linear /= spatial_tiles;
                const int64_t channel_tile_id = group_linear % channel_tiles;
                group_linear /= channel_tiles;
                const int64_t group = group_linear % desc.groups;
                const int64_t batch = group_linear / desc.groups;
                const int64_t spatial = spatial_tile_id * tile + spatial_lane;
                const int64_t local_oc0 = channel_tile_id * channel_block + output_channel_lane;
                const int64_t local_oc1 = local_oc0 + tile;
                const int64_t oc0 = group * desc.output_channels_per_group + local_oc0;
                const int64_t oc1 = group * desc.output_channels_per_group + local_oc1;
                const bool spatial_active = spatial < output_volume;
                const bool channel_active0 = local_oc0 < desc.output_channels_per_group;
                const bool channel_active1 = local_oc1 < desc.output_channels_per_group;
                const int64_t ox = spatial % desc.output_size[0];
                const int64_t rem = spatial / desc.output_size[0];
                const int64_t oy = rem % desc.output_size[1];
                const int64_t oz = rem / desc.output_size[1];
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                if (spatial_active && bias) {
                    if (channel_active0) {
                        sum0 = bias_type == 0 ? static_cast<const float*>(bias)[oc0]
                                              : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc0]);
                    }
                    if (channel_active1) {
                        sum1 = bias_type == 0 ? static_cast<const float*>(bias)[oc1]
                                              : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc1]);
                    }
                }
                const int64_t reduction = desc.kernel_volume * desc.input_channels_per_group;
                const int64_t row_channels = Transposed
                    ? desc.output_channels_per_group : desc.input_channels_per_group;
                const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                    ? desc.kernel_volume * row_channels : row_channels;
                const bool pointwise = desc.kernel_volume == 1 && params.stride[0] == 1 &&
                    params.stride[1] == 1 && params.stride[2] == 1 &&
                    params.padding_before[0] == 0 && params.padding_before[1] == 0 &&
                    params.padding_before[2] == 0;
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
                            input_value = static_cast<float>(*reinterpret_cast<const ValueT*>(input_address));
                        }
                    }
                    input_tile[spatial_lane * stride + output_channel_lane] = input_value;
                    const int64_t weight_reduction = reduction_base + spatial_lane;
                    float weight_value0 = 0.0f;
                    float weight_value1 = 0.0f;
                    if ((channel_active0 || channel_active1) && weight_reduction < reduction) {
                        const int64_t kernel = (pointwise || common_3x3)
                            ? tile_kernel : weight_reduction / desc.input_channels_per_group;
                        const int64_t local_ic = (pointwise || common_3x3)
                            ? tile_channel_base + spatial_lane : weight_reduction % desc.input_channels_per_group;
                        const int64_t weight_outer = Transposed
                            ? group * desc.input_channels_per_group + local_ic : oc0;
                        if (channel_active0) {
                            const int64_t weight_inner = Transposed ? local_oc0 : local_ic;
                            weight_value0 = conv_nd_weight_sycl<QuantType, WeightT>(
                                weight, kernel, weight_outer, weight_inner, row_elements, desc.kernel_volume,
                                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                        }
                        if (channel_active1) {
                            const int64_t second_outer = Transposed
                                ? weight_outer : oc1;
                            const int64_t weight_inner = Transposed ? local_oc1 : local_ic;
                            weight_value1 = conv_nd_weight_sycl<QuantType, WeightT>(
                                weight, kernel, second_outer, weight_inner, row_elements, desc.kernel_volume,
                                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                        }
                    }
                    weight_tile[output_channel_lane * stride + spatial_lane] = weight_value0;
                    weight_tile[tile * stride + output_channel_lane * stride + spatial_lane] = weight_value1;
                    item.barrier(::sycl::access::fence_space::local_space);
                    if (spatial_active && (channel_active0 || channel_active1)) {
                        const int64_t remaining = reduction - reduction_base;
                        const int count = remaining < tile ? static_cast<int>(remaining) : tile;
                        for (int k = 0; k < count; ++k) {
                            const float input_value = input_tile[spatial_lane * stride + k];
                            if (channel_active0) {
                                sum0 += input_value * weight_tile[output_channel_lane * stride + k];
                            }
                            if (channel_active1) {
                                sum1 += input_value * weight_tile[
                                    tile * stride + output_channel_lane * stride + k];
                            }
                        }
                    }
                    item.barrier(::sycl::access::fence_space::local_space);
                }
                if (!spatial_active) return;
                const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
                if (channel_active0) {
                    char* output_address = desc.spatial_dims == 2
                        ? reinterpret_cast<char*>(destination) + batch * output_nb3 + oc0 * output_nb2 + oy * output_nb1 + ox * output_nb0
                        : reinterpret_cast<char*>(destination) + batch * output_nb2 + oc0 * output_nb1 + output_flat * output_nb0;
                    *reinterpret_cast<ValueT*>(output_address) = static_cast<ValueT>(sum0);
                }
                if (channel_active1) {
                    char* output_address = desc.spatial_dims == 2
                        ? reinterpret_cast<char*>(destination) + batch * output_nb3 + oc1 * output_nb2 + oy * output_nb1 + ox * output_nb0
                        : reinterpret_cast<char*>(destination) + batch * output_nb2 + oc1 * output_nb1 + output_flat * output_nb0;
                    *reinterpret_cast<ValueT*>(output_address) = static_cast<ValueT>(sum1);
                }
            });
    });
}

template <int QuantType, typename WeightT, typename ValueT, int Dims>
void launch_conv_transpose_nd_phase_gemm_impl_sycl(
    ::sycl::queue* queue, const ops_conv_nd_node_params& values,
    const ops_conv_nd_desc& desc, ggml_tensor* output
) {
    constexpr int tile = 16;
    constexpr int outputs_per_lane = 4;
    constexpr int channel_block = outputs_per_lane * tile;
    constexpr int local_stride = tile + 1;
    constexpr size_t local_size = tile * tile;
    const auto params = values.encoded;
    const int64_t channel_tiles = (desc.output_channels_per_group + channel_block - 1) / channel_block;
    const int64_t phase_count = static_cast<int64_t>(params.stride[0]) * params.stride[1] *
        (Dims == 3 ? params.stride[2] : 1);
    const int64_t max_phase_volume = ((desc.output_size[0] + params.stride[0] - 1) / params.stride[0]) *
                                     ((desc.output_size[1] + params.stride[1] - 1) / params.stride[1]) *
                                     (Dims == 3
                                          ? (desc.output_size[2] + params.stride[2] - 1) / params.stride[2]
                                          : 1);
    const int64_t spatial_tiles = (max_phase_volume + tile - 1) / tile;
    const int64_t total_groups = desc.batch * desc.groups * phase_count * channel_tiles * spatial_tiles;
    const void* weight = values.weight->data;
    const ValueT* input = static_cast<const ValueT*>(values.input->data);
    const void* bias = values.bias ? values.bias->data : nullptr;
    ValueT* destination = static_cast<ValueT*>(output->data);
    const int bias_type = !values.bias || values.bias->type == GGML_TYPE_F32 ? 0 : 1;
    const size_t weight_nb0 = values.weight->nb[0], weight_nb1 = values.weight->nb[1],
                 weight_nb2 = values.weight->nb[2];
    const size_t input_nb0 = values.input->nb[0], input_nb1 = values.input->nb[1];
    const size_t input_nb2 = values.input->nb[2], input_nb3 = values.input->nb[3];
    const size_t output_nb0 = output->nb[0], output_nb1 = output->nb[1];
    const size_t output_nb2 = output->nb[2], output_nb3 = output->nb[3];
    queue->submit([&](::sycl::handler& handler) {
        ::sycl::local_accessor<float, 1> input_tile(::sycl::range<1>(tile * local_stride), handler);
        ::sycl::local_accessor<float, 1> weight_tile(::sycl::range<1>(outputs_per_lane * tile * local_stride), handler);
        const ::sycl::nd_range<1> launch_range(::sycl::range<1>(total_groups * local_size),
                                               ::sycl::range<1>(local_size));
        const auto kernel = [=](::sycl::nd_item<1> item) __attribute__((always_inline)) {
            const int lane = static_cast<int>(item.get_local_linear_id());
            const int spatial_lane = lane & (tile - 1);
            const int output_channel_lane = lane / tile;
            int64_t group_linear = item.get_group_linear_id();
            const int64_t spatial_tile_id = group_linear % spatial_tiles;
            group_linear /= spatial_tiles;
            const int64_t channel_tile_id = group_linear % channel_tiles;
            group_linear /= channel_tiles;
            const int64_t phase = group_linear % phase_count;
            group_linear /= phase_count;
            const int64_t group = group_linear % desc.groups;
            const int64_t batch = group_linear / desc.groups;
            const int64_t phase_x = phase % params.stride[0];
            const int64_t phase_rem = phase / params.stride[0];
            const int64_t phase_y = Dims == 2 ? phase_rem : phase_rem % params.stride[1];
            const int64_t phase_z = Dims == 3 ? phase_rem / params.stride[1] : 0;
            const int64_t padding_x_mod = params.padding_before[0] % params.stride[0];
            const int64_t padding_y_mod = params.padding_before[1] % params.stride[1];
            const int64_t padding_z_mod = Dims == 3 ? params.padding_before[2] % params.stride[2] : 0;
            const int64_t first_x = (phase_x - padding_x_mod + params.stride[0]) % params.stride[0];
            const int64_t first_y = (phase_y - padding_y_mod + params.stride[1]) % params.stride[1];
            const int64_t first_z = Dims == 3
                ? (phase_z - padding_z_mod + params.stride[2]) % params.stride[2] : 0;
            const int64_t phase_width =
                first_x < desc.output_size[0] ? (desc.output_size[0] - 1 - first_x) / params.stride[0] + 1 : 0;
            const int64_t phase_height =
                first_y < desc.output_size[1] ? (desc.output_size[1] - 1 - first_y) / params.stride[1] + 1 : 0;
            const int64_t phase_depth = Dims == 3
                ? (first_z < desc.output_size[2]
                       ? (desc.output_size[2] - 1 - first_z) / params.stride[2] + 1 : 0)
                : 1;
            const int64_t phase_volume = phase_width * phase_height * phase_depth;
            const int64_t packed_spatial = spatial_tile_id * tile + spatial_lane;
            const bool spatial_active = packed_spatial < phase_volume;
            const int64_t packed_x = phase_width > 0 ? packed_spatial % phase_width : 0;
            const int64_t packed_rem = phase_width > 0 ? packed_spatial / phase_width : 0;
            const int64_t packed_y = Dims == 2
                ? packed_rem : (phase_height > 0 ? packed_rem % phase_height : 0);
            const int64_t packed_z = Dims == 3 && phase_height > 0 ? packed_rem / phase_height : 0;
            const int64_t ox = first_x + packed_x * params.stride[0];
            const int64_t oy = first_y + packed_y * params.stride[1];
            const int64_t oz = Dims == 3 ? first_z + packed_z * params.stride[2] : 0;
            const int64_t local_oc_base = channel_tile_id * channel_block + output_channel_lane;
            int64_t local_oc[outputs_per_lane];
            int64_t oc[outputs_per_lane];
            bool channel_active[outputs_per_lane];
            float sum[outputs_per_lane] = {};
#pragma unroll
            for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                local_oc[output_slot] = local_oc_base + output_slot * tile;
                oc[output_slot] = group * desc.output_channels_per_group + local_oc[output_slot];
                channel_active[output_slot] = local_oc[output_slot] < desc.output_channels_per_group;
            }
            if (spatial_active && bias) {
#pragma unroll
                for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                    if (channel_active[output_slot]) {
                        sum[output_slot] =
                            bias_type == 0
                                ? static_cast<const float*>(bias)[oc[output_slot]]
                                : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc[output_slot]]);
                    }
                }
            }

            const int64_t kernel_x_count =
                phase_x < desc.kernel_size[0] ? (desc.kernel_size[0] - 1 - phase_x) / params.stride[0] + 1 : 0;
            const int64_t kernel_y_count =
                phase_y < desc.kernel_size[1] ? (desc.kernel_size[1] - 1 - phase_y) / params.stride[1] + 1 : 0;
            const int64_t kernel_z_count = Dims == 3
                ? (phase_z < desc.kernel_size[2]
                       ? (desc.kernel_size[2] - 1 - phase_z) / params.stride[2] + 1 : 0)
                : 1;
            const int64_t phase_kernel_count = kernel_x_count * kernel_y_count * kernel_z_count;
            const int64_t reduction = phase_kernel_count * desc.input_channels_per_group;
            const bool channel_tiles_aligned =
                (Dims == 3 || !std::is_same_v<ValueT, ::sycl::half>) &&
                desc.input_channels_per_group % tile == 0;
            const int64_t row_elements = desc.weight_layout == ops_weight_layout::flattened_rows
                                             ? desc.output_channels_per_group * desc.kernel_volume
                                             : desc.output_channels_per_group;
            for (int64_t reduction_base = 0; reduction_base < reduction; reduction_base += tile) {
                const int64_t tile_phase_kernel =
                    channel_tiles_aligned ? reduction_base / desc.input_channels_per_group : -1;
                const int64_t tile_channel_base =
                    channel_tiles_aligned ? reduction_base - tile_phase_kernel * desc.input_channels_per_group : -1;
                const int64_t tile_kx =
                    channel_tiles_aligned ? phase_x + (tile_phase_kernel % kernel_x_count) * params.stride[0] : -1;
                const int64_t tile_ky =
                    channel_tiles_aligned
                        ? phase_y + (Dims == 2
                              ? tile_phase_kernel / kernel_x_count
                              : (tile_phase_kernel / kernel_x_count) % kernel_y_count) * params.stride[1]
                        : -1;
                const int64_t tile_kz =
                    Dims == 3 && channel_tiles_aligned
                        ? phase_z + (tile_phase_kernel / (kernel_x_count * kernel_y_count)) * params.stride[2]
                        : 0;
                const int64_t input_reduction = reduction_base + output_channel_lane;
                float input_value = 0.0f;
                if (spatial_active && input_reduction < reduction) {
                    const int64_t phase_kernel =
                        channel_tiles_aligned ? tile_phase_kernel : input_reduction / desc.input_channels_per_group;
                    const int64_t local_ic = channel_tiles_aligned ? tile_channel_base + output_channel_lane
                                                                   : input_reduction % desc.input_channels_per_group;
                    const int64_t kx =
                        channel_tiles_aligned ? tile_kx : phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
                    const int64_t ky =
                        channel_tiles_aligned
                            ? tile_ky
                            : phase_y + (Dims == 2
                                  ? phase_kernel / kernel_x_count
                                  : (phase_kernel / kernel_x_count) % kernel_y_count) * params.stride[1];
                    const int64_t kz = Dims == 3
                        ? (channel_tiles_aligned
                               ? tile_kz
                               : phase_z + (phase_kernel / (kernel_x_count * kernel_y_count)) * params.stride[2])
                        : 0;
                    const int64_t ix = (ox + params.padding_before[0] - kx) / params.stride[0];
                    const int64_t iy = (oy + params.padding_before[1] - ky) / params.stride[1];
                    const int64_t iz = Dims == 3
                        ? (oz + params.padding_before[2] - kz) / params.stride[2] : 0;
                    if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1] &&
                        (Dims == 2 || (iz >= 0 && iz < desc.input_size[2]))) {
                        const int64_t ic = group * desc.input_channels_per_group + local_ic;
                        const char* address;
                        if constexpr (Dims == 2) {
                            address = reinterpret_cast<const char*>(input) + batch * input_nb3 +
                                ic * input_nb2 + iy * input_nb1 + ix * input_nb0;
                        } else {
                            const int64_t input_flat = ix + desc.input_size[0] * (iy + desc.input_size[1] * iz);
                            address = reinterpret_cast<const char*>(input) + batch * input_nb2 +
                                ic * input_nb1 + input_flat * input_nb0;
                        }
                        input_value = static_cast<float>(*reinterpret_cast<const ValueT*>(address));
                    }
                }
                input_tile[spatial_lane * local_stride + output_channel_lane] = input_value;

                const int64_t weight_reduction = reduction_base + spatial_lane;
                float weight_value[outputs_per_lane] = {};
                if (weight_reduction < reduction) {
                    const int64_t phase_kernel =
                        channel_tiles_aligned ? tile_phase_kernel : weight_reduction / desc.input_channels_per_group;
                    const int64_t local_ic = channel_tiles_aligned ? tile_channel_base + spatial_lane
                                                                   : weight_reduction % desc.input_channels_per_group;
                    const int64_t kx =
                        channel_tiles_aligned ? tile_kx : phase_x + (phase_kernel % kernel_x_count) * params.stride[0];
                    const int64_t ky =
                        channel_tiles_aligned
                            ? tile_ky
                            : phase_y + (Dims == 2
                                  ? phase_kernel / kernel_x_count
                                  : (phase_kernel / kernel_x_count) % kernel_y_count) * params.stride[1];
                    const int64_t kz = Dims == 3
                        ? (channel_tiles_aligned
                               ? tile_kz
                               : phase_z + (phase_kernel / (kernel_x_count * kernel_y_count)) * params.stride[2])
                        : 0;
                    const int64_t kernel = kx + desc.kernel_size[0] * (ky + desc.kernel_size[1] * kz);
                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
#pragma unroll
                    for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                        if (channel_active[output_slot]) {
                            weight_value[output_slot] = conv_nd_weight_sycl<QuantType, WeightT>(
                                weight, kernel, ic, local_oc[output_slot], row_elements, desc.kernel_volume,
                                desc.weight_layout, weight_nb0, weight_nb1, weight_nb2);
                        }
                    }
                }
#pragma unroll
                for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                    weight_tile[(output_slot * tile + output_channel_lane) * local_stride + spatial_lane] =
                        weight_value[output_slot];
                }
                item.barrier(::sycl::access::fence_space::local_space);
                if (spatial_active) {
                    const int64_t remaining = reduction - reduction_base;
                    const int count = remaining < tile ? static_cast<int>(remaining) : tile;
                    for (int k = 0; k < count; ++k) {
                        const float input_value = input_tile[spatial_lane * local_stride + k];
#pragma unroll
                        for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                            if (channel_active[output_slot]) {
                                sum[output_slot] +=
                                    input_value *
                                    weight_tile[(output_slot * tile + output_channel_lane) * local_stride + k];
                            }
                        }
                    }
                }
                item.barrier(::sycl::access::fence_space::local_space);
            }
            if (!spatial_active) return;
#pragma unroll
            for (int output_slot = 0; output_slot < outputs_per_lane; ++output_slot) {
                if (channel_active[output_slot]) {
                    char* address;
                    if constexpr (Dims == 2) {
                        address = reinterpret_cast<char*>(destination) + batch * output_nb3 +
                            oc[output_slot] * output_nb2 + oy * output_nb1 + ox * output_nb0;
                    } else {
                        const int64_t output_flat = ox + desc.output_size[0] * (oy + desc.output_size[1] * oz);
                        address = reinterpret_cast<char*>(destination) + batch * output_nb2 +
                            oc[output_slot] * output_nb1 + output_flat * output_nb0;
                    }
                    *reinterpret_cast<ValueT*>(address) = static_cast<ValueT>(sum[output_slot]);
                }
            }
        };
        if constexpr (std::is_same_v<ValueT, ::sycl::half>) {
            handler.parallel_for<ConvTransposeNDPhaseGemmKernel<QuantType, WeightT, ValueT, Dims>>(
                launch_range, [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] { kernel(item); });
        } else {
            handler.parallel_for<ConvTransposeNDPhaseGemmKernel<QuantType, WeightT, ValueT, Dims>>(
                launch_range, kernel);
        }
    });
}

template <int QuantType, typename WeightT, typename ValueT>
void launch_conv_transpose_nd_phase_gemm_sycl(
    ::sycl::queue* queue, const ops_conv_nd_node_params& values,
    const ops_conv_nd_desc& desc, ggml_tensor* output
) {
    if (desc.spatial_dims == 2) {
        launch_conv_transpose_nd_phase_gemm_impl_sycl<QuantType, WeightT, ValueT, 2>(
            queue, values, desc, output);
    } else {
        launch_conv_transpose_nd_phase_gemm_impl_sycl<QuantType, WeightT, ValueT, 3>(
            queue, values, desc, output);
    }
}

template <typename ValueT>
bool dispatch_conv_nd_weight_sycl(::sycl::queue* queue, bool transposed,
    const ops_conv_nd_node_params& params, const ops_conv_nd_desc& desc, ggml_tensor* output) {
    if constexpr (std::is_same_v<ValueT, float>) {
        if (params.weight->type == GGML_TYPE_F32 &&
            launch_pointwise_conv_nd_mkl_sycl<float>(queue, transposed, params, desc, output)) return true;
    } else if constexpr (std::is_same_v<ValueT, ::sycl::half>) {
        if (params.weight->type == GGML_TYPE_F16 &&
            launch_pointwise_conv_nd_mkl_sycl<::sycl::half>(queue, transposed, params, desc, output)) return true;
    }
#define LAUNCH(Q, W) do { \
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2]; \
    const bool use_implicit = !transposed && output_volume >= 64 && \
        desc.input_channels_per_group >= 16 && desc.output_channels_per_group >= 16; \
    const bool use_transposed_pointwise = transposed && output_volume >= 64 && desc.kernel_volume == 1 && \
        desc.input_channels_per_group >= 16 && desc.output_channels_per_group >= 16 && \
        params.encoded.stride[0] == 1 && params.encoded.stride[1] == 1 && params.encoded.stride[2] == 1 && \
        params.encoded.padding_before[0] == 0 && params.encoded.padding_before[1] == 0 && \
        params.encoded.padding_before[2] == 0; \
    const int64_t max_phase_volume = \
        ((desc.output_size[0] + params.encoded.stride[0] - 1) / params.encoded.stride[0]) * \
        ((desc.output_size[1] + params.encoded.stride[1] - 1) / params.encoded.stride[1]) * \
        ((desc.output_size[2] + params.encoded.stride[2] - 1) / params.encoded.stride[2]); \
    const bool use_transposed_phase = transposed && \
        params.encoded.dilation[0] == 1 && params.encoded.dilation[1] == 1 && \
        params.encoded.dilation[2] == 1 && \
        (params.encoded.stride[0] > 1 || params.encoded.stride[1] > 1 || params.encoded.stride[2] > 1) && \
        max_phase_volume >= 16 && desc.input_channels_per_group >= 16 && \
        desc.output_channels_per_group >= 16; \
    if (use_implicit) launch_conv_nd_implicit_gemm_sycl<Q, W, ValueT, false>(queue, params, desc, output); \
    else if (use_transposed_pointwise) launch_conv_nd_implicit_gemm_sycl<Q, W, ValueT, true>(queue, params, desc, output); \
    else if (use_transposed_phase) launch_conv_transpose_nd_phase_gemm_sycl<Q, W, ValueT>(queue, params, desc, output); \
    else if (transposed) launch_conv_nd_sycl<Q, W, ValueT, true>(queue, params, desc, output); \
    else launch_conv_nd_sycl<Q, W, ValueT, false>(queue, params, desc, output); \
} while (0)
    switch (params.weight->type) {
        case GGML_TYPE_F32: LAUNCH(-1, float); break;
        case GGML_TYPE_F16: LAUNCH(-1, ::sycl::half); break;
        case GGML_TYPE_Q4_0: LAUNCH(GGML_TYPE_Q4_0, void); break;
        case GGML_TYPE_Q4_1: LAUNCH(GGML_TYPE_Q4_1, void); break;
        case GGML_TYPE_Q5_0: LAUNCH(GGML_TYPE_Q5_0, void); break;
        case GGML_TYPE_Q5_1: LAUNCH(GGML_TYPE_Q5_1, void); break;
        case GGML_TYPE_Q8_0: LAUNCH(GGML_TYPE_Q8_0, void); break;
        case GGML_TYPE_Q2_K: LAUNCH(GGML_TYPE_Q2_K, void); break;
        case GGML_TYPE_Q3_K: LAUNCH(GGML_TYPE_Q3_K, void); break;
        case GGML_TYPE_Q4_K: LAUNCH(GGML_TYPE_Q4_K, void); break;
        case GGML_TYPE_Q5_K: LAUNCH(GGML_TYPE_Q5_K, void); break;
        case GGML_TYPE_Q6_K: LAUNCH(GGML_TYPE_Q6_K, void); break;
        case GGML_TYPE_IQ4_NL: LAUNCH(GGML_TYPE_IQ4_NL, void); break;
        case GGML_TYPE_IQ4_XS: LAUNCH(GGML_TYPE_IQ4_XS, void); break;
        case GGML_TYPE_MXFP4: LAUNCH(GGML_TYPE_MXFP4, void); break;
        default: return false;
    }
#undef LAUNCH
    return true;
}

bool ggml_sycl_op_conv_nd_entry(ggml_backend_t backend, ggml_tensor* node) {
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
    auto* queue = static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
    if (!queue) return false;
    if (params.input->type == GGML_TYPE_F32) return dispatch_conv_nd_weight_sycl<float>(queue, transposed, params, desc, node);
    if (params.input->type == GGML_TYPE_F16) return dispatch_conv_nd_weight_sycl<::sycl::half>(queue, transposed, params, desc, node);
    return false;
}

} // namespace ggml_ops_ext::sycl
