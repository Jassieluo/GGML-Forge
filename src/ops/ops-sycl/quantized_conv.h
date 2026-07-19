#pragma once

namespace ggml_ops_ext {
namespace sycl {

template <int Type>
inline float load_quantized_row_value_sycl(const void* data, int64_t row, int64_t column, int64_t row_elements) {
    if constexpr (Type == GGML_TYPE_Q4_0) {
        const int64_t blocks_per_row = row_elements / QK4_0;
        const block_q4_0& block = static_cast<const block_q4_0*>(data)[row * blocks_per_row + column / QK4_0];
        const int index = static_cast<int>(column % QK4_0);
        const uint8_t packed = block.qs[index % (QK4_0 / 2)];
        const int quant = ((index < QK4_0 / 2 ? packed : packed >> 4) & 0x0f) - 8;
        return static_cast<float>(block.d) * quant;
    } else if constexpr (Type == GGML_TYPE_Q4_1) {
        const int64_t blocks_per_row = row_elements / QK4_1;
        const block_q4_1& block = static_cast<const block_q4_1*>(data)[row * blocks_per_row + column / QK4_1];
        const int index = static_cast<int>(column % QK4_1);
        const uint8_t packed = block.qs[index % 16];
        const int quant = index < 16 ? packed & 0x0f : packed >> 4;
        return static_cast<float>(block.dm[0]) * quant + static_cast<float>(block.dm[1]);
    } else if constexpr (Type == GGML_TYPE_Q5_0 || Type == GGML_TYPE_Q5_1) {
        constexpr int block_size = Type == GGML_TYPE_Q5_0 ? QK5_0 : QK5_1;
        const int64_t blocks_per_row = row_elements / block_size;
        const int index = static_cast<int>(column % block_size);
        if constexpr (Type == GGML_TYPE_Q5_0) {
            const block_q5_0& block = static_cast<const block_q5_0*>(data)[row * blocks_per_row + column / block_size];
            uint32_t high = 0;
            for (int i = 0; i < 4; ++i) high |= uint32_t(block.qh[i]) << (8 * i);
            const uint8_t packed = block.qs[index % 16];
            const int quant = (index < 16 ? packed & 0x0f : packed >> 4) | (((high >> index) & 1) << 4);
            return static_cast<float>(block.d) * (quant - 16);
        } else {
            const block_q5_1& block = static_cast<const block_q5_1*>(data)[row * blocks_per_row + column / block_size];
            uint32_t high = 0;
            for (int i = 0; i < 4; ++i) high |= uint32_t(block.qh[i]) << (8 * i);
            const uint8_t packed = block.qs[index % 16];
            const int quant = (index < 16 ? packed & 0x0f : packed >> 4) | (((high >> index) & 1) << 4);
            return static_cast<float>(block.dm[0]) * quant + static_cast<float>(block.dm[1]);
        }
    } else if constexpr (Type == GGML_TYPE_Q8_0) {
        const int64_t blocks_per_row = row_elements / QK8_0;
        const block_q8_0& block = static_cast<const block_q8_0*>(data)[row * blocks_per_row + column / QK8_0];
        return static_cast<float>(block.d) * block.qs[column % QK8_0];
    } else if constexpr (Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q5_K) {
        const int64_t blocks_per_row = row_elements / QK_K;
        const int index = static_cast<int>(column % QK_K);
        const int group = index / 32;
        if constexpr (Type == GGML_TYPE_Q4_K) {
        const block_q4_K& block = static_cast<const block_q4_K*>(data)[row * blocks_per_row + column / QK_K];
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
        return static_cast<float>(block.dm.x()) * scale * quant -
               static_cast<float>(block.dm.y()) * minimum;
        } else {
            const block_q5_K& block = static_cast<const block_q5_K*>(data)[row * blocks_per_row + column / QK_K];
            uint8_t scale, minimum;
            if (group < 4) { scale = block.scales[group] & 63; minimum = block.scales[group + 4] & 63; }
            else { scale = (block.scales[group + 4] & 0x0f) | ((block.scales[group - 4] >> 6) << 4);
                   minimum = (block.scales[group + 4] >> 4) | ((block.scales[group] >> 6) << 4); }
            const int il = index / 64, in = index % 64, ir = (in & 31) / 2, iq = in & 1;
            const uint8_t packed = block.qs[32 * il + 2 * ir + iq];
            const uint8_t high = block.qh[2 * ir + iq];
            const int quant = (in >= 32 ? packed >> 4 : packed & 0x0f) +
                              ((high & (1 << (2 * il + (in >= 32 ? 1 : 0)))) ? 16 : 0);
            return static_cast<float>(block.dm[0]) * scale * quant - static_cast<float>(block.dm[1]) * minimum;
        }
    } else if constexpr (Type == GGML_TYPE_Q2_K) {
        const int64_t blocks_per_row = row_elements / QK_K;
        const block_q2_K& block = static_cast<const block_q2_K*>(data)[row * blocks_per_row + column / QK_K];
        const int index = static_cast<int>(column % QK_K), n = index / 128, r = index % 128;
        const int group = r / 32, lane = r % 32, scale_index = 8 * n + lane / 16;
        const uint8_t packed = block.qs[32 * n + lane], scale = block.scales[scale_index + 2 * group];
        return static_cast<float>(block.dm[0]) * (scale & 15) * ((packed >> (2 * group)) & 3) -
               static_cast<float>(block.dm[1]) * (scale >> 4);
    } else if constexpr (Type == GGML_TYPE_Q3_K) {
        const int64_t blocks_per_row = row_elements / QK_K;
        const block_q3_K& block = static_cast<const block_q3_K*>(data)[row * blocks_per_row + column / QK_K];
        const int index = static_cast<int>(column % QK_K), n = index / 128, r = index % 128;
        const int j = r / 32, lane = r % 32, is = 8 * n + 2 * j + lane / 16;
        const uint8_t us = is < 4 ? (block.scales[is] & 15) | ((block.scales[is + 8] & 3) << 4) :
                           is < 8 ? (block.scales[is] & 15) | (((block.scales[is + 4] >> 2) & 3) << 4) :
                           is < 12 ? (block.scales[is - 8] >> 4) | (((block.scales[is] >> 4) & 3) << 4) :
                                     (block.scales[is - 8] >> 4) | (((block.scales[is - 4] >> 6) & 3) << 4);
        const int quant = ((block.qs[32 * n + lane] >> (2 * j)) & 3) -
                          ((block.hmask[lane] & (1 << (4 * n + j))) ? 0 : 4);
        return static_cast<float>(block.d) * (static_cast<int>(us) - 32) * quant;
    } else if constexpr (Type == GGML_TYPE_IQ4_NL) {
        const int64_t blocks_per_row = row_elements / QK4_NL;
        const block_iq4_nl& block = static_cast<const block_iq4_nl*>(data)[row * blocks_per_row + column / QK4_NL];
        const int index = static_cast<int>(column % QK4_NL);
        const uint8_t packed = block.qs[index % 16], quant = index < 16 ? packed & 15 : packed >> 4;
        return static_cast<float>(block.d) * kvalues_iq4nl[quant];
    } else if constexpr (Type == GGML_TYPE_IQ4_XS) {
        const int64_t blocks_per_row = row_elements / QK_K;
        const block_iq4_xs& block = static_cast<const block_iq4_xs*>(data)[row * blocks_per_row + column / QK_K];
        const int index = static_cast<int>(column % QK_K), group = index / 32, within = index % 32;
        const uint8_t packed = block.qs[16 * group + within % 16], quant = within < 16 ? packed & 15 : packed >> 4;
        const int scale = ((block.scales_l[group / 2] >> (4 * (group % 2))) & 15) |
                          (((block.scales_h >> (2 * group)) & 3) << 4);
        return static_cast<float>(block.d) * (scale - 32) * kvalues_iq4nl[quant];
    } else if constexpr (Type == GGML_TYPE_MXFP4) {
        const int64_t blocks_per_row = row_elements / QK_MXFP4;
        const block_mxfp4& block = static_cast<const block_mxfp4*>(data)[row * blocks_per_row + column / QK_MXFP4];
        const int index = static_cast<int>(column % QK_MXFP4);
        const uint8_t packed = block.qs[index % 16], quant = index < 16 ? packed & 15 : packed >> 4;
        uint32_t bits = block.e == 0 ? 0x00400000u : uint32_t(block.e) << 23;
        float scale = ::sycl::bit_cast<float>(bits);
        return scale * kvalues_mxfp4[quant] * 0.5f;
    } else {
        static_assert(Type == GGML_TYPE_Q6_K);
        const int64_t blocks_per_row = row_elements / QK_K;
        const block_q6_K& block = static_cast<const block_q6_K*>(data)[row * blocks_per_row + column / QK_K];
        const int index = static_cast<int>(column % QK_K), ip = index / 128, in = index % 128;
        const int lane = in & 31, group = in / 32, scale_index = 8 * ip + lane / 16;
        const uint8_t low0 = block.ql[64 * ip + lane], low1 = block.ql[64 * ip + lane + 32];
        const uint8_t high = block.qh[32 * ip + lane];
        int quant, scale;
        if (group == 0) { quant = (low0 & 15) | (((high >> 0) & 3) << 4); scale = block.scales[scale_index]; }
        else if (group == 1) { quant = (low1 & 15) | (((high >> 2) & 3) << 4); scale = block.scales[scale_index + 2]; }
        else if (group == 2) { quant = (low0 >> 4) | (((high >> 4) & 3) << 4); scale = block.scales[scale_index + 4]; }
        else { quant = (low1 >> 4) | (((high >> 6) & 3) << 4); scale = block.scales[scale_index + 6]; }
        return static_cast<float>(block.d) * scale * (quant - 32);
    }
}

template <int WeightType, typename T>
class QuantizedConv1dDirectKernel;

template <int WeightType, typename T>
class QuantizedConv1dCachedInputKernel;

template <int WeightType, typename T>
void launch_quantized_conv_1d_direct_sycl(
    ::sycl::queue* queue, const void* weight, const T* input, const void* bias, int bias_type, T* output,
    int64_t input_length, int64_t output_length, int64_t input_channels, int64_t output_channels,
    int64_t kernel, int64_t batch, int stride, int padding, int dilation, int groups,
    size_t input_nb0, size_t input_nb1, size_t input_nb2,
    size_t output_nb0, size_t output_nb1, size_t output_nb2
) {
    constexpr int64_t subgroup_size = 32;
    constexpr int64_t subgroups_per_workgroup = 8;
    constexpr int64_t time_tile = 8;
    constexpr int64_t local_size = subgroup_size * subgroups_per_workgroup;
    const int64_t time_tiles = (output_length + time_tile - 1) / time_tile;
    const int64_t channel_tiles = (output_channels + subgroups_per_workgroup - 1) / subgroups_per_workgroup;
    const int64_t workgroups = batch * channel_tiles * time_tiles;
    const int64_t global_size = workgroups * local_size;
    const int64_t cached_input_elements = input_channels * time_tile;
    if (groups == 1 && cached_input_elements <= 8192) {
        queue->submit([&](::sycl::handler& cgh) {
            ::sycl::local_accessor<float, 1> input_tile(
                ::sycl::range<1>(cached_input_elements), cgh);
            cgh.parallel_for<QuantizedConv1dCachedInputKernel<WeightType, T>>(
                ::sycl::nd_range<1>(global_size, local_size),
                [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                    const auto subgroup = item.get_sub_group();
                    const int64_t subgroup_id = item.get_local_id(0) / subgroup_size;
                    const int64_t workgroup = item.get_group(0);
                    const int64_t time_tile_index = workgroup % time_tiles;
                    const int64_t channel_batch = workgroup / time_tiles;
                    const int64_t channel_tile_index = channel_batch % channel_tiles;
                    const int64_t n = channel_batch / channel_tiles;
                    const int64_t oc = channel_tile_index * subgroups_per_workgroup + subgroup_id;
                    const int64_t lane = subgroup.get_local_id()[0];
                    const int64_t output_start = time_tile_index * time_tile;
                    float sums[time_tile] = {};

                    for (int64_t kw = 0; kw < kernel; ++kw) {
                        for (int64_t index = item.get_local_id(0); index < cached_input_elements; index += local_size) {
                            const int64_t t = index / input_channels;
                            const int64_t ic = index - t * input_channels;
                            const int64_t ow = output_start + t;
                            const int64_t iw = ow * stride - padding + kw * dilation;
                            float value = 0.0f;
                            if (ow < output_length && iw >= 0 && iw < input_length) {
                                value = static_cast<float>(*reinterpret_cast<const T*>(
                                    reinterpret_cast<const char*>(input) + n * input_nb2 +
                                    ic * input_nb1 + iw * input_nb0));
                            }
                            input_tile[index] = value;
                        }
                        item.barrier(::sycl::access::fence_space::local_space);

                        if (oc < output_channels) {
                            for (int64_t ic = lane; ic < input_channels; ic += subgroup_size) {
                                const float weight_value = load_quantized_row_value_sycl<WeightType>(
                                    weight, oc * kernel + kw, ic, input_channels);
                                #pragma unroll
                                for (int64_t t = 0; t < time_tile; ++t) {
                                    sums[t] += input_tile[t * input_channels + ic] * weight_value;
                                }
                            }
                        }
                        item.barrier(::sycl::access::fence_space::local_space);
                    }

                    if (oc < output_channels) {
                        #pragma unroll
                        for (int64_t t = 0; t < time_tile; ++t) {
                            sums[t] = ::sycl::reduce_over_group(subgroup, sums[t], ::sycl::plus<float>());
                        }
                        if (lane == 0) {
                            const float bias_value = bias
                                ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                                  : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc]))
                                : 0.0f;
                            #pragma unroll
                            for (int64_t t = 0; t < time_tile; ++t) {
                                const int64_t ow = output_start + t;
                                if (ow >= output_length) continue;
                                *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + n * output_nb2 +
                                    oc * output_nb1 + ow * output_nb0) = static_cast<T>(sums[t] + bias_value);
                            }
                        }
                    }
                });
        });
        return;
    }
    queue->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<QuantizedConv1dDirectKernel<WeightType, T>>(
            ::sycl::nd_range<1>(global_size, local_size), [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t subgroup_id = item.get_local_id(0) / subgroup_size;
                const int64_t workgroup = item.get_group(0);
                const int64_t time_tile_index = workgroup % time_tiles;
                const int64_t channel_batch = workgroup / time_tiles;
                const int64_t channel_tile_index = channel_batch % channel_tiles;
                const int64_t n = channel_batch / channel_tiles;
                const int64_t oc = channel_tile_index * subgroups_per_workgroup + subgroup_id;
                if (oc >= output_channels) return;
                const int64_t lane = subgroup.get_local_id()[0];
                const int64_t output_start = time_tile_index * time_tile;
                const int64_t input_per_group = input_channels / groups;
                const int64_t output_per_group = output_channels / groups;
                const int64_t group = oc / output_per_group;
                float sums[time_tile] = {};
                for (int64_t kw = 0; kw < kernel; ++kw) {
                    for (int64_t local_ic = lane; local_ic < input_per_group; local_ic += subgroup_size) {
                        const int64_t ic = group * input_per_group + local_ic;
                        const float weight_value = load_quantized_row_value_sycl<WeightType>(
                            weight, oc * kernel + kw, local_ic, input_per_group);
                        #pragma unroll
                        for (int64_t t = 0; t < time_tile; ++t) {
                            const int64_t ow = output_start + t;
                            const int64_t iw = ow * stride - padding + kw * dilation;
                            if (ow >= output_length || iw < 0 || iw >= input_length) continue;
                            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
                            sums[t] += static_cast<float>(value) * weight_value;
                        }
                    }
                }
                #pragma unroll
                for (int64_t t = 0; t < time_tile; ++t) {
                    sums[t] = ::sycl::reduce_over_group(subgroup, sums[t], ::sycl::plus<float>());
                }
                if (lane == 0) {
                    const float bias_value = bias
                        ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                          : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc]))
                        : 0.0f;
                    #pragma unroll
                    for (int64_t t = 0; t < time_tile; ++t) {
                        const int64_t ow = output_start + t;
                        if (ow >= output_length) continue;
                        *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + n * output_nb2 +
                            oc * output_nb1 + ow * output_nb0) = static_cast<T>(sums[t] + bias_value);
                    }
                }
            });
    });
}

template <int WeightType, typename T>
class QuantizedConvTranspose1dDirectKernel;

template <int WeightType, typename T>
class QuantizedConvTranspose1dScalarKernel;

template <int WeightType, typename T>
class QuantizedConvTranspose1dPhaseKernel;

template <int WeightType, typename T>
void launch_quantized_conv_transpose_1d_direct_sycl(
    ::sycl::queue* queue, const void* weight, const T* input, const void* bias, int bias_type, T* output,
    int64_t input_length, int64_t output_length, int64_t input_channels, int64_t output_channels_per_group,
    int64_t kernel, int64_t batch, int stride, int padding, int dilation, int groups,
    size_t input_nb0, size_t input_nb1, size_t input_nb2,
    size_t output_nb0, size_t output_nb1, size_t output_nb2
) {
    const int64_t output_channels = output_channels_per_group * groups;
    const int64_t total = batch * output_channels * output_length;
    const int64_t input_per_group = input_channels / groups;
    if (input_per_group <= 8) {
        queue->submit([&](::sycl::handler& cgh) {
            cgh.parallel_for<QuantizedConvTranspose1dScalarKernel<WeightType, T>>(
                ::sycl::range<1>(total), [=](::sycl::id<1> item) {
                    const int64_t linear = item[0];
                    const int64_t ow = linear % output_length;
                    const int64_t oc = (linear / output_length) % output_channels;
                    const int64_t n = linear / (output_length * output_channels);
                    const int64_t group = oc / output_channels_per_group;
                    const int64_t local_oc = oc % output_channels_per_group;
                    float sum = bias ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                        : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc])) : 0.0f;
                    for (int64_t local_ic = 0; local_ic < input_per_group; ++local_ic) {
                        const int64_t ic = group * input_per_group + local_ic;
                        for (int64_t kw = 0; kw < kernel; ++kw) {
                            const int64_t numerator = ow + padding - kw * dilation;
                            if (numerator < 0 || numerator % stride != 0) continue;
                            const int64_t iw = numerator / stride;
                            if (iw >= input_length) continue;
                            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
                            sum += static_cast<float>(value) * load_quantized_row_value_sycl<WeightType>(
                                weight, ic * kernel + kw, local_oc, output_channels_per_group);
                        }
                    }
                    *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + n * output_nb2 +
                        oc * output_nb1 + ow * output_nb0) = static_cast<T>(sum);
                });
        });
        return;
    }
    if (stride > 1 && dilation == 1) {
        constexpr int64_t subgroup_size = 16;
        constexpr int64_t subgroups_per_workgroup = 16;
        constexpr int64_t time_tile = 8;
        constexpr int64_t local_size = subgroup_size * subgroups_per_workgroup;
        const int64_t max_phase_length = (output_length + stride - 1) / stride;
        const int64_t phase_tiles = (max_phase_length + time_tile - 1) / time_tile;
        const int64_t channel_tiles =
            (output_channels + subgroups_per_workgroup - 1) / subgroups_per_workgroup;
        const int64_t workgroups = batch * channel_tiles * stride * phase_tiles;
        const int64_t max_phase_kernel_count = (kernel + stride - 1) / stride;
        const int64_t cached_input_elements = input_per_group * time_tile * max_phase_kernel_count;
        const bool use_input_cache = cached_input_elements <= 8192 &&
            output_channels_per_group % subgroups_per_workgroup == 0;
        queue->submit([&](::sycl::handler& cgh) {
            ::sycl::local_accessor<float, 1> input_cache(
                ::sycl::range<1>(use_input_cache ? cached_input_elements : 1), cgh);
            cgh.parallel_for<QuantizedConvTranspose1dPhaseKernel<WeightType, T>>(
                ::sycl::nd_range<1>(workgroups * local_size, local_size),
                [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                    const auto subgroup = item.get_sub_group();
                    const int64_t subgroup_id = item.get_local_id(0) / subgroup_size;
                    int64_t workgroup = item.get_group(0);
                    const int64_t phase_tile = workgroup % phase_tiles;
                    workgroup /= phase_tiles;
                    const int64_t phase = workgroup % stride;
                    workgroup /= stride;
                    const int64_t channel_tile = workgroup % channel_tiles;
                    const int64_t n = workgroup / channel_tiles;
                    const int64_t oc = channel_tile * subgroups_per_workgroup + subgroup_id;
                    const bool channel_active = oc < output_channels;
                    const int64_t lane = subgroup.get_local_id()[0];
                    const int64_t tile_first_oc = channel_tile * subgroups_per_workgroup;
                    const int64_t group = (channel_active ? oc : tile_first_oc) / output_channels_per_group;
                    const int64_t local_oc = channel_active ? oc % output_channels_per_group : 0;
                    const int64_t padding_mod = padding % stride;
                    const int64_t first_output = (phase - padding_mod + stride) % stride;
                    const int64_t phase_length = first_output < output_length
                        ? (output_length - 1 - first_output) / stride + 1 : 0;
                    const int64_t packed_start = phase_tile * time_tile;
                    float sums[time_tile] = {};

                    if (use_input_cache) {
                        for (int64_t index = item.get_local_id(0); index < cached_input_elements;
                             index += local_size) {
                            const int64_t phase_kernel = index / (input_per_group * time_tile);
                            const int64_t cache_rem = index % (input_per_group * time_tile);
                            const int64_t local_ic = cache_rem / time_tile;
                            const int64_t t = cache_rem % time_tile;
                            const int64_t kw = phase + phase_kernel * stride;
                            const int64_t packed = packed_start + t;
                            float value = 0.0f;
                            if (kw < kernel && packed < phase_length) {
                                const int64_t ow = first_output + packed * stride;
                                const int64_t numerator = ow + padding - kw;
                                const int64_t iw = numerator >= 0 ? numerator / stride : -1;
                                if (iw >= 0 && iw < input_length) {
                                    const int64_t ic = group * input_per_group + local_ic;
                                    value = static_cast<float>(*reinterpret_cast<const T*>(
                                        reinterpret_cast<const char*>(input) + n * input_nb2 +
                                        ic * input_nb1 + iw * input_nb0));
                                }
                            }
                            input_cache[index] = value;
                        }
                    }
                    if (use_input_cache) {
                        item.barrier(::sycl::access::fence_space::local_space);
                    }
                    if (!channel_active) return;

                    for (int64_t local_ic = lane; local_ic < input_per_group;
                         local_ic += subgroup_size) {
                        const int64_t ic = group * input_per_group + local_ic;
                        int64_t phase_kernel = 0;
                        for (int64_t kw = phase; kw < kernel; kw += stride, ++phase_kernel) {
                            const float weight_value = load_quantized_row_value_sycl<WeightType>(
                                weight, ic * kernel + kw, local_oc, output_channels_per_group);
#pragma unroll
                            for (int64_t t = 0; t < time_tile; ++t) {
                                const int64_t packed = packed_start + t;
                                if (packed >= phase_length) continue;
                                const int64_t ow = first_output + packed * stride;
                                const int64_t numerator = ow + padding - kw;
                                if (numerator < 0) continue;
                                const int64_t iw = numerator / stride;
                                if (iw >= input_length) continue;
                                const float value = use_input_cache
                                    ? input_cache[(phase_kernel * input_per_group + local_ic) * time_tile + t]
                                    : static_cast<float>(*reinterpret_cast<const T*>(
                                          reinterpret_cast<const char*>(input) + n * input_nb2 +
                                          ic * input_nb1 + iw * input_nb0));
                                sums[t] += value * weight_value;
                            }
                        }
                    }
#pragma unroll
                    for (int64_t t = 0; t < time_tile; ++t) {
                        sums[t] = ::sycl::reduce_over_group(subgroup, sums[t], ::sycl::plus<float>());
                    }
                    if (lane == 0) {
                        const float bias_value = bias
                            ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                              : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc]))
                            : 0.0f;
#pragma unroll
                        for (int64_t t = 0; t < time_tile; ++t) {
                            const int64_t packed = packed_start + t;
                            if (packed >= phase_length) continue;
                            const int64_t ow = first_output + packed * stride;
                            *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + n * output_nb2 +
                                oc * output_nb1 + ow * output_nb0) = static_cast<T>(sums[t] + bias_value);
                        }
                    }
                });
        });
        return;
    }
    constexpr int64_t subgroup_size = 16;
    constexpr int64_t subgroups_per_workgroup = 16;
    constexpr int64_t time_tile = 8;
    constexpr int64_t local_size = subgroup_size * subgroups_per_workgroup;
    const int64_t time_tiles = (output_length + time_tile - 1) / time_tile;
    const int64_t channel_tiles = (output_channels + subgroups_per_workgroup - 1) / subgroups_per_workgroup;
    const int64_t workgroups = batch * channel_tiles * time_tiles;
    const int64_t global_size = workgroups * local_size;
    queue->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for<QuantizedConvTranspose1dDirectKernel<WeightType, T>>(
            ::sycl::nd_range<1>(global_size, local_size), [=](::sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
                const auto subgroup = item.get_sub_group();
                const int64_t subgroup_id = item.get_local_id(0) / subgroup_size;
                const int64_t workgroup = item.get_group(0);
                const int64_t time_tile_index = workgroup % time_tiles;
                const int64_t channel_batch = workgroup / time_tiles;
                const int64_t channel_tile_index = channel_batch % channel_tiles;
                const int64_t n = channel_batch / channel_tiles;
                const int64_t oc = channel_tile_index * subgroups_per_workgroup + subgroup_id;
                if (oc >= output_channels) return;
                const int64_t lane = subgroup.get_local_id()[0];
                const int64_t output_start = time_tile_index * time_tile;
                const int64_t group = oc / output_channels_per_group;
                const int64_t local_oc = oc % output_channels_per_group;
                float sums[time_tile] = {};
                for (int64_t local_ic = lane; local_ic < input_per_group; local_ic += subgroup_size) {
                    const int64_t ic = group * input_per_group + local_ic;
                    for (int64_t kw = 0; kw < kernel; ++kw) {
                        const float weight_value = load_quantized_row_value_sycl<WeightType>(
                            weight, ic * kernel + kw, local_oc, output_channels_per_group);
                        #pragma unroll
                        for (int64_t t = 0; t < time_tile; ++t) {
                            const int64_t ow = output_start + t;
                            const int64_t numerator = ow + padding - kw * dilation;
                            if (ow >= output_length || numerator < 0 || numerator % stride != 0) continue;
                            const int64_t iw = numerator / stride;
                            if (iw >= input_length) continue;
                            const T value = *reinterpret_cast<const T*>(reinterpret_cast<const char*>(input) +
                                n * input_nb2 + ic * input_nb1 + iw * input_nb0);
                            sums[t] += static_cast<float>(value) * weight_value;
                        }
                    }
                }
                #pragma unroll
                for (int64_t t = 0; t < time_tile; ++t) {
                    sums[t] = ::sycl::reduce_over_group(subgroup, sums[t], ::sycl::plus<float>());
                }
                if (lane == 0) {
                    const float bias_value = bias
                        ? (bias_type == 0 ? static_cast<const float*>(bias)[oc]
                                          : static_cast<float>(static_cast<const ::sycl::half*>(bias)[oc]))
                        : 0.0f;
                    #pragma unroll
                    for (int64_t t = 0; t < time_tile; ++t) {
                        const int64_t ow = output_start + t;
                        if (ow >= output_length) continue;
                        *reinterpret_cast<T*>(reinterpret_cast<char*>(output) + n * output_nb2 +
                            oc * output_nb1 + ow * output_nb0) = static_cast<T>(sums[t] + bias_value);
                    }
                }
            });
    });
}

} // namespace sycl
} // namespace ggml_ops_ext
