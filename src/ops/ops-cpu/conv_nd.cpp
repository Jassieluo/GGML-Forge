#include "ops/ops.h"
#include "ops/cpu.h"
#include "matmul_f32.h"

#include <algorithm>
#include <atomic>
#include <vector>

#if defined(_MSC_VER)
#define FORGE_ALWAYS_INLINE
#else
#define FORGE_ALWAYS_INLINE __attribute__((always_inline))
#endif

namespace ggml_ops_ext::cpu {

namespace {

float load_float(const ggml_tensor* tensor, size_t offset) {
    const char* data = static_cast<const char*>(tensor->data) + offset;
    switch (tensor->type) {
        case GGML_TYPE_F32: return *reinterpret_cast<const float*>(data);
        case GGML_TYPE_F16: return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(data));
        default: return 0.0f;
    }
}

template <ggml_type Type>
inline float load_activation(const ggml_tensor* tensor, size_t offset) {
    const char* data = static_cast<const char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) {
        return *reinterpret_cast<const float*>(data);
    } else if constexpr (Type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(data));
    } else {
        return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(data));
    }
}

template <ggml_type Type>
inline void store_activation(ggml_tensor* tensor, size_t offset, float value) {
    char* data = static_cast<char*>(tensor->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) {
        *reinterpret_cast<float*>(data) = value;
    } else if constexpr (Type == GGML_TYPE_F16) {
        *reinterpret_cast<ggml_fp16_t*>(data) = ggml_fp32_to_fp16(value);
    } else {
        *reinterpret_cast<ggml_bf16_t*>(data) = ggml_fp32_to_bf16(value);
    }
}

inline size_t activation_offset(
    const ggml_tensor* tensor, int dims, int64_t x, int64_t y, int64_t z,
    int64_t channel, int64_t batch, const ops_conv_nd_desc& desc
) {
    if (dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    const int64_t flat = x + desc.input_size[0] * (y + desc.input_size[1] * z);
    return batch * tensor->nb[2] + channel * tensor->nb[1] + flat * tensor->nb[0];
}

inline size_t output_offset(
    const ggml_tensor* tensor, int dims, int64_t x, int64_t y, int64_t z,
    int64_t channel, int64_t batch, const ops_conv_nd_desc& desc
) {
    if (dims == 2) {
        return batch * tensor->nb[3] + channel * tensor->nb[2] + y * tensor->nb[1] + x * tensor->nb[0];
    }
    const int64_t flat = x + desc.output_size[0] * (y + desc.output_size[1] * z);
    return batch * tensor->nb[2] + channel * tensor->nb[1] + flat * tensor->nb[0];
}

bool decode_weight_row(
    const ggml_tensor* weight, int64_t kernel_index, int64_t outer,
    int64_t row_elements, float* output
) {
    const char* row = static_cast<const char*>(weight->data) +
        kernel_index * weight->nb[1] + outer * weight->nb[2];
    if (ggml_is_quantized(weight->type)) {
        const ggml_type_traits* traits = ggml_get_type_traits(weight->type);
        if (!traits || !traits->to_float || row_elements % traits->blck_size != 0) return false;
        traits->to_float(row, output, row_elements);
        return true;
    }
    for (int64_t i = 0; i < row_elements; ++i) {
        output[i] = load_float(weight, static_cast<size_t>(row - static_cast<const char*>(weight->data)) + i * weight->nb[0]);
    }
    return true;
}

bool decode_flattened_row(
    const ggml_tensor* weight, int64_t row_index, int64_t row_elements, float* output
) {
    const char* row = static_cast<const char*>(weight->data) + row_index * weight->nb[1];
    if (ggml_is_quantized(weight->type)) {
        const ggml_type_traits* traits = ggml_get_type_traits(weight->type);
        if (!traits || !traits->to_float || row_elements % traits->blck_size != 0) return false;
        traits->to_float(row, output, row_elements);
        return true;
    }
    for (int64_t i = 0; i < row_elements; ++i) {
        output[i] = load_float(weight, row_index * weight->nb[1] + i * weight->nb[0]);
    }
    return true;
}

bool execute_pointwise_f32(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume != 1 || params.weight->type != GGML_TYPE_F32 ||
        params.input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.encoded.stride[axis] != 1 || params.encoded.dilation[axis] != 1 ||
            params.encoded.padding_before[axis] != 0 || params.encoded.padding_after[axis] != 0) return false;
    }
    const int dims = desc.spatial_dims;
    const int64_t spatial = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    if (spatial < 16 || desc.input_channels_per_group < 16 || desc.output_channels_per_group < 16) return false;
    const int64_t input_channels = desc.input_channels_per_group * desc.groups;
    const int64_t output_channels = desc.output_channels_per_group * desc.groups;
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t weight_row_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? params.weight->nb[1] : params.weight->nb[2];
    if (params.input->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
        input_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        output_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        input_batch_stride != static_cast<size_t>(spatial * input_channels) * sizeof(float) ||
        output_batch_stride != static_cast<size_t>(spatial * output_channels) * sizeof(float) ||
        weight_row_stride != static_cast<size_t>(desc.input_channels_per_group) * sizeof(float)) return false;

    const int threads = backend_thread_count(backend);
    for (int64_t batch = 0; batch < desc.batch; ++batch) {
        for (int64_t group = 0; group < desc.groups; ++group) {
            const float* weight = reinterpret_cast<const float*>(
                static_cast<const char*>(params.weight->data) +
                group * desc.output_channels_per_group * weight_row_stride);
            const float* input = reinterpret_cast<const float*>(
                static_cast<const char*>(params.input->data) + batch * input_batch_stride +
                group * desc.input_channels_per_group * input_channel_stride);
            float* output = reinterpret_cast<float*>(
                static_cast<char*>(node->data) + batch * output_batch_stride +
                group * desc.output_channels_per_group * output_channel_stride);
            // [Cout, Cin] x [Cin, spatial] -> [Cout, spatial]. This is true
            // pointwise GEMM: no im2col and no activation-sized reorder buffer.
            ops_matmul_f32_nn(
                desc.output_channels_per_group, spatial, desc.input_channels_per_group,
                weight, input, output, threads);
        }
    }
    if (params.bias) {
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t oc = 0; oc < output_channels; ++oc) {
                const float bias = load_float(params.bias, oc * params.bias->nb[0]);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride + oc * output_channel_stride);
                for (int64_t position = 0; position < spatial; ++position) output[position] += bias;
            }
        }
    }
    return true;
}

bool execute_pointwise_transposed_f32(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume != 1 || params.weight->type != GGML_TYPE_F32 ||
        params.input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.encoded.stride[axis] != 1 || params.encoded.dilation[axis] != 1 ||
            params.encoded.padding_before[axis] != 0 || params.encoded.padding_after[axis] != 0 ||
            params.encoded.output_padding[axis] != 0) return false;
    }
    const int dims = desc.spatial_dims;
    const int64_t spatial = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    if (spatial < 16 || desc.input_channels_per_group < 16 || desc.output_channels_per_group < 16) return false;
    const int64_t input_channels = desc.input_channels_per_group * desc.groups;
    const int64_t output_channels = desc.output_channels_per_group * desc.groups;
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t weight_row_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? params.weight->nb[1] : params.weight->nb[2];
    if (params.input->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
        input_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        output_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        input_batch_stride != static_cast<size_t>(spatial * input_channels) * sizeof(float) ||
        output_batch_stride != static_cast<size_t>(spatial * output_channels) * sizeof(float) ||
        weight_row_stride != static_cast<size_t>(desc.output_channels_per_group) * sizeof(float)) return false;

    const int threads = backend_thread_count(backend);
    for (int64_t batch = 0; batch < desc.batch; ++batch) {
        for (int64_t group = 0; group < desc.groups; ++group) {
            const float* weight = reinterpret_cast<const float*>(
                static_cast<const char*>(params.weight->data) +
                group * desc.input_channels_per_group * weight_row_stride);
            const float* input = reinterpret_cast<const float*>(
                static_cast<const char*>(params.input->data) + batch * input_batch_stride +
                group * desc.input_channels_per_group * input_channel_stride);
            float* output = reinterpret_cast<float*>(
                static_cast<char*>(node->data) + batch * output_batch_stride +
                group * desc.output_channels_per_group * output_channel_stride);
            // Stored ConvTranspose rows are [Cin,Cout]. Compute W^T x X
            // directly into [Cout,spatial], without col2im or atomics.
            ops_matmul_f32_tn(
                desc.output_channels_per_group, spatial, desc.input_channels_per_group,
                weight, input, output, threads);
        }
    }
    if (params.bias) {
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t oc = 0; oc < output_channels; ++oc) {
                const float bias = load_float(params.bias, oc * params.bias->nb[0]);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride + oc * output_channel_stride);
                for (int64_t position = 0; position < spatial; ++position) output[position] += bias;
            }
        }
    }
    return true;
}

bool execute_pointwise_quantized_f32(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume != 1 || !ggml_is_quantized(params.weight->type) ||
        params.input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.encoded.stride[axis] != 1 || params.encoded.dilation[axis] != 1 ||
            params.encoded.padding_before[axis] != 0 || params.encoded.padding_after[axis] != 0) return false;
    }
    const int dims = desc.spatial_dims;
    const int64_t spatial = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    if (spatial < 16 || desc.input_channels_per_group < 16 || desc.output_channels_per_group < 16) return false;
    const int64_t input_channels = desc.input_channels_per_group * desc.groups;
    const int64_t output_channels = desc.output_channels_per_group * desc.groups;
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t weight_row_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? params.weight->nb[1] : params.weight->nb[2];
    const size_t expected_weight_row = ggml_row_size(params.weight->type, desc.input_channels_per_group);
    const ggml_type_traits* traits = ggml_get_type_traits(params.weight->type);
    if (!traits || !traits->to_float || expected_weight_row == 0 ||
        params.input->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
        input_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        output_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        input_batch_stride != static_cast<size_t>(spatial * input_channels) * sizeof(float) ||
        output_batch_stride != static_cast<size_t>(spatial * output_channels) * sizeof(float) ||
        weight_row_stride != expected_weight_row) return false;

    constexpr int64_t output_tile = 32;
    const int threads = backend_thread_count(backend);
    for (int64_t group = 0; group < desc.groups; ++group) {
        for (int64_t output_base = 0; output_base < desc.output_channels_per_group; output_base += output_tile) {
            const int64_t count = std::min<int64_t>(output_tile, desc.output_channels_per_group - output_base);
            std::vector<float> decoded(static_cast<size_t>(count * desc.input_channels_per_group));
            for (int64_t lane = 0; lane < count; ++lane) {
                const int64_t oc = group * desc.output_channels_per_group + output_base + lane;
                const char* row = static_cast<const char*>(params.weight->data) + oc * weight_row_stride;
                traits->to_float(row, decoded.data() + lane * desc.input_channels_per_group,
                                 desc.input_channels_per_group);
            }
            for (int64_t batch = 0; batch < desc.batch; ++batch) {
                const float* input = reinterpret_cast<const float*>(
                    static_cast<const char*>(params.input->data) + batch * input_batch_stride +
                    group * desc.input_channels_per_group * input_channel_stride);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride +
                    (group * desc.output_channels_per_group + output_base) * output_channel_stride);
                ops_matmul_f32_nn(count, spatial, desc.input_channels_per_group,
                                  decoded.data(), input, output, threads);
            }
        }
    }
    if (params.bias) {
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t oc = 0; oc < output_channels; ++oc) {
                const float bias = load_float(params.bias, oc * params.bias->nb[0]);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride + oc * output_channel_stride);
                for (int64_t position = 0; position < spatial; ++position) output[position] += bias;
            }
        }
    }
    return true;
}

bool execute_pointwise_transposed_quantized_f32(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume != 1 || !ggml_is_quantized(params.weight->type) ||
        params.input->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    for (int axis = 0; axis < 3; ++axis) {
        if (params.encoded.stride[axis] != 1 || params.encoded.dilation[axis] != 1 ||
            params.encoded.padding_before[axis] != 0 || params.encoded.padding_after[axis] != 0 ||
            params.encoded.output_padding[axis] != 0) return false;
    }
    const int dims = desc.spatial_dims;
    const int64_t spatial = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    if (spatial < 16 || desc.input_channels_per_group < 16 || desc.output_channels_per_group < 16) return false;
    const int64_t input_channels = desc.input_channels_per_group * desc.groups;
    const int64_t output_channels = desc.output_channels_per_group * desc.groups;
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t weight_row_stride = desc.weight_layout == ops_weight_layout::flattened_rows
        ? params.weight->nb[1] : params.weight->nb[2];
    const size_t expected_weight_row = ggml_row_size(params.weight->type, desc.output_channels_per_group);
    const ggml_type_traits* traits = ggml_get_type_traits(params.weight->type);
    if (!traits || !traits->to_float || expected_weight_row == 0 ||
        params.input->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) ||
        input_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        output_channel_stride != static_cast<size_t>(spatial) * sizeof(float) ||
        input_batch_stride != static_cast<size_t>(spatial * input_channels) * sizeof(float) ||
        output_batch_stride != static_cast<size_t>(spatial * output_channels) * sizeof(float) ||
        weight_row_stride != expected_weight_row) return false;

    constexpr int64_t input_tile = 32;
    const int threads = backend_thread_count(backend);
    for (int64_t group = 0; group < desc.groups; ++group) {
        for (int64_t input_base = 0; input_base < desc.input_channels_per_group; input_base += input_tile) {
            const int64_t count = std::min<int64_t>(input_tile, desc.input_channels_per_group - input_base);
            std::vector<float> decoded(static_cast<size_t>(count * desc.output_channels_per_group));
            for (int64_t lane = 0; lane < count; ++lane) {
                const int64_t ic = group * desc.input_channels_per_group + input_base + lane;
                const char* row = static_cast<const char*>(params.weight->data) + ic * weight_row_stride;
                traits->to_float(row, decoded.data() + lane * desc.output_channels_per_group,
                                 desc.output_channels_per_group);
            }
            for (int64_t batch = 0; batch < desc.batch; ++batch) {
                const float* input = reinterpret_cast<const float*>(
                    static_cast<const char*>(params.input->data) + batch * input_batch_stride +
                    (group * desc.input_channels_per_group + input_base) * input_channel_stride);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride +
                    group * desc.output_channels_per_group * output_channel_stride);
                ops_matmul_f32_tn_accumulate(
                    desc.output_channels_per_group, spatial, count,
                    decoded.data(), input, output, input_base == 0 ? 0.0f : 1.0f, threads);
            }
        }
    }
    if (params.bias) {
        #pragma omp parallel for collapse(2) num_threads(threads) schedule(static)
        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t oc = 0; oc < output_channels; ++oc) {
                const float bias = load_float(params.bias, oc * params.bias->nb[0]);
                float* output = reinterpret_cast<float*>(
                    static_cast<char*>(node->data) + batch * output_batch_stride + oc * output_channel_stride);
                for (int64_t position = 0; position < spatial; ++position) output[position] += bias;
            }
        }
    }
    return true;
}

template <ggml_type InputType>
bool execute_forward_tiled_gemm(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume <= 1 || params.input->type != InputType || node->type != InputType) return false;

    const int dims = desc.spatial_dims;
    const int64_t input_volume = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t element_size = ggml_type_size(InputType);
    if (output_volume < 64 || desc.input_channels_per_group < 16 ||
        desc.output_channels_per_group < 16 || params.input->nb[0] != element_size ||
        node->nb[0] != element_size ||
        input_channel_stride != static_cast<size_t>(input_volume) * element_size ||
        output_channel_stride != static_cast<size_t>(output_volume) * element_size) return false;

    constexpr int64_t position_tile = 128;
    const int64_t reduction = desc.kernel_volume * desc.input_channels_per_group;
    const int threads = backend_thread_count(backend);
    std::vector<float> decoded(static_cast<size_t>(desc.output_channels_per_group * reduction));
    std::vector<float> input_tile(static_cast<size_t>(reduction * position_tile));
    std::vector<float> output_tile;
    if constexpr (InputType != GGML_TYPE_F32) {
        output_tile.resize(static_cast<size_t>(desc.output_channels_per_group * position_tile));
    }

    std::vector<int64_t> input_positions(
        static_cast<size_t>(desc.kernel_volume * output_volume), -1);
    for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
        const int64_t kx = kernel % desc.kernel_size[0];
        const int64_t kernel_rem = kernel / desc.kernel_size[0];
        const int64_t ky = kernel_rem % desc.kernel_size[1];
        const int64_t kz = kernel_rem / desc.kernel_size[1];
        int64_t* mapped = input_positions.data() + kernel * output_volume;
        for (int64_t output_flat = 0; output_flat < output_volume; ++output_flat) {
            const int64_t ox = output_flat % desc.output_size[0];
            const int64_t output_rem = output_flat / desc.output_size[0];
            const int64_t oy = output_rem % desc.output_size[1];
            const int64_t oz = output_rem / desc.output_size[1];
            const int64_t ix = ox * params.encoded.stride[0] -
                params.encoded.padding_before[0] + kx * params.encoded.dilation[0];
            const int64_t iy = oy * params.encoded.stride[1] -
                params.encoded.padding_before[1] + ky * params.encoded.dilation[1];
            const int64_t iz = oz * params.encoded.stride[2] -
                params.encoded.padding_before[2] + kz * params.encoded.dilation[2];
            if (ix >= 0 && ix < desc.input_size[0] && iy >= 0 && iy < desc.input_size[1] &&
                iz >= 0 && iz < desc.input_size[2]) {
                mapped[output_flat] = ix + desc.input_size[0] *
                    (iy + desc.input_size[1] * iz);
            }
        }
    }

    for (int64_t group = 0; group < desc.groups; ++group) {
        for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
            const int64_t oc = group * desc.output_channels_per_group + local_oc;
            float* row = decoded.data() + local_oc * reduction;
            if (desc.weight_layout == ops_weight_layout::flattened_rows) {
                if (!decode_flattened_row(params.weight, oc, reduction, row)) return false;
            } else {
                for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                    if (!decode_weight_row(params.weight, kernel, oc,
                                           desc.input_channels_per_group,
                                           row + kernel * desc.input_channels_per_group)) return false;
                }
            }
        }

        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t output_base = 0; output_base < output_volume; output_base += position_tile) {
                const int64_t count = std::min<int64_t>(position_tile, output_volume - output_base);
                const auto fill_input_row = [&](int64_t r) FORGE_ALWAYS_INLINE {
                    const int64_t kernel = desc.weight_layout == ops_weight_layout::flattened_rows
                        ? r % desc.kernel_volume : r / desc.input_channels_per_group;
                    const int64_t local_ic = desc.weight_layout == ops_weight_layout::flattened_rows
                        ? r / desc.kernel_volume : r % desc.input_channels_per_group;
                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
                    const int64_t* mapped = input_positions.data() + kernel * output_volume + output_base;
                    float* tile_row = input_tile.data() + r * count;
                    for (int64_t position = 0; position < count; ++position) {
                        tile_row[position] = mapped[position] >= 0
                            ? load_activation<InputType>(
                                  params.input, batch * input_batch_stride +
                                      ic * input_channel_stride + mapped[position] * params.input->nb[0])
                            : 0.0f;
                    }
                };
                if constexpr (InputType == GGML_TYPE_F32) {
                    for (int64_t r = 0; r < reduction; ++r) fill_input_row(r);
                } else {
                    #pragma omp parallel for num_threads(threads) schedule(static)
                    for (int64_t r = 0; r < reduction; ++r) fill_input_row(r);
                }

                float* output = nullptr;
                if constexpr (InputType == GGML_TYPE_F32) {
                    output = reinterpret_cast<float*>(
                        static_cast<char*>(node->data) + batch * output_batch_stride +
                        group * desc.output_channels_per_group * output_channel_stride) + output_base;
                    ops_matmul_f32_nn_strided(
                        desc.output_channels_per_group, count, reduction,
                        decoded.data(), input_tile.data(), output,
                        output_volume, threads);
                } else {
                    output = output_tile.data();
                    ops_matmul_f32_nn(
                        desc.output_channels_per_group, count, reduction,
                        decoded.data(), input_tile.data(), output, threads);
                }

                if (params.bias) {
                    for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
                        const int64_t oc = group * desc.output_channels_per_group + local_oc;
                        const float bias = load_float(params.bias, oc * params.bias->nb[0]);
                        float* row = output + local_oc *
                            (InputType == GGML_TYPE_F32 ? output_volume : count);
                        #pragma omp simd
                        for (int64_t position = 0; position < count; ++position) row[position] += bias;
                    }
                }
                if constexpr (InputType != GGML_TYPE_F32) {
                    #pragma omp parallel for num_threads(threads) schedule(static)
                    for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
                        const int64_t oc = group * desc.output_channels_per_group + local_oc;
                        const float* row = output + local_oc * count;
                        for (int64_t position = 0; position < count; ++position) {
                            const int64_t output_flat = output_base + position;
                            store_activation<InputType>(
                                node, batch * output_batch_stride + oc * output_channel_stride +
                                    output_flat * node->nb[0],
                                row[position]);
                        }
                    }
                }
            }
        }
    }
    return true;
}

template <ggml_type InputType>
bool execute_transposed_tiled_gemm(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    if (desc.kernel_volume <= 1 || params.input->type != InputType || node->type != InputType) return false;

    const int dims = desc.spatial_dims;
    const int64_t input_volume = desc.input_size[0] * desc.input_size[1] * desc.input_size[2];
    const int64_t output_volume = desc.output_size[0] * desc.output_size[1] * desc.output_size[2];
    const size_t input_channel_stride = dims == 2 ? params.input->nb[2] : params.input->nb[1];
    const size_t input_batch_stride = dims == 2 ? params.input->nb[3] : params.input->nb[2];
    const size_t output_channel_stride = dims == 2 ? node->nb[2] : node->nb[1];
    const size_t output_batch_stride = dims == 2 ? node->nb[3] : node->nb[2];
    const size_t element_size = ggml_type_size(InputType);
    if (output_volume < 64 || desc.input_channels_per_group < 16 ||
        desc.output_channels_per_group < 16 || params.input->nb[0] != element_size ||
        node->nb[0] != element_size ||
        input_channel_stride != static_cast<size_t>(input_volume) * element_size ||
        output_channel_stride != static_cast<size_t>(output_volume) * element_size) return false;

    constexpr int64_t position_tile = 256;
    const int64_t reduction = desc.kernel_volume * desc.input_channels_per_group;
    const int threads = backend_thread_count(backend);
    std::vector<float> decoded(static_cast<size_t>(desc.output_channels_per_group * reduction));
    std::vector<float> input_tile(static_cast<size_t>(reduction * position_tile));
    std::vector<float> weight_row;
    std::vector<float> output_tile;
    if constexpr (InputType != GGML_TYPE_F32) {
        output_tile.resize(static_cast<size_t>(desc.output_channels_per_group * position_tile));
    }

    for (int64_t group = 0; group < desc.groups; ++group) {
        std::fill(decoded.begin(), decoded.end(), 0.0f);
        if (desc.weight_layout == ops_weight_layout::flattened_rows) {
            weight_row.resize(static_cast<size_t>(desc.kernel_volume * desc.output_channels_per_group));
            for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                if (!decode_flattened_row(params.weight, ic,
                                          desc.kernel_volume * desc.output_channels_per_group,
                                          weight_row.data())) return false;
                for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
                    float* output_row = decoded.data() + local_oc * reduction;
                    for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                        output_row[kernel * desc.input_channels_per_group + local_ic] =
                            weight_row[local_oc * desc.kernel_volume + kernel];
                    }
                }
            }
        } else {
            weight_row.resize(static_cast<size_t>(desc.output_channels_per_group));
            for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                    if (!decode_weight_row(params.weight, kernel, ic,
                                           desc.output_channels_per_group,
                                           weight_row.data())) return false;
                    for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
                        decoded[local_oc * reduction + kernel * desc.input_channels_per_group + local_ic] =
                            weight_row[local_oc];
                    }
                }
            }
        }

        const bool phase_path = params.encoded.dilation[0] == 1 &&
            params.encoded.dilation[1] == 1 && params.encoded.dilation[2] == 1 &&
            (params.encoded.stride[0] > 1 || params.encoded.stride[1] > 1 ||
             params.encoded.stride[2] > 1);
        if (phase_path) {
            std::vector<int64_t> phase_kernels;
            std::vector<int64_t> phase_positions;
            std::vector<int64_t> phase_input_positions;
            std::vector<float> phase_weights;
            std::vector<float> phase_output(
                static_cast<size_t>(desc.output_channels_per_group * position_tile));
            for (int64_t phase_z = 0; phase_z < params.encoded.stride[2]; ++phase_z) {
                for (int64_t phase_y = 0; phase_y < params.encoded.stride[1]; ++phase_y) {
                    for (int64_t phase_x = 0; phase_x < params.encoded.stride[0]; ++phase_x) {
                        phase_kernels.clear();
                        for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                            const int64_t kx = kernel % desc.kernel_size[0];
                            const int64_t kernel_rem = kernel / desc.kernel_size[0];
                            const int64_t ky = kernel_rem % desc.kernel_size[1];
                            const int64_t kz = kernel_rem / desc.kernel_size[1];
                            if ((phase_x + params.encoded.padding_before[0] - kx) % params.encoded.stride[0] == 0 &&
                                (phase_y + params.encoded.padding_before[1] - ky) % params.encoded.stride[1] == 0 &&
                                (phase_z + params.encoded.padding_before[2] - kz) % params.encoded.stride[2] == 0) {
                                phase_kernels.push_back(kernel);
                            }
                        }
                        if (phase_kernels.empty()) continue;

                        phase_positions.clear();
                        for (int64_t oz = phase_z; oz < desc.output_size[2]; oz += params.encoded.stride[2]) {
                            for (int64_t oy = phase_y; oy < desc.output_size[1]; oy += params.encoded.stride[1]) {
                                for (int64_t ox = phase_x; ox < desc.output_size[0]; ox += params.encoded.stride[0]) {
                                    phase_positions.push_back(
                                        ox + desc.output_size[0] * (oy + desc.output_size[1] * oz));
                                }
                            }
                        }

                        phase_input_positions.assign(
                            phase_kernels.size() * phase_positions.size(), -1);
                        for (size_t phase_kernel = 0; phase_kernel < phase_kernels.size(); ++phase_kernel) {
                            const int64_t kernel = phase_kernels[phase_kernel];
                            const int64_t kx = kernel % desc.kernel_size[0];
                            const int64_t kernel_rem = kernel / desc.kernel_size[0];
                            const int64_t ky = kernel_rem % desc.kernel_size[1];
                            const int64_t kz = kernel_rem / desc.kernel_size[1];
                            int64_t* mapped = phase_input_positions.data() +
                                phase_kernel * phase_positions.size();
                            for (size_t position = 0; position < phase_positions.size(); ++position) {
                                const int64_t output_flat = phase_positions[position];
                                const int64_t ox = output_flat % desc.output_size[0];
                                const int64_t output_rem = output_flat / desc.output_size[0];
                                const int64_t oy = output_rem % desc.output_size[1];
                                const int64_t oz = output_rem / desc.output_size[1];
                                const int64_t ix = (ox + params.encoded.padding_before[0] - kx) /
                                    params.encoded.stride[0];
                                const int64_t iy = (oy + params.encoded.padding_before[1] - ky) /
                                    params.encoded.stride[1];
                                const int64_t iz = (oz + params.encoded.padding_before[2] - kz) /
                                    params.encoded.stride[2];
                                if (ix >= 0 && ix < desc.input_size[0] &&
                                    iy >= 0 && iy < desc.input_size[1] &&
                                    iz >= 0 && iz < desc.input_size[2]) {
                                    mapped[position] = ix + desc.input_size[0] *
                                        (iy + desc.input_size[1] * iz);
                                }
                            }
                        }

                        const int64_t phase_reduction =
                            static_cast<int64_t>(phase_kernels.size()) * desc.input_channels_per_group;
                        phase_weights.resize(
                            static_cast<size_t>(desc.output_channels_per_group * phase_reduction));
                        for (int64_t local_oc = 0; local_oc < desc.output_channels_per_group; ++local_oc) {
                            float* phase_row = phase_weights.data() + local_oc * phase_reduction;
                            const float* full_row = decoded.data() + local_oc * reduction;
                            for (size_t phase_kernel = 0; phase_kernel < phase_kernels.size(); ++phase_kernel) {
                                const float* source = full_row +
                                    phase_kernels[phase_kernel] * desc.input_channels_per_group;
                                std::copy_n(source, desc.input_channels_per_group,
                                            phase_row + phase_kernel * desc.input_channels_per_group);
                            }
                        }

                        for (int64_t batch = 0; batch < desc.batch; ++batch) {
                            for (size_t position_base = 0; position_base < phase_positions.size();
                                 position_base += position_tile) {
                                const int64_t count = std::min<int64_t>(
                                    position_tile, phase_positions.size() - position_base);
                                const auto fill_input_row = [&](int64_t r) FORGE_ALWAYS_INLINE {
                                    const int64_t phase_kernel = r / desc.input_channels_per_group;
                                    const int64_t local_ic = r % desc.input_channels_per_group;
                                    const int64_t ic = group * desc.input_channels_per_group + local_ic;
                                    const int64_t* mapped = phase_input_positions.data() +
                                        phase_kernel * phase_positions.size() + position_base;
                                    float* tile_row = input_tile.data() + r * count;
                                    for (int64_t position = 0; position < count; ++position) {
                                        tile_row[position] = mapped[position] >= 0
                                            ? load_activation<InputType>(
                                                  params.input, batch * input_batch_stride +
                                                      ic * input_channel_stride +
                                                      mapped[position] * params.input->nb[0])
                                            : 0.0f;
                                    }
                                };
                                if constexpr (InputType == GGML_TYPE_F32) {
                                    for (int64_t r = 0; r < phase_reduction; ++r) fill_input_row(r);
                                } else {
                                    #pragma omp parallel for num_threads(threads) schedule(static)
                                    for (int64_t r = 0; r < phase_reduction; ++r) fill_input_row(r);
                                }

                                ops_matmul_f32_nn(
                                    desc.output_channels_per_group, count, phase_reduction,
                                    phase_weights.data(), input_tile.data(), phase_output.data(), threads);
                                const auto store_output_row = [&](int64_t local_oc) FORGE_ALWAYS_INLINE {
                                    const int64_t oc = group * desc.output_channels_per_group + local_oc;
                                    const float bias = params.bias
                                        ? load_float(params.bias, oc * params.bias->nb[0]) : 0.0f;
                                    const float* row = phase_output.data() + local_oc * count;
                                    for (int64_t position = 0; position < count; ++position) {
                                        const int64_t output_flat = phase_positions[position_base + position];
                                        store_activation<InputType>(
                                            node, batch * output_batch_stride + oc * output_channel_stride +
                                                output_flat * node->nb[0],
                                            row[position] + bias);
                                    }
                                };
                                if constexpr (InputType == GGML_TYPE_F32) {
                                    for (int64_t local_oc = 0;
                                         local_oc < desc.output_channels_per_group; ++local_oc) {
                                        store_output_row(local_oc);
                                    }
                                } else {
                                    #pragma omp parallel for num_threads(threads) schedule(static)
                                    for (int64_t local_oc = 0;
                                         local_oc < desc.output_channels_per_group; ++local_oc) {
                                        store_output_row(local_oc);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            continue;
        }

        for (int64_t batch = 0; batch < desc.batch; ++batch) {
            for (int64_t output_base = 0; output_base < output_volume; output_base += position_tile) {
                const int64_t count = std::min<int64_t>(position_tile, output_volume - output_base);
                const auto fill_input_row = [&](int64_t r) FORGE_ALWAYS_INLINE {
                    const int64_t kernel = r / desc.input_channels_per_group;
                    const int64_t local_ic = r % desc.input_channels_per_group;
                    const int64_t kx = kernel % desc.kernel_size[0];
                    const int64_t kernel_rem = kernel / desc.kernel_size[0];
                    const int64_t ky = kernel_rem % desc.kernel_size[1];
                    const int64_t kz = kernel_rem / desc.kernel_size[1];
                    float* tile_row = input_tile.data() + r * count;
                    for (int64_t position = 0; position < count; ++position) {
                        const int64_t output_flat = output_base + position;
                        const int64_t ox = output_flat % desc.output_size[0];
                        const int64_t output_rem = output_flat / desc.output_size[0];
                        const int64_t oy = output_rem % desc.output_size[1];
                        const int64_t oz = output_rem / desc.output_size[1];
                        const int64_t sx = ox + params.encoded.padding_before[0] -
                            kx * params.encoded.dilation[0];
                        const int64_t sy = oy + params.encoded.padding_before[1] -
                            ky * params.encoded.dilation[1];
                        const int64_t sz = oz + params.encoded.padding_before[2] -
                            kz * params.encoded.dilation[2];
                        float value = 0.0f;
                        if (sx >= 0 && sy >= 0 && sz >= 0 &&
                            sx % params.encoded.stride[0] == 0 &&
                            sy % params.encoded.stride[1] == 0 &&
                            sz % params.encoded.stride[2] == 0) {
                            const int64_t ix = sx / params.encoded.stride[0];
                            const int64_t iy = sy / params.encoded.stride[1];
                            const int64_t iz = sz / params.encoded.stride[2];
                            if (ix < desc.input_size[0] && iy < desc.input_size[1] && iz < desc.input_size[2]) {
                                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                                value = load_activation<InputType>(
                                    params.input,
                                    activation_offset(params.input, dims, ix, iy, iz, ic, batch, desc));
                            }
                        }
                        tile_row[position] = value;
                    }
                };
                if constexpr (InputType == GGML_TYPE_F32) {
                    for (int64_t r = 0; r < reduction; ++r) fill_input_row(r);
                } else {
                    #pragma omp parallel for num_threads(threads) schedule(static)
                    for (int64_t r = 0; r < reduction; ++r) fill_input_row(r);
                }

                float* output = nullptr;
                if constexpr (InputType == GGML_TYPE_F32) {
                    output = reinterpret_cast<float*>(
                        static_cast<char*>(node->data) + batch * output_batch_stride +
                        group * desc.output_channels_per_group * output_channel_stride) + output_base;
                    ops_matmul_f32_nn_strided(
                        desc.output_channels_per_group, count, reduction,
                        decoded.data(), input_tile.data(), output,
                        output_volume, threads);
                } else {
                    output = output_tile.data();
                    ops_matmul_f32_nn(
                        desc.output_channels_per_group, count, reduction,
                        decoded.data(), input_tile.data(), output, threads);
                }

                const auto store_output_row = [&](int64_t local_oc) FORGE_ALWAYS_INLINE {
                    const int64_t oc = group * desc.output_channels_per_group + local_oc;
                    const float bias = params.bias ? load_float(params.bias, oc * params.bias->nb[0]) : 0.0f;
                    float* row = output + local_oc *
                        (InputType == GGML_TYPE_F32 ? output_volume : count);
                    if (bias != 0.0f) {
                        #pragma omp simd
                        for (int64_t position = 0; position < count; ++position) row[position] += bias;
                    }
                    if constexpr (InputType != GGML_TYPE_F32) {
                        for (int64_t position = 0; position < count; ++position) {
                            const int64_t output_flat = output_base + position;
                            const int64_t ox = output_flat % desc.output_size[0];
                            const int64_t output_rem = output_flat / desc.output_size[0];
                            const int64_t oy = output_rem % desc.output_size[1];
                            const int64_t oz = output_rem / desc.output_size[1];
                            store_activation<InputType>(
                                node, output_offset(node, dims, ox, oy, oz, oc, batch, desc), row[position]);
                        }
                    }
                };
                if constexpr (InputType == GGML_TYPE_F32) {
                    for (int64_t local_oc = 0;
                         local_oc < desc.output_channels_per_group; ++local_oc) {
                        store_output_row(local_oc);
                    }
                } else {
                    #pragma omp parallel for num_threads(threads) schedule(static)
                    for (int64_t local_oc = 0;
                         local_oc < desc.output_channels_per_group; ++local_oc) {
                        store_output_row(local_oc);
                    }
                }
            }
        }
    }
    return true;
}

template <ggml_type InputType>
bool execute_forward(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    const int threads = backend_thread_count(backend);
    const ggml_tensor* weight = params.weight;
    const ggml_tensor* input = params.input;
    const int dims = desc.spatial_dims;
    constexpr int64_t spatial_tile = 8;
    std::atomic<bool> decode_failed{false};

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t oc = 0; oc < desc.output_channels; ++oc) {
            std::vector<float> decoded(static_cast<size_t>(desc.kernel_volume * desc.input_channels_per_group));
            bool decoded_ok = true;
            if (desc.weight_layout == ops_weight_layout::flattened_rows) {
                decoded_ok = decode_flattened_row(weight, oc, desc.kernel_volume * desc.input_channels_per_group, decoded.data());
            } else {
                for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                    decoded_ok = decoded_ok && decode_weight_row(
                        weight, kernel, oc, desc.input_channels_per_group,
                        decoded.data() + kernel * desc.input_channels_per_group);
                }
            }
            if (!decoded_ok) {
                // Zero the channel so downstream never reads uninitialized memory,
                // and fail the whole op instead of silently skipping the channel.
                decode_failed.store(true, std::memory_order_relaxed);
                for (int64_t batch = 0; batch < desc.batch; ++batch) {
                    for (int64_t oz = 0; oz < desc.output_size[2]; ++oz) {
                        for (int64_t oy = 0; oy < desc.output_size[1]; ++oy) {
                            for (int64_t ox = 0; ox < desc.output_size[0]; ++ox) {
                                store_activation<InputType>(
                                    node, output_offset(node, dims, ox, oy, oz, oc, batch, desc), 0.0f);
                            }
                        }
                    }
                }
                continue;
            }
            const int64_t group = oc / desc.output_channels_per_group;
            const float bias = params.bias ? load_float(params.bias, oc * params.bias->nb[0]) : 0.0f;
            for (int64_t batch = 0; batch < desc.batch; ++batch) {
              for (int64_t oz = 0; oz < desc.output_size[2]; ++oz) {
               for (int64_t oy = 0; oy < desc.output_size[1]; ++oy) {
                for (int64_t ox_base = 0; ox_base < desc.output_size[0]; ox_base += spatial_tile) {
                 const int64_t tile_count = std::min<int64_t>(spatial_tile, desc.output_size[0] - ox_base);
                 float sums[spatial_tile];
                 std::fill_n(sums, tile_count, bias);
                for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                    const int64_t kx = kernel % desc.kernel_size[0];
                    const int64_t krem = kernel / desc.kernel_size[0];
                    const int64_t ky = krem % desc.kernel_size[1];
                    const int64_t kz = krem / desc.kernel_size[1];
                    const int64_t iy = oy * params.encoded.stride[1] - params.encoded.padding_before[1] + ky * params.encoded.dilation[1];
                    const int64_t iz = oz * params.encoded.stride[2] - params.encoded.padding_before[2] + kz * params.encoded.dilation[2];
                    if (iy < 0 || iy >= desc.input_size[1] || iz < 0 || iz >= desc.input_size[2]) continue;
                    for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                        const int64_t ic = group * desc.input_channels_per_group + local_ic;
                        const int64_t weight_index = desc.weight_layout == ops_weight_layout::flattened_rows
                            ? kernel + desc.kernel_volume * local_ic
                            : kernel * desc.input_channels_per_group + local_ic;
                        const float weight_value = decoded[weight_index];
                        #pragma omp simd
                        for (int64_t lane = 0; lane < tile_count; ++lane) {
                            const int64_t ox = ox_base + lane;
                            const int64_t ix = ox * params.encoded.stride[0] - params.encoded.padding_before[0] + kx * params.encoded.dilation[0];
                            if (ix >= 0 && ix < desc.input_size[0]) {
                                sums[lane] += load_activation<InputType>(input, activation_offset(input, dims, ix, iy, iz, ic, batch, desc)) * weight_value;
                            }
                        }
                    }
                }
                 for (int64_t lane = 0; lane < tile_count; ++lane) {
                    store_activation<InputType>(
                        node, output_offset(node, dims, ox_base + lane, oy, oz, oc, batch, desc), sums[lane]);
                 }
                }
               }
              }
            }
    }
    return !decode_failed.load(std::memory_order_relaxed);
}

template <ggml_type InputType>
bool execute_transposed(
    ggml_backend_t backend, ggml_tensor* node, const ops_conv_nd_node_params& params,
    const ops_conv_nd_desc& desc
) {
    const int threads = backend_thread_count(backend);
    const int dims = desc.spatial_dims;
    constexpr int64_t spatial_tile = 8;
    std::atomic<bool> decode_failed{false};

    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t oc = 0; oc < desc.output_channels; ++oc) {
            const int64_t group = oc / desc.output_channels_per_group;
            const int64_t local_oc = oc % desc.output_channels_per_group;
            std::vector<float> decoded(static_cast<size_t>(desc.kernel_volume * desc.input_channels_per_group));
            bool decoded_ok = true;
            for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                const int64_t ic = group * desc.input_channels_per_group + local_ic;
                if (desc.weight_layout == ops_weight_layout::flattened_rows) {
                    std::vector<float> row(static_cast<size_t>(desc.kernel_volume * desc.output_channels_per_group));
                    decoded_ok = decoded_ok && decode_flattened_row(
                        params.weight, ic, desc.kernel_volume * desc.output_channels_per_group, row.data());
                    for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                        decoded[kernel * desc.input_channels_per_group + local_ic] =
                            row[kernel + desc.kernel_volume * local_oc];
                    }
                } else {
                    std::vector<float> row(static_cast<size_t>(desc.output_channels_per_group));
                    for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                        decoded_ok = decoded_ok && decode_weight_row(
                            params.weight, kernel, ic, desc.output_channels_per_group, row.data());
                        decoded[kernel * desc.input_channels_per_group + local_ic] = row[local_oc];
                    }
                }
            }
            if (!decoded_ok) {
                decode_failed.store(true, std::memory_order_relaxed);
                for (int64_t batch = 0; batch < desc.batch; ++batch) {
                    for (int64_t oz = 0; oz < desc.output_size[2]; ++oz) {
                        for (int64_t oy = 0; oy < desc.output_size[1]; ++oy) {
                            for (int64_t ox = 0; ox < desc.output_size[0]; ++ox) {
                                store_activation<InputType>(
                                    node, output_offset(node, dims, ox, oy, oz, oc, batch, desc), 0.0f);
                            }
                        }
                    }
                }
                continue;
            }
            const float bias = params.bias ? load_float(params.bias, oc * params.bias->nb[0]) : 0.0f;
            const bool phase_path = params.encoded.dilation[0] == 1 &&
                params.encoded.dilation[1] == 1 && params.encoded.dilation[2] == 1 &&
                (params.encoded.stride[0] > 1 || params.encoded.stride[1] > 1 || params.encoded.stride[2] > 1);
            for (int64_t batch = 0; batch < desc.batch; ++batch) {
              for (int64_t oz = 0; oz < desc.output_size[2]; ++oz) {
               for (int64_t oy = 0; oy < desc.output_size[1]; ++oy) {
                for (int64_t ox_base = 0; ox_base < desc.output_size[0]; ox_base += spatial_tile) {
                 const int64_t tile_count = std::min<int64_t>(spatial_tile, desc.output_size[0] - ox_base);
                 float sums[spatial_tile];
                 std::fill_n(sums, tile_count, bias);
                 if (phase_path) {
                    for (int64_t lane = 0; lane < tile_count; ++lane) {
                        const int64_t ox = ox_base + lane;
                        const int64_t kx_start = (ox + params.encoded.padding_before[0]) % params.encoded.stride[0];
                        const int64_t ky_start = (oy + params.encoded.padding_before[1]) % params.encoded.stride[1];
                        const int64_t kz_start = (oz + params.encoded.padding_before[2]) % params.encoded.stride[2];
                        for (int64_t kz = kz_start; kz < desc.kernel_size[2]; kz += params.encoded.stride[2]) {
                            const int64_t sz = oz + params.encoded.padding_before[2] - kz;
                            if (sz < 0) continue;
                            const int64_t iz = sz / params.encoded.stride[2];
                            if (iz >= desc.input_size[2]) continue;
                            for (int64_t ky = ky_start; ky < desc.kernel_size[1]; ky += params.encoded.stride[1]) {
                                const int64_t sy = oy + params.encoded.padding_before[1] - ky;
                                if (sy < 0) continue;
                                const int64_t iy = sy / params.encoded.stride[1];
                                if (iy >= desc.input_size[1]) continue;
                                for (int64_t kx = kx_start; kx < desc.kernel_size[0]; kx += params.encoded.stride[0]) {
                                    const int64_t sx = ox + params.encoded.padding_before[0] - kx;
                                    if (sx < 0) continue;
                                    const int64_t ix = sx / params.encoded.stride[0];
                                    if (ix >= desc.input_size[0]) continue;
                                    const int64_t kernel = kx + desc.kernel_size[0] * (ky + desc.kernel_size[1] * kz);
                                    const float* row = decoded.data() + kernel * desc.input_channels_per_group;
                                    for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                                        const int64_t ic = group * desc.input_channels_per_group + local_ic;
                                        sums[lane] += load_activation<InputType>(params.input,
                                            activation_offset(params.input, dims, ix, iy, iz, ic, batch, desc)) * row[local_ic];
                                    }
                                }
                            }
                        }
                        store_activation<InputType>(
                            node, output_offset(node, dims, ox, oy, oz, oc, batch, desc), sums[lane]);
                    }
                    continue;
                 }
                for (int64_t kernel = 0; kernel < desc.kernel_volume; ++kernel) {
                    const int64_t kx = kernel % desc.kernel_size[0];
                    const int64_t krem = kernel / desc.kernel_size[0];
                    const int64_t ky = krem % desc.kernel_size[1];
                    const int64_t kz = krem / desc.kernel_size[1];
                    const int64_t sy = oy + params.encoded.padding_before[1] - ky * params.encoded.dilation[1];
                    const int64_t sz = oz + params.encoded.padding_before[2] - kz * params.encoded.dilation[2];
                    if (sy < 0 || sz < 0 || sy % params.encoded.stride[1] || sz % params.encoded.stride[2]) continue;
                    const int64_t iy = sy / params.encoded.stride[1];
                    const int64_t iz = sz / params.encoded.stride[2];
                    if (iy >= desc.input_size[1] || iz >= desc.input_size[2]) continue;
                    const float* row = decoded.data() + kernel * desc.input_channels_per_group;
                    for (int64_t local_ic = 0; local_ic < desc.input_channels_per_group; ++local_ic) {
                        const int64_t ic = group * desc.input_channels_per_group + local_ic;
                        const float weight_value = row[local_ic];
                        #pragma omp simd
                        for (int64_t lane = 0; lane < tile_count; ++lane) {
                            const int64_t sx = ox_base + lane + params.encoded.padding_before[0] - kx * params.encoded.dilation[0];
                            if (sx < 0 || sx % params.encoded.stride[0]) continue;
                            const int64_t ix = sx / params.encoded.stride[0];
                            if (ix < desc.input_size[0]) {
                                sums[lane] += load_activation<InputType>(params.input, activation_offset(params.input, dims, ix, iy, iz, ic, batch, desc)) * weight_value;
                            }
                        }
                    }
                }
                 for (int64_t lane = 0; lane < tile_count; ++lane) {
                    store_activation<InputType>(
                        node, output_offset(node, dims, ox_base + lane, oy, oz, oc, batch, desc), sums[lane]);
                 }
                }
               }
              }
            }
    }
    return !decode_failed.load(std::memory_order_relaxed);
}

} // namespace

bool ops_cpu_op_conv_nd(ggml_backend_t backend, ggml_tensor* node) {
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
    if (!transposed && execute_pointwise_f32(backend, node, params, desc)) return true;
    if (transposed && execute_pointwise_transposed_f32(backend, node, params, desc)) return true;
    if (!transposed && execute_pointwise_quantized_f32(backend, node, params, desc)) return true;
    if (transposed && execute_pointwise_transposed_quantized_f32(backend, node, params, desc)) return true;
    if (!transposed) {
        if (params.input->type == GGML_TYPE_F32 &&
            execute_forward_tiled_gemm<GGML_TYPE_F32>(backend, node, params, desc)) return true;
        if (params.input->type == GGML_TYPE_F16 &&
            execute_forward_tiled_gemm<GGML_TYPE_F16>(backend, node, params, desc)) return true;
        if (params.input->type == GGML_TYPE_BF16 &&
            execute_forward_tiled_gemm<GGML_TYPE_BF16>(backend, node, params, desc)) return true;
    }
    if (transposed) {
        if (params.input->type == GGML_TYPE_F32 &&
            execute_transposed_tiled_gemm<GGML_TYPE_F32>(backend, node, params, desc)) return true;
        if (params.input->type == GGML_TYPE_F16 &&
            execute_transposed_tiled_gemm<GGML_TYPE_F16>(backend, node, params, desc)) return true;
        if (params.input->type == GGML_TYPE_BF16 &&
            execute_transposed_tiled_gemm<GGML_TYPE_BF16>(backend, node, params, desc)) return true;
    }
    switch (params.input->type) {
        case GGML_TYPE_F32:
            return transposed ? execute_transposed<GGML_TYPE_F32>(backend, node, params, desc)
                              : execute_forward<GGML_TYPE_F32>(backend, node, params, desc);
        case GGML_TYPE_F16:
            return transposed ? execute_transposed<GGML_TYPE_F16>(backend, node, params, desc)
                              : execute_forward<GGML_TYPE_F16>(backend, node, params, desc);
        case GGML_TYPE_BF16:
            return transposed ? execute_transposed<GGML_TYPE_BF16>(backend, node, params, desc)
                              : execute_forward<GGML_TYPE_BF16>(backend, node, params, desc);
        default: return false;
    }
}

} // namespace ggml_ops_ext::cpu

#undef FORGE_ALWAYS_INLINE
