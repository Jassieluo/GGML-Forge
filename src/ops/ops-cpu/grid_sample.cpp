#include "ops/cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>

namespace ggml_ops_ext::cpu {
namespace {

template <ggml_type Type> float load(const ggml_tensor* t, size_t offset) {
    const char* p = static_cast<const char*>(t->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) return *reinterpret_cast<const float*>(p);
    if constexpr (Type == GGML_TYPE_F16) return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(p));
    return ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t*>(p));
}
template <ggml_type Type> void store(ggml_tensor* t, size_t offset, float value) {
    char* p = static_cast<char*>(t->data) + offset;
    if constexpr (Type == GGML_TYPE_F32) *reinterpret_cast<float*>(p) = value;
    else if constexpr (Type == GGML_TYPE_F16) *reinterpret_cast<ggml_fp16_t*>(p) = ggml_fp32_to_fp16(value);
    else *reinterpret_cast<ggml_bf16_t*>(p) = ggml_fp32_to_bf16(value);
}
float unnormalize(float value, int64_t size, bool align) {
    return align ? (value + 1.0f) * (size - 1) * 0.5f : ((value + 1.0f) * size - 1.0f) * 0.5f;
}
float reflect(float value, int64_t size, bool align) {
    const float low = align ? 0.0f : -0.5f, high = align ? float(size - 1) : float(size) - 0.5f;
    if (high <= low) return 0.0f;
    value = std::abs(value - low); const float span = high - low;
    const float extra = std::fmod(value, span); const int flips = int(std::floor(value / span));
    return (flips & 1 ? span - extra : extra) + low;
}
template <ggml_type Type>
float sample(const ggml_tensor* input, int64_t x, int64_t y, int64_t c, int64_t n,
             const ops_grid_sample_2d_desc& d) {
    if (d.padding == ops_grid_padding_mode::border) {
        x = std::clamp<int64_t>(x, 0, d.input_width - 1); y = std::clamp<int64_t>(y, 0, d.input_height - 1);
    } else if (x < 0 || x >= d.input_width || y < 0 || y >= d.input_height) return 0.0f;
    return load<Type>(input, x * input->nb[0] + y * input->nb[1] + c * input->nb[2] + n * input->nb[3]);
}
template <ggml_type InputType, ggml_type GridType>
bool execute(ggml_backend_t backend, ggml_tensor* output, const ggml_tensor* input,
             const ggml_tensor* grid, const ops_grid_sample_2d_desc& d) {
    const int64_t total = d.output_width * d.output_height * d.channels * d.batch;
    const int threads = backend_thread_count(backend);
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index = 0; index < total; ++index) {
        int64_t q = index; const int64_t ox = q % d.output_width; q /= d.output_width;
        const int64_t oy = q % d.output_height; q /= d.output_height;
        const int64_t c = q % d.channels; const int64_t n = q / d.channels;
        const size_t grid_base = ox * grid->nb[1] + oy * grid->nb[2] + n * grid->nb[3];
        float x = unnormalize(load<GridType>(grid, grid_base), d.input_width, d.align_corners);
        float y = unnormalize(load<GridType>(grid, grid_base + grid->nb[0]), d.input_height, d.align_corners);
        if (d.padding == ops_grid_padding_mode::reflection) {
            x = std::clamp(reflect(x, d.input_width, d.align_corners), 0.0f, float(d.input_width - 1));
            y = std::clamp(reflect(y, d.input_height, d.align_corners), 0.0f, float(d.input_height - 1));
        }
        float value;
        if (d.mode == ops_grid_sample_mode::nearest) {
            value = sample<InputType>(input, int64_t(std::nearbyint(x)), int64_t(std::nearbyint(y)), c, n, d);
        } else {
            const int64_t x0 = int64_t(std::floor(x)), y0 = int64_t(std::floor(y));
            const float wx = x - x0, wy = y - y0;
            value = sample<InputType>(input, x0, y0, c, n, d) * (1 - wx) * (1 - wy) +
                    sample<InputType>(input, x0 + 1, y0, c, n, d) * wx * (1 - wy) +
                    sample<InputType>(input, x0, y0 + 1, c, n, d) * (1 - wx) * wy +
                    sample<InputType>(input, x0 + 1, y0 + 1, c, n, d) * wx * wy;
        }
        store<InputType>(output, ox * output->nb[0] + oy * output->nb[1] + c * output->nb[2] + n * output->nb[3], value);
    }
    return true;
}
template <ggml_type InputType>
bool dispatch_grid(ggml_backend_t b, ggml_tensor* o, const ggml_tensor* i, const ggml_tensor* g,
                   const ops_grid_sample_2d_desc& d) {
    return g->type == GGML_TYPE_F32 ? execute<InputType, GGML_TYPE_F32>(b, o, i, g, d)
         : g->type == GGML_TYPE_F16 ? execute<InputType, GGML_TYPE_F16>(b, o, i, g, d) : false;
}
} // namespace

bool ops_cpu_op_grid_sample_2d(ggml_backend_t backend, ggml_tensor* node) {
    if (!node || !node->src[0] || !node->src[1]) return false;
    ops_grid_sample_2d_params p{}; std::memcpy(&p, node->op_params, sizeof(p));
    ggml_tensor* s[] = {node->src[0], node->src[1]};
    ops_request r = {ggml_backend_get_device(backend), static_cast<int>(node->op), s, 2, &p, sizeof(p), node};
    ops_grid_sample_2d_desc d; if (!ops_validate_grid_sample_2d(r, &d)) return false;
    switch (s[0]->type) {
    case GGML_TYPE_F32: return dispatch_grid<GGML_TYPE_F32>(backend, node, s[0], s[1], d);
    case GGML_TYPE_F16: return dispatch_grid<GGML_TYPE_F16>(backend, node, s[0], s[1], d);
    case GGML_TYPE_BF16: return dispatch_grid<GGML_TYPE_BF16>(backend, node, s[0], s[1], d);
    default: return false;
    }
}
} // namespace ggml_ops_ext::cpu
