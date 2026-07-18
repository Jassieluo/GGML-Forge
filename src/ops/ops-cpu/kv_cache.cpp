#include "ops/ops.h"
#include "ggml-impl.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace ggml_ops_ext::cpu {

static bool write_cache_rows(ggml_tensor* cache, const ggml_tensor* values, int32_t position) {
    const int64_t width = cache->ne[0];
    const int64_t count = values->ne[1];
    if (position < 0 || position + count > cache->ne[1]) return false;
    const ggml_type_traits* source_traits = ggml_get_type_traits(values->type);
    if (!source_traits) return false;

    std::vector<float> row(static_cast<size_t>(width));
    for (int64_t batch = 0; batch < values->ne[3]; ++batch) {
        for (int64_t head = 0; head < values->ne[2]; ++head) {
            for (int64_t token = 0; token < count; ++token) {
                const char* source = static_cast<const char*>(values->data) +
                    batch * values->nb[3] + head * values->nb[2] + token * values->nb[1];
                char* destination = static_cast<char*>(cache->data) +
                    batch * cache->nb[3] + head * cache->nb[2] + (position + token) * cache->nb[1];
                if (values->type == GGML_TYPE_F32) {
                    std::memcpy(row.data(), source, static_cast<size_t>(width) * sizeof(float));
                } else {
                    source_traits->to_float(source, row.data(), width);
                }
                if (cache->type == GGML_TYPE_F32) {
                    std::memcpy(destination, row.data(), static_cast<size_t>(width) * sizeof(float));
                } else if (cache->type == GGML_TYPE_F16) {
                    auto* output = reinterpret_cast<ggml_fp16_t*>(destination);
                    for (int64_t i = 0; i < width; ++i) output[i] = ggml_fp32_to_fp16(row[i]);
                } else if (cache->type == GGML_TYPE_Q8_0) {
                    auto* output = reinterpret_cast<block_q8_0*>(destination);
                    for (int64_t block = 0; block < width / QK8_0; ++block) {
                        float amax = 0.0f;
                        for (int i = 0; i < QK8_0; ++i) amax = std::max(amax, std::abs(row[block * QK8_0 + i]));
                        const float d = amax / 127.0f;
                        output[block].d = ggml_fp32_to_fp16(d);
                        for (int i = 0; i < QK8_0; ++i) output[block].qs[i] = static_cast<int8_t>(
                            std::lrint(d == 0.0f ? 0.0f : row[block * QK8_0 + i] / d));
                    }
                } else if (cache->type == GGML_TYPE_Q4_0) {
                    auto* output = reinterpret_cast<block_q4_0*>(destination);
                    for (int64_t block = 0; block < width / QK4_0; ++block) {
                        float amax = 0.0f;
                        for (int i = 0; i < QK4_0; ++i) amax = std::max(amax, std::abs(row[block * QK4_0 + i]));
                        const float d = amax / 8.0f;
                        output[block].d = ggml_fp32_to_fp16(d);
                        for (int i = 0; i < QK4_0 / 2; ++i) {
                            const auto quant = [&](float value) {
                                return std::clamp(static_cast<int>(std::lrint(d == 0.0f ? 0.0f : value / d)) + 8, 0, 15);
                            };
                            const int low = quant(row[block * QK4_0 + i]);
                            const int high = quant(row[block * QK4_0 + i + QK4_0 / 2]);
                            output[block].qs[i] = static_cast<uint8_t>(low | (high << 4));
                        }
                    }
                } else {
                    return false;
                }
            }
        }
    }
    return true;
}

bool ops_cpu_op_kv_cache_update(ggml_backend_t, ggml_tensor* node) {
    ops_kv_cache_update_params params;
    if (!ops_extract_kv_cache_update_params(node, params)) return false;
    const int32_t position = *static_cast<const int32_t*>(params.position->data);
    if (!write_cache_rows(params.cache_k, params.new_k, position) ||
        !write_cache_rows(params.cache_v, params.new_v, position)) return false;
    *static_cast<int32_t*>(node->data) = position + static_cast<int32_t>(params.new_k->ne[1]);
    return true;
}

} // namespace ggml_ops_ext::cpu
