// Numerical tests for nn::KVCache (prefill + decode across every cache type
// KVCache::allocate accepts) and nn::LayerNorm::forward_residual (the fused
// residual + norm path). Everything is compared against plain host references.
#include "nn/nn.h"
#include "ops/ops.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
#define OPS_IMPORT extern "C" __declspec(dllimport)
#else
#define OPS_IMPORT extern "C"
#endif
OPS_IMPORT void ggml_ops_ext_cpu_init();

namespace {

bool check_error(const std::string& label, const std::vector<float>& actual,
                 const std::vector<float>& expected, float tolerance) {
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    const bool passed = maximum <= tolerance;
    std::cout << label << " error=" << maximum << (passed ? " PASSED\n" : " FAILED\n");
    return passed;
}

std::vector<float> random_values(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> values(count);
    for (float& value : values) value = dist(rng);
    return values;
}

// Mirror the cache-write storage exactly: what comes back out of the cache is
// the quantized representation, so the reference must attend over the same
// values. Scales are stored as F16, matching the kernels.
std::vector<float> cache_roundtrip(const std::vector<float>& values, ggml_type type) {
    std::vector<float> result(values.size());
    if (type == GGML_TYPE_F32) {
        return values;
    }
    if (type == GGML_TYPE_F16) {
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]));
        }
        return result;
    }
    constexpr size_t block = 32;
    for (size_t start = 0; start < values.size(); start += block) {
        const size_t count = std::min(block, values.size() - start);
        if (type == GGML_TYPE_Q8_0) {
            float amax = 0.0f;
            for (size_t i = 0; i < count; ++i) amax = std::max(amax, std::abs(values[start + i]));
            const float d = amax / 127.0f;
            const float d_stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for (size_t i = 0; i < count; ++i) {
                const int q = d == 0.0f
                    ? 0 : static_cast<int>(std::lrint(values[start + i] / d));
                result[start + i] = static_cast<float>(q) * d_stored;
            }
        } else { // Q4_0: signed extreme, d = max / -8, codes 0..15 biased by 8.
            float amax = 0.0f;
            float smax = 0.0f;
            for (size_t i = 0; i < count; ++i) {
                const float value = values[start + i];
                if (std::abs(value) > amax) {
                    amax = std::abs(value);
                    smax = value;
                }
            }
            const float d = smax / -8.0f;
            const float d_stored = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for (size_t i = 0; i < count; ++i) {
                const float scaled = d == 0.0f ? 0.0f : values[start + i] / d;
                const int q = std::max(0, std::min(15, static_cast<int>(scaled + 8.5f)));
                result[start + i] = static_cast<float>(q - 8) * d_stored;
            }
        }
    }
    return result;
}

// Reference attention over [head_dim, kv_len, heads] caches for queries
// [head_dim, q_len, heads]; no mask, full attention.
std::vector<float> reference_attention(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    int64_t head_dim, int64_t q_len, int64_t kv_len, int64_t heads, float scale) {
    std::vector<float> output(static_cast<size_t>(head_dim * q_len * heads));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t qi = 0; qi < q_len; ++qi) {
            const float* q_row = q.data() + (head * q_len + qi) * head_dim;
            std::vector<float> scores(static_cast<size_t>(kv_len));
            float max_score = -1e30f;
            for (int64_t ki = 0; ki < kv_len; ++ki) {
                const float* k_row = k.data() + (head * kv_len + ki) * head_dim;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) dot += q_row[d] * k_row[d];
                scores[ki] = dot * scale;
                max_score = std::max(max_score, scores[ki]);
            }
            float total = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - max_score);
                total += score;
            }
            float* out_row = output.data() + (head * q_len + qi) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (int64_t ki = 0; ki < kv_len; ++ki) {
                    const float* v_row = v.data() + (head * kv_len + ki) * head_dim;
                    value += (scores[ki] / total) * v_row[d];
                }
                out_row[d] = value;
            }
        }
    }
    return output;
}

bool run_kv_cache(ggml_backend_t backend, ggml_type cache_type, const char* type_name,
                  float tolerance) {
    constexpr int head_dim = 32;
    constexpr int heads = 2;
    constexpr int layers = 1;
    constexpr int max_len = 16;
    constexpr int prefill_len = 5;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::string label = std::string("KVCache[") + type_name + "]";

    nn::KVCache cache;
    if (!cache.allocate(backend, head_dim, max_len, heads, layers,
                        {cache_type, cache_type})) {
        std::cout << label << " allocate FAILED\n";
        return false;
    }

    const auto q_data = random_values(static_cast<size_t>(head_dim * prefill_len * heads), 11);
    const auto k_data = random_values(static_cast<size_t>(head_dim * prefill_len * heads), 22);
    const auto v_data = random_values(static_cast<size_t>(head_dim * prefill_len * heads), 33);

    bool passed = true;
    // --- Prefill: write [prefill_len] tokens and attend over them.
    {
        nn::Context context(16 * 1024 * 1024);
        nn::Executor executor(backend);
        ggml_tensor* q = context.input<float>(
            "q", {head_dim, prefill_len, heads}, nn::data::borrow(q_data));
        ggml_tensor* new_k = context.input<float>(
            "k", {head_dim, prefill_len, heads}, nn::data::borrow(k_data));
        ggml_tensor* new_v = context.input<float>(
            "v", {head_dim, prefill_len, heads}, nn::data::borrow(v_data));
        ggml_tensor* out = cache.prefill_attention(
            context, 0, q, new_k, new_v, scale, backend);
        ggml_cgraph* graph = context.build(out);
        executor.prepare(context, graph);
        executor.compute(context, graph);

        std::vector<float> actual(q_data.size());
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        const auto expected = reference_attention(
            q_data, cache_roundtrip(k_data, cache_type), cache_roundtrip(v_data, cache_type),
            head_dim, prefill_len, prefill_len, heads, scale);
        passed &= check_error(label + " prefill", actual, expected, tolerance);
    }

    // --- Decode: append one token at position prefill_len and attend over all
    // prefill_len + 1 valid entries. This reads the cache the prefill graph
    // wrote, so it also verifies the write actually happened (the old API
    // could silently skip it).
    {
        const auto q1 = random_values(static_cast<size_t>(head_dim * heads), 44);
        const auto k1 = random_values(static_cast<size_t>(head_dim * heads), 55);
        const auto v1 = random_values(static_cast<size_t>(head_dim * heads), 66);
        const int32_t position_value = prefill_len;
        const int32_t valid_value = prefill_len + 1;

        nn::Context context(16 * 1024 * 1024);
        nn::Executor executor(backend);
        ggml_tensor* q = context.input<float>(
            "q1", {head_dim, 1, heads}, nn::data::borrow(q1));
        ggml_tensor* new_k = context.input<float>(
            "k1", {head_dim, 1, heads}, nn::data::borrow(k1));
        ggml_tensor* new_v = context.input<float>(
            "v1", {head_dim, 1, heads}, nn::data::borrow(v1));
        ggml_tensor* position = context.input<int32_t>(
            "position", {1}, nn::data::borrow(&position_value, 1));
        ggml_tensor* valid_length = context.input<int32_t>(
            "valid_length", {1}, nn::data::borrow(&valid_value, 1));
        ggml_tensor* out = cache.decode_attention(
            context, 0, q, new_k, new_v, position, valid_length, scale, backend);
        ggml_cgraph* graph = context.build(out);
        executor.prepare(context, graph);
        executor.compute(context, graph);

        std::vector<float> actual(q1.size());
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));

        // Host reference over the concatenated per-head sequences; every cache
        // entry (prefill block and the new token) went through cache storage.
        const auto k_stored = cache_roundtrip(k_data, cache_type);
        const auto v_stored = cache_roundtrip(v_data, cache_type);
        const auto k1_stored = cache_roundtrip(k1, cache_type);
        const auto v1_stored = cache_roundtrip(v1, cache_type);
        const int64_t kv_len = valid_value;
        std::vector<float> k_all(static_cast<size_t>(head_dim * kv_len * heads));
        std::vector<float> v_all(k_all.size());
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t token = 0; token < prefill_len; ++token) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    k_all[(head * kv_len + token) * head_dim + d] =
                        k_stored[(head * prefill_len + token) * head_dim + d];
                    v_all[(head * kv_len + token) * head_dim + d] =
                        v_stored[(head * prefill_len + token) * head_dim + d];
                }
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                k_all[(head * kv_len + prefill_len) * head_dim + d] = k1_stored[head * head_dim + d];
                v_all[(head * kv_len + prefill_len) * head_dim + d] = v1_stored[head * head_dim + d];
            }
        }
        const auto expected = reference_attention(
            q1, k_all, v_all, head_dim, 1, kv_len, heads, scale);
        passed &= check_error(label + " decode", actual, expected, tolerance);
    }
    return passed;
}

bool run_layer_norm_residual(ggml_backend_t backend, bool with_affine) {
    constexpr int64_t width = 24;
    constexpr int64_t rows = 4;
    constexpr float eps = 1e-5f;
    const auto x_data = random_values(static_cast<size_t>(width * rows), 77);
    const auto res_data = random_values(static_cast<size_t>(width * rows), 88);
    const auto gamma_data = random_values(static_cast<size_t>(width), 99);
    const auto beta_data = random_values(static_cast<size_t>(width), 111);

    nn::Context context(4 * 1024 * 1024);
    nn::Executor executor(backend);
    ggml_tensor* x = context.input<float>("x", {width, rows}, nn::data::borrow(x_data));
    ggml_tensor* residual =
        context.input<float>("residual", {width, rows}, nn::data::borrow(res_data));
    ggml_tensor* gamma = with_affine
        ? context.input<float>("gamma", {width}, nn::data::borrow(gamma_data)) : nullptr;
    ggml_tensor* beta = with_affine
        ? context.input<float>("beta", {width}, nn::data::borrow(beta_data)) : nullptr;

    nn::LayerNorm norm(gamma, beta, eps);
    ggml_tensor* out = norm.forward_residual(context.native_handle(), x, residual, backend);
    ggml_cgraph* graph = context.build(out);
    executor.prepare(context, graph);
    executor.compute(context, graph);

    std::vector<float> actual(x_data.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));

    std::vector<float> expected(x_data.size());
    for (int64_t row = 0; row < rows; ++row) {
        float mean = 0.0f;
        std::vector<float> summed(static_cast<size_t>(width));
        for (int64_t i = 0; i < width; ++i) {
            summed[i] = x_data[row * width + i] + res_data[row * width + i];
            mean += summed[i];
        }
        mean /= static_cast<float>(width);
        float variance = 0.0f;
        for (int64_t i = 0; i < width; ++i) {
            variance += (summed[i] - mean) * (summed[i] - mean);
        }
        variance /= static_cast<float>(width);
        const float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int64_t i = 0; i < width; ++i) {
            float value = (summed[i] - mean) * inv_std;
            if (with_affine) value = value * gamma_data[i] + beta_data[i];
            expected[row * width + i] = value;
        }
    }
    return check_error(std::string("LayerNorm.forward_residual") +
                           (with_affine ? " affine" : " plain"),
                       actual, expected, 3e-5f);
}

} // namespace

int main() {
    ggml_ops_ext_cpu_init();
    ggml_backend_load_all();
    ggml_ops_ext::acquire_ops_hook();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        std::cerr << "failed to initialize CPU backend\n";
        return 1;
    }

    bool passed = true;
    // The reference replays cache storage (quantize + F16 scales), so these
    // bounds cover only the attention arithmetic, not quantization loss.
    passed &= run_kv_cache(backend, GGML_TYPE_F32, "F32", 2e-4f);
    passed &= run_kv_cache(backend, GGML_TYPE_F16, "F16", 2e-3f);
    passed &= run_kv_cache(backend, GGML_TYPE_Q8_0, "Q8_0", 2e-3f);
    passed &= run_kv_cache(backend, GGML_TYPE_Q4_0, "Q4_0", 2e-3f);
    passed &= run_layer_norm_residual(backend, true);
    passed &= run_layer_norm_residual(backend, false);

    ggml_backend_free(backend);
    ggml_ops_ext::release_ops_hook();
    std::cout << (passed ? "all nn kv-cache tests passed\n" : "nn kv-cache tests FAILED\n");
    return passed ? 0 : 1;
}
