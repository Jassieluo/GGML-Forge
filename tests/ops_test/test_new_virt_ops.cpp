// Tests for GGML_OP_OPS_VIRT_FUSED_NORM_ACT / POS_ENCODING / SAMPLE_DIST.
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#define OPS_IMPORT extern "C" __declspec(dllimport)
#else
#define OPS_IMPORT extern "C"
#endif
OPS_IMPORT void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
OPS_IMPORT void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
OPS_IMPORT void ggml_ops_ext_sycl_init();
#endif

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

// Kernel-required builders return nullptr when a backend has no kernel for the
// op. CPU must always provide one, so nullptr is only tolerated on GPU backends.
bool skip_unsupported(const std::string& backend_name, const char* op_name) {
    const bool is_cpu = backend_name.rfind("CPU", 0) == 0;
    std::cout << backend_name << " " << op_name
              << (is_cpu ? " FAILED (missing CPU kernel)\n" : " SKIPPED (no kernel)\n");
    return !is_cpu;
}

float reference_activation(float value, int32_t activation) {
    switch (activation) {
        case 0: return value / (1.0f + std::exp(-value));
        case 1:
            return 0.5f * value *
                   (1.0f + std::tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
        case 2: return value > 0.0f ? value : 0.0f;
        default: return value;
    }
}

bool run_fused_norm_act(ggml_backend_t backend, const std::string& name, bool with_residual,
                        ggml_ops_gate_activation activation) {
    constexpr int64_t width = 24;
    constexpr int64_t rows = 5;
    constexpr float eps = 1e-5f;
    std::vector<float> x(width * rows), gamma(width), beta(width), residual(width * rows);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.13f) * 1.7f;
    for (size_t i = 0; i < residual.size(); ++i) residual[i] = std::cos(float(i) * 0.29f);
    for (size_t i = 0; i < gamma.size(); ++i) gamma[i] = 0.8f + 0.05f * float(i % 5);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = -0.1f + 0.03f * float(i % 7);

    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    ggml_tensor* gamma_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_tensor* beta_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    ggml_tensor* residual_t =
        with_residual ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows) : nullptr;
    ggml_tensor* out = ggml_ops_fused_norm_act(ctx, x_t, gamma_t, beta_t, residual_t, eps,
                                               activation, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name, "FusedNormAct"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x_t, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(gamma_t, gamma.data(), 0, gamma.size() * sizeof(float));
    ggml_backend_tensor_set(beta_t, beta.data(), 0, beta.size() * sizeof(float));
    if (residual_t) {
        ggml_backend_tensor_set(residual_t, residual.data(), 0, residual.size() * sizeof(float));
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed =
        ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> expected(x.size());
    for (int64_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        std::vector<float> row(width);
        for (int64_t i = 0; i < width; ++i) {
            row[i] = x[r * width + i] + (with_residual ? residual[r * width + i] : 0.0f);
            mean += row[i];
        }
        mean /= float(width);
        float variance = 0.0f;
        for (int64_t i = 0; i < width; ++i) variance += (row[i] - mean) * (row[i] - mean);
        variance /= float(width);
        const float inv_std = 1.0f / std::sqrt(variance + eps);
        for (int64_t i = 0; i < width; ++i) {
            expected[r * width + i] = reference_activation(
                (row[i] - mean) * inv_std * gamma[i] + beta[i], int32_t(activation));
        }
    }
    std::vector<float> actual(x.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    passed &= check_error(name + " FusedNormAct act=" + std::to_string(int(activation)) +
                              (with_residual ? " +res" : ""),
                          actual, expected, 3e-5f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

bool run_pos_encoding(ggml_backend_t backend, const std::string& name, bool with_position) {
    constexpr int64_t width = 17;
    constexpr int64_t tokens = 9;
    constexpr int64_t heads = 2;
    constexpr float base = 10000.0f;
    constexpr int32_t offset = 3;
    constexpr int32_t dynamic = 5;
    std::vector<float> x(width * tokens * heads);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.07f);

    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* x_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, tokens, heads);
    ggml_tensor* pos_t = with_position ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1) : nullptr;
    ggml_tensor* out = ggml_ops_pos_encoding(ctx, x_t, pos_t, base, offset, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name, "PosEncoding"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x_t, x.data(), 0, x.size() * sizeof(float));
    if (pos_t) ggml_backend_tensor_set(pos_t, &dynamic, 0, sizeof(dynamic));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed =
        ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    const int64_t start = offset + (with_position ? dynamic : 0);
    std::vector<float> expected(x.size());
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t i = 0; i < width; ++i) {
                const int64_t pair = i - (i % 2);
                const float angle = float(start + t) *
                    std::exp(-std::log(base) * float(pair) / float(width));
                const float pe = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
                const size_t index = size_t((h * tokens + t) * width + i);
                expected[index] = x[index] + pe;
            }
        }
    }
    std::vector<float> actual(x.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    passed &= check_error(name + " PosEncoding" + (with_position ? " dyn" : ""), actual,
                          expected, 3e-5f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int32_t reference_sample(const std::vector<float>& values, float u, int32_t top_k, float top_p,
                         float temperature) {
    const int64_t vocab = int64_t(values.size());
    u = std::min(std::max(u, 0.0f), 0.999999f);
    std::vector<int32_t> candidates(static_cast<size_t>(vocab));
    std::iota(candidates.begin(), candidates.end(), 0);
    const auto by_logit_desc = [&](int32_t a, int32_t b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    };
    const int64_t keep = top_k > 0 ? std::min<int64_t>(top_k, vocab) : vocab;
    std::sort(candidates.begin(), candidates.end(), by_logit_desc);
    candidates.resize(size_t(keep));
    std::vector<float> probs(candidates.size());
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((values[candidates[i]] - values[candidates[0]]) / temperature);
        total += probs[i];
    }
    for (float& p : probs) p /= total;
    size_t kept = candidates.size();
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= top_p) { kept = i + 1; break; }
        }
    }
    float kept_total = 0.0f;
    for (size_t i = 0; i < kept; ++i) kept_total += probs[i];
    const float threshold = u * kept_total;
    float cumulative = 0.0f;
    for (size_t i = 0; i < kept; ++i) {
        cumulative += probs[i];
        if (cumulative > threshold) return candidates[i];
    }
    return candidates[kept - 1];
}

bool run_sample(ggml_backend_t backend, const std::string& name, int32_t top_k, float top_p,
                float temperature, float u) {
    constexpr int64_t vocab = 96;
    std::vector<float> logits(static_cast<size_t>(vocab));
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = 3.0f * std::sin(float(i) * 0.37f) + 0.02f * float(i % 11);
    }

    ggml_context* ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor* logits_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_tensor* uniform_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor* out =
        ggml_ops_sample_dist(ctx, logits_t, uniform_t, top_k, top_p, temperature, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name, "SampleDist"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(logits_t, logits.data(), 0, logits.size() * sizeof(float));
    ggml_backend_tensor_set(uniform_t, &u, 0, sizeof(u));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed =
        ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    int32_t actual = -1;
    ggml_backend_tensor_get(out, &actual, 0, sizeof(actual));
    const int32_t expected = reference_sample(logits, u, top_k, top_p, temperature);
    const bool match = passed && actual == expected;
    std::cout << name << " SampleDist k=" << top_k << " p=" << top_p << " T=" << temperature
              << " u=" << u << " got=" << actual << " want=" << expected
              << (match ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return match;
}

} // namespace

int main() {
    ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    ggml_ops_ext_sycl_init();
#endif
    ggml_backend_load_all();
    ggml_ops_ext::acquire_ops_hook();
    bool passed = true;
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        passed &= run_fused_norm_act(backend, name, false, ggml_ops_gate_activation::identity);
        passed &= run_fused_norm_act(backend, name, true, ggml_ops_gate_activation::silu);
        passed &= run_fused_norm_act(backend, name, true, ggml_ops_gate_activation::gelu);
        passed &= run_fused_norm_act(backend, name, false, ggml_ops_gate_activation::relu);
        passed &= run_pos_encoding(backend, name, false);
        passed &= run_pos_encoding(backend, name, true);
        passed &= run_sample(backend, name, 0, 1.0f, 1.0f, 0.5f);
        passed &= run_sample(backend, name, 8, 1.0f, 0.8f, 0.31f);
        passed &= run_sample(backend, name, 0, 0.9f, 1.0f, 0.77f);
        passed &= run_sample(backend, name, 12, 0.85f, 0.7f, 0.999f);
        passed &= run_sample(backend, name, 1, 1.0f, 1.0f, 0.42f);
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
