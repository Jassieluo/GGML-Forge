// Numerical golden test for nn::MultiHeadAttention::forward: the full
// projection -> split-heads -> attention -> merge-heads -> output-projection
// pipeline is compared against a plain host reference. Guards attention
// refactors with an end-to-end oracle instead of relying on downstream
// model output staying plausible.
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

// weight layout matches nn::functional::linear (ggml_mul_mat(weight, input)):
// ne[0] = in_dim, so W(i -> o) sits at index o * in_dim + i.
std::vector<float> host_linear(const std::vector<float>& x, const std::vector<float>& w,
                               const std::vector<float>& b, int64_t tokens, int64_t in_dim,
                               int64_t out_dim) {
    std::vector<float> out(static_cast<size_t>(tokens * out_dim));
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < out_dim; ++o) {
            float value = b[o];
            for (int64_t i = 0; i < in_dim; ++i) {
                value += w[o * in_dim + i] * x[t * in_dim + i];
            }
            out[t * out_dim + o] = value;
        }
    }
    return out;
}

// Full attention (no mask) over per-head sequences laid out as
// [head_dim, len, heads], i.e. index (head * len + token) * head_dim + d.
std::vector<float> host_attention(
    const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
    int64_t head_dim, int64_t len, int64_t heads, float scale) {
    std::vector<float> output(static_cast<size_t>(head_dim * len * heads));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t qi = 0; qi < len; ++qi) {
            const float* q_row = q.data() + (head * len + qi) * head_dim;
            std::vector<float> scores(static_cast<size_t>(len));
            float max_score = -1e30f;
            for (int64_t ki = 0; ki < len; ++ki) {
                const float* k_row = k.data() + (head * len + ki) * head_dim;
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
            float* out_row = output.data() + (head * len + qi) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (int64_t ki = 0; ki < len; ++ki) {
                    const float* v_row = v.data() + (head * len + ki) * head_dim;
                    value += (scores[ki] / total) * v_row[d];
                }
                out_row[d] = value;
            }
        }
    }
    return output;
}

bool run_mha(ggml_backend_t backend) {
    constexpr int heads = 2;
    constexpr int head_dim = 16;
    constexpr int64_t hidden = heads * head_dim;
    constexpr int64_t tokens = 6;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const auto x_data = random_values(static_cast<size_t>(hidden * tokens), 7);
    const auto wq = random_values(static_cast<size_t>(hidden * hidden), 17);
    const auto wk = random_values(static_cast<size_t>(hidden * hidden), 27);
    const auto wv = random_values(static_cast<size_t>(hidden * hidden), 37);
    const auto wo = random_values(static_cast<size_t>(hidden * hidden), 47);
    const auto bq = random_values(static_cast<size_t>(hidden), 57);
    const auto bk = random_values(static_cast<size_t>(hidden), 67);
    const auto bv = random_values(static_cast<size_t>(hidden), 77);
    const auto bo = random_values(static_cast<size_t>(hidden), 87);

    nn::Context context(16 * 1024 * 1024);
    nn::Executor executor(backend);

    nn::MultiHeadAttention mha(heads, head_dim);
    mha.q_proj.weight.bind(context.input<float>("wq", {hidden, hidden}, nn::data::borrow(wq)));
    mha.q_proj.bias.bind(context.input<float>("bq", {hidden}, nn::data::borrow(bq)));
    mha.k_proj.weight.bind(context.input<float>("wk", {hidden, hidden}, nn::data::borrow(wk)));
    mha.k_proj.bias.bind(context.input<float>("bk", {hidden}, nn::data::borrow(bk)));
    mha.v_proj.weight.bind(context.input<float>("wv", {hidden, hidden}, nn::data::borrow(wv)));
    mha.v_proj.bias.bind(context.input<float>("bv", {hidden}, nn::data::borrow(bv)));
    mha.out_proj.weight.bind(context.input<float>("wo", {hidden, hidden}, nn::data::borrow(wo)));
    mha.out_proj.bias.bind(context.input<float>("bo", {hidden}, nn::data::borrow(bo)));

    ggml_tensor* x = context.input<float>("x", {hidden, tokens}, nn::data::borrow(x_data));
    ggml_tensor* out = mha.forward(context, x, nullptr, backend);
    ggml_cgraph* graph = context.build(out);
    executor.prepare(context, graph);
    executor.compute(context, graph);

    std::vector<float> actual(static_cast<size_t>(hidden * tokens));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));

    // Host reference: project, split heads, attend, merge heads, project.
    const auto q_proj = host_linear(x_data, wq, bq, tokens, hidden, hidden);
    const auto k_proj = host_linear(x_data, wk, bk, tokens, hidden, hidden);
    const auto v_proj = host_linear(x_data, wv, bv, tokens, hidden, hidden);
    auto split = [&](const std::vector<float>& proj) {
        std::vector<float> value(proj.size());
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t t = 0; t < tokens; ++t) {
                for (int64_t d = 0; d < head_dim; ++d) {
                    value[(head * tokens + t) * head_dim + d] =
                        proj[t * hidden + head * head_dim + d];
                }
            }
        }
        return value;
    };
    const auto attn = host_attention(
        split(q_proj), split(k_proj), split(v_proj), head_dim, tokens, heads, scale);
    std::vector<float> merged(static_cast<size_t>(hidden * tokens));
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t t = 0; t < tokens; ++t) {
            for (int64_t d = 0; d < head_dim; ++d) {
                merged[t * hidden + head * head_dim + d] =
                    attn[(head * tokens + t) * head_dim + d];
            }
        }
    }
    const auto expected = host_linear(merged, wo, bo, tokens, hidden, hidden);

    return check_error("MultiHeadAttention.forward", actual, expected, 1e-4f);
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
    const bool passed = run_mha(backend);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
