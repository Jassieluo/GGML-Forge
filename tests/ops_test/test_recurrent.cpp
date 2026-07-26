// Tests for GGML_OP_OPS_VIRT_GRU / LSTM against a naive host recurrence with
// torch.nn.GRU / torch.nn.LSTM semantics. Suite selection keeps reruns
// scoped to the edited op:
//   test_recurrent          both
//   test_recurrent gru      GRU only
//   test_recurrent lstm     LSTM only
#include "ops/ops.h"

#include <cmath>
#include <cstring>
#include <iostream>
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

constexpr int64_t kInput = 24;
constexpr int64_t kHidden = 48;
constexpr int64_t kSteps = 32;

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

bool skip_unsupported(const std::string& backend_name, const char* op_name) {
    const bool is_cpu = backend_name.rfind("CPU", 0) == 0;
    std::cout << backend_name << " " << op_name
              << (is_cpu ? " FAILED (missing CPU kernel)\n" : " SKIPPED (no kernel)\n");
    return !is_cpu;
}

std::vector<float> pattern(size_t count, float step, float scale) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = scale * std::sin(float(i) * step + 0.3f);
    }
    return values;
}

float sigmoidf(float value) { return 1.0f / (1.0f + std::exp(-value)); }

// row-major matvec matching the contract weight layout: out[row] =
// bias[row] + sum_i w[row * in + i] * v[i].
void matvec(const float* w, const float* bias, const float* v, float* out,
            int64_t rows, int64_t in) {
    for (int64_t row = 0; row < rows; ++row) {
        float sum = bias ? bias[row] : 0.0f;
        for (int64_t i = 0; i < in; ++i) sum += w[row * in + i] * v[i];
        out[row] = sum;
    }
}

struct Weights {
    std::vector<float> w_ih, w_hh, b_ih, b_hh, h0, c0;
    Weights(int gates, uint32_t salt) {
        // Small magnitudes keep gate activations away from saturation so
        // implementation differences show up in the error metric.
        w_ih = pattern(static_cast<size_t>(gates * kHidden * kInput), 0.013f + salt * 1e-4f, 0.25f);
        w_hh = pattern(static_cast<size_t>(gates * kHidden * kHidden), 0.017f + salt * 1e-4f, 0.25f);
        b_ih = pattern(static_cast<size_t>(gates * kHidden), 0.11f, 0.1f);
        b_hh = pattern(static_cast<size_t>(gates * kHidden), 0.13f, 0.1f);
        h0 = pattern(static_cast<size_t>(kHidden), 0.21f, 0.5f);
        c0 = pattern(static_cast<size_t>(kHidden), 0.27f, 0.5f);
    }
};

std::vector<float> reference_gru(const std::vector<float>& x, const Weights& w,
                                 bool with_bias, bool with_state, bool reverse) {
    std::vector<float> out(static_cast<size_t>(kHidden * kSteps));
    std::vector<float> h(static_cast<size_t>(kHidden), 0.0f);
    if (with_state) h = w.h0;
    std::vector<float> gx(static_cast<size_t>(3 * kHidden)), rh(gx.size());
    for (int64_t step = 0; step < kSteps; ++step) {
        const int64_t t = reverse ? kSteps - 1 - step : step;
        matvec(w.w_ih.data(), with_bias ? w.b_ih.data() : nullptr,
               x.data() + t * kInput, gx.data(), 3 * kHidden, kInput);
        matvec(w.w_hh.data(), with_bias ? w.b_hh.data() : nullptr,
               h.data(), rh.data(), 3 * kHidden, kHidden);
        for (int64_t j = 0; j < kHidden; ++j) {
            const float r = sigmoidf(gx[j] + rh[j]);
            const float z = sigmoidf(gx[kHidden + j] + rh[kHidden + j]);
            const float n = std::tanh(gx[2 * kHidden + j] + r * rh[2 * kHidden + j]);
            h[j] = (1.0f - z) * n + z * h[j];
            out[t * kHidden + j] = h[j];
        }
    }
    return out;
}

std::vector<float> reference_lstm(const std::vector<float>& x, const Weights& w,
                                  bool with_bias, bool with_state, bool reverse) {
    std::vector<float> out(static_cast<size_t>(kHidden * kSteps));
    std::vector<float> h(static_cast<size_t>(kHidden), 0.0f);
    std::vector<float> c(static_cast<size_t>(kHidden), 0.0f);
    if (with_state) { h = w.h0; c = w.c0; }
    std::vector<float> gx(static_cast<size_t>(4 * kHidden)), rh(gx.size());
    for (int64_t step = 0; step < kSteps; ++step) {
        const int64_t t = reverse ? kSteps - 1 - step : step;
        matvec(w.w_ih.data(), with_bias ? w.b_ih.data() : nullptr,
               x.data() + t * kInput, gx.data(), 4 * kHidden, kInput);
        matvec(w.w_hh.data(), with_bias ? w.b_hh.data() : nullptr,
               h.data(), rh.data(), 4 * kHidden, kHidden);
        for (int64_t j = 0; j < kHidden; ++j) {
            const float i = sigmoidf(gx[j] + rh[j]);
            const float f = sigmoidf(gx[kHidden + j] + rh[kHidden + j]);
            const float g = std::tanh(gx[2 * kHidden + j] + rh[2 * kHidden + j]);
            const float o = sigmoidf(gx[3 * kHidden + j] + rh[3 * kHidden + j]);
            c[j] = f * c[j] + i * g;
            h[j] = o * std::tanh(c[j]);
            out[t * kHidden + j] = h[j];
        }
    }
    return out;
}

bool run_case(ggml_backend_t backend, const std::string& name, bool lstm,
              bool with_bias, bool with_state, bool reverse) {
    const int gates = lstm ? 4 : 3;
    const Weights weights(gates, lstm ? 7 : 3);
    const auto x_data = pattern(static_cast<size_t>(kInput * kSteps), 0.071f, 1.0f);

    ggml_context* ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInput, kSteps);
    ggml_tensor* w_ih = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kInput, gates * kHidden);
    ggml_tensor* w_hh = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kHidden, gates * kHidden);
    ggml_tensor* b_ih = with_bias
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gates * kHidden) : nullptr;
    ggml_tensor* b_hh = with_bias
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gates * kHidden) : nullptr;
    ggml_tensor* h0 = (with_state && with_bias)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden) : nullptr;
    ggml_tensor* c0 = (lstm && with_state && with_bias)
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kHidden) : nullptr;
    ggml_tensor* out = lstm
        ? ggml_ops_lstm(ctx, x, w_ih, w_hh, b_ih, b_hh, h0, c0, reverse, backend)
        : ggml_ops_gru(ctx, x, w_ih, w_hh, b_ih, b_hh, h0, reverse, backend);
    const char* op_name = lstm ? "LSTM" : "GRU";
    if (!out) { ggml_free(ctx); return skip_unsupported(name, op_name); }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(w_ih, weights.w_ih.data(), 0, weights.w_ih.size() * sizeof(float));
    ggml_backend_tensor_set(w_hh, weights.w_hh.data(), 0, weights.w_hh.size() * sizeof(float));
    if (b_ih) ggml_backend_tensor_set(b_ih, weights.b_ih.data(), 0, weights.b_ih.size() * sizeof(float));
    if (b_hh) ggml_backend_tensor_set(b_hh, weights.b_hh.data(), 0, weights.b_hh.size() * sizeof(float));
    if (h0) ggml_backend_tensor_set(h0, weights.h0.data(), 0, weights.h0.size() * sizeof(float));
    if (c0) ggml_backend_tensor_set(c0, weights.c0.data(), 0, weights.c0.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> actual(static_cast<size_t>(kHidden * kSteps));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    const auto expected = lstm
        ? reference_lstm(x_data, weights, with_bias, with_state && with_bias, reverse)
        : reference_gru(x_data, weights, with_bias, with_state && with_bias, reverse);

    std::string label = name + " " + op_name;
    if (with_bias) label += " +bias";
    if (with_state && with_bias) label += " +state";
    if (reverse) label += " reverse";
    passed &= check_error(label, actual, expected, 2e-4f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

} // namespace

int main(int argc, char** argv) {
    const std::string suite = argc > 1 ? argv[1] : "all";
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
        if (suite == "all" || suite == "gru") {
            passed &= run_case(backend, name, false, true, true, false);
            passed &= run_case(backend, name, false, true, false, true);
            passed &= run_case(backend, name, false, false, false, false);
        }
        if (suite == "all" || suite == "lstm") {
            passed &= run_case(backend, name, true, true, true, false);
            passed &= run_case(backend, name, true, true, false, true);
            passed &= run_case(backend, name, true, false, false, false);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
