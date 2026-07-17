#include "ops/ops.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
extern "C" __declspec(dllimport) void ggml_ops_ext_cpu_init();
#else
extern "C" void ggml_ops_ext_cpu_init();
#endif
#ifdef GGML_USE_CUDA
extern "C" void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
extern "C" void ggml_ops_ext_sycl_init();
#endif

namespace {

struct GraphCase {
    ggml_context* context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    ggml_tensor* output = nullptr;

    GraphCase() = default;
    GraphCase(const GraphCase&) = delete;
    GraphCase& operator=(const GraphCase&) = delete;
    GraphCase(GraphCase&& other) noexcept
        : context(std::exchange(other.context, nullptr)),
          buffer(std::exchange(other.buffer, nullptr)),
          graph(std::exchange(other.graph, nullptr)),
          output(std::exchange(other.output, nullptr)) {}

    ~GraphCase() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};

std::vector<uint8_t> quantize(ggml_type type, const std::vector<float>& source, int64_t rows, int64_t row_size) {
    std::vector<uint8_t> result(ggml_row_size(type, row_size) * rows);
    const size_t written = ggml_quantize_chunk(
        type, source.data(), result.data(), 0, rows, row_size, nullptr);
    if (written != result.size()) throw std::runtime_error("quantization failed");
    return result;
}

GraphCase build_case(
    ggml_backend_t backend,
    bool optimized,
    ggml_type cache_type,
    int head_dim,
    int q_len,
    int kv_len,
    int heads,
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data) {
    GraphCase result;
    result.context = ggml_init({32 * 1024 * 1024, nullptr, true});
    if (!result.context) throw std::runtime_error("context allocation failed");

    ggml_tensor* q = ggml_new_tensor_4d(result.context, GGML_TYPE_F32, head_dim, q_len, heads, 1);
    ggml_tensor* k = ggml_new_tensor_4d(result.context, cache_type, head_dim, kv_len, heads, 1);
    ggml_tensor* v = ggml_new_tensor_4d(result.context, cache_type, head_dim, kv_len, heads, 1);
    ggml_tensor* attention_k = k;
    ggml_tensor* attention_v = v;
    if (!optimized && cache_type != GGML_TYPE_F32) {
        attention_k = ggml_cont(result.context, ggml_cast(result.context, k, GGML_TYPE_F32));
        attention_v = ggml_cont(result.context, ggml_cast(result.context, v, GGML_TYPE_F32));
    }
    result.output = ggml_ops_attention(
        result.context, q, attention_k, attention_v, nullptr, nullptr,
        1.0f / std::sqrt(static_cast<float>(head_dim)), -1, optimized ? backend : nullptr);

    result.buffer = ggml_backend_alloc_ctx_tensors(result.context, backend);
    if (!result.buffer) throw std::runtime_error("backend allocation failed");
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    if (cache_type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
        ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    } else {
        const auto packed_k = quantize(cache_type, k_data, kv_len * heads, head_dim);
        const auto packed_v = quantize(cache_type, v_data, kv_len * heads, head_dim);
        ggml_backend_tensor_set(k, packed_k.data(), 0, packed_k.size());
        ggml_backend_tensor_set(v, packed_v.data(), 0, packed_v.size());
    }
    result.graph = ggml_new_graph(result.context);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

double measure(ggml_backend_t backend, GraphCase& graph_case, int iterations) {
    for (int i = 0; i < 5; ++i) {
        ggml_ops_ext::ops_backend_graph_compute(backend, graph_case.graph);
    }
    ggml_backend_synchronize(backend);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_ops_ext::ops_backend_graph_compute(backend, graph_case.graph);
        ggml_backend_synchronize(backend);
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

void benchmark(
    ggml_backend_t backend,
    const char* backend_name,
    const char* workload,
    ggml_type cache_type,
    int head_dim,
    int q_len,
    int kv_len,
    int heads,
    int iterations) {
    std::vector<float> q(static_cast<size_t>(head_dim * q_len * heads));
    std::vector<float> k(static_cast<size_t>(head_dim * kv_len * heads));
    std::vector<float> v(k.size());
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = std::cos(static_cast<float>(i) * 0.017f);
        v[i] = std::sin(static_cast<float>(i) * 0.019f + 0.3f);
    }

    GraphCase baseline = build_case(backend, false, cache_type, head_dim, q_len, kv_len, heads, q, k, v);
    GraphCase optimized = build_case(backend, true, cache_type, head_dim, q_len, kv_len, heads, q, k, v);
    const double baseline_us = measure(backend, baseline, iterations);
    const double optimized_us = measure(backend, optimized, iterations);
    std::vector<float> baseline_output(q.size());
    std::vector<float> optimized_output(q.size());
    ggml_backend_tensor_get(baseline.output, baseline_output.data(), 0, baseline_output.size() * sizeof(float));
    ggml_backend_tensor_get(optimized.output, optimized_output.data(), 0, optimized_output.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t i = 0; i < baseline_output.size(); ++i) {
        max_error = std::max(max_error, std::abs(baseline_output[i] - optimized_output[i]));
    }
    if (max_error > 2e-3f) throw std::runtime_error("attention benchmark output mismatch");
    std::cout << backend_name << " " << workload << " " << ggml_type_name(cache_type)
              << ": baseline=" << baseline_us << " us, streaming=" << optimized_us
              << " us, speedup=" << baseline_us / optimized_us << "x, max_error=" << max_error << "\n";
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
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = ggml_backend_dev_name(device) ? ggml_backend_dev_name(device) : "unknown";
        if (name.find("BLAS") != std::string::npos) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        benchmark(backend, name.c_str(), "decode", GGML_TYPE_F32, 32, 1, 512, 16, 100);
        benchmark(backend, name.c_str(), "decode", GGML_TYPE_F16, 32, 1, 512, 16, 100);
        benchmark(backend, name.c_str(), "decode", GGML_TYPE_Q4_0, 32, 1, 512, 16, 100);
        benchmark(backend, name.c_str(), "prefill", GGML_TYPE_F32, 64, 256, 256, 8, 20);
        benchmark(backend, name.c_str(), "prefill", GGML_TYPE_F16, 64, 256, 256, 8, 20);
        benchmark(backend, name.c_str(), "prefill", GGML_TYPE_Q4_0, 64, 256, 256, 8, 20);
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return 0;
}
