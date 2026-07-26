// Benchmark for the Snake activation: SIMD-dispatched op (AVX2 sine on
// capable hosts) versus the original scalar std::sin loop, on a
// vocoder-representative tensor size.
#include "ops/ops.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#ifdef _WIN32
#define OPS_IMPORT extern "C" __declspec(dllimport)
#else
#define OPS_IMPORT extern "C"
#endif
OPS_IMPORT void ggml_ops_ext_cpu_init();

namespace {

// The pre-SIMD reference implementation, kept verbatim for comparison.
void scalar_snake(const int64_t n, float* y, const float* x, const float alpha) {
    const float inv_alpha = 1.0f / alpha;
    for (int64_t i = 0; i < n; ++i) {
        const float value = x[i];
        const float sine = std::sin(alpha * value);
        y[i] = value + sine * sine * inv_alpha;
    }
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

    // Vocoder-shaped tensor: [samples, channels] as one contiguous block.
    constexpr int64_t elements = int64_t(512) * 16384;
    constexpr float alpha = 0.7f;
    constexpr int repeats = 20;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input(elements);
    for (float& value : input) value = dist(rng);
    std::vector<float> output(elements);

    // Scalar baseline (single thread, like the old per-row inner loop).
    scalar_snake(elements, output.data(), input.data(), alpha); // warm-up
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        scalar_snake(elements, output.data(), input.data(), alpha);
    }
    const double scalar_time = seconds_since(start) / repeats;

    // Op path (SIMD dispatch + OMP threading).
    ggml_context* ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, elements);
    ggml_tensor* out = ggml_ops_snake(ctx, x, alpha, backend);
    if (!out) {
        std::cerr << "snake op not supported on CPU backend\n";
        return 1;
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));

    ggml_ops_ext::ops_backend_graph_compute(backend, graph); // warm-up
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    }
    const double op_time = seconds_since(start) / repeats;

    // Verify agreement while we are here.
    std::vector<float> op_output(elements);
    ggml_backend_tensor_get(out, op_output.data(), 0, op_output.size() * sizeof(float));
    scalar_snake(elements, output.data(), input.data(), alpha);
    float max_error = 0.0f;
    for (int64_t i = 0; i < elements; ++i) {
        max_error = std::max(max_error, std::abs(op_output[i] - output[i]));
    }

    const double giga = 1e-9 * double(elements);
    std::cout << "elements            : " << elements << "\n";
    std::cout << "scalar std::sin     : " << scalar_time * 1e3 << " ms  ("
              << giga / scalar_time << " Gelem/s)\n";
    std::cout << "snake op (SIMD+OMP) : " << op_time * 1e3 << " ms  ("
              << giga / op_time << " Gelem/s)\n";
    std::cout << "speedup             : " << scalar_time / op_time << "x\n";
    std::cout << "max abs difference  : " << max_error << "\n";

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_ops_ext::release_ops_hook();
    return 0;
}
