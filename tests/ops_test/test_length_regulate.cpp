// Tests for GGML_OP_OPS_VIRT_LENGTH_REGULATE: duration-based frame expansion
// including the clamp behaviors (zero durations, truncation, zero fill).
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

bool skip_unsupported(const std::string& backend_name) {
    const bool is_cpu = backend_name.rfind("CPU", 0) == 0;
    std::cout << backend_name << " LengthRegulate"
              << (is_cpu ? " FAILED (missing CPU kernel)\n" : " SKIPPED (no kernel)\n");
    return !is_cpu;
}

bool run_case(ggml_backend_t backend, const std::string& name,
              const std::vector<int32_t>& durations, int32_t total, const char* label) {
    constexpr int64_t channels = 6;
    const int64_t frames = static_cast<int64_t>(durations.size());
    std::vector<float> x(static_cast<size_t>(channels * frames));
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.37f) * 2.0f;

    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* x_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels, frames);
    ggml_tensor* dur_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames);
    ggml_tensor* out = ggml_ops_length_regulate(ctx, x_t, dur_t, total, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(x_t, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(dur_t, durations.data(), 0, durations.size() * sizeof(int32_t));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> expected(static_cast<size_t>(channels * total), 0.0f);
    int64_t pos = 0;
    for (int64_t t = 0; t < frames && pos < total; ++t) {
        for (int32_t k = 0; k < durations[t] && pos < total; ++k) {
            std::memcpy(expected.data() + pos * channels, x.data() + t * channels,
                        static_cast<size_t>(channels) * sizeof(float));
            ++pos;
        }
    }
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    passed &= maximum == 0.0f;
    std::cout << name << " LengthRegulate " << label << " error=" << maximum
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
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
        passed &= run_case(backend, name, {2, 0, 3, 1, 4}, 10, "exact");
        passed &= run_case(backend, name, {2, 0, 3, 1, 4}, 7, "truncated");
        passed &= run_case(backend, name, {2, 0, 3, 1, 4}, 13, "zero-filled");
        passed &= run_case(backend, name, {1}, 1, "single");
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
