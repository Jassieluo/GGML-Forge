#include "ops/ops.h"

#include <algorithm>
#include <cmath>
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

static bool run_backend(ggml_backend_t backend, const std::string& name) {
    ggml_context* ctx = ggml_init({ 2 * 1024 * 1024, nullptr, true });
    ggml_tensor* norm_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    ggml_tensor* norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* gated_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 8);
    ggml_tensor* rms = ggml_ops_rms_norm(ctx, norm_input, norm_weight, 1e-5f);
    ggml_tensor* gated = ggml_ops_gated_activation(ctx, gated_input, ggml_ops_gate_activation::silu, 1);
    ggml_tensor* l2 = ggml_ops_l2_normalize(ctx, gated_input, 1, 1e-12f);
    if (!rms || !gated || !l2 || gated->ne[0] != 3 || gated->ne[1] != 4) return false;

    std::vector<float> norm_values(12), gamma = { 0.5f, 1.0f, 1.5f, 2.0f }, gated_values(24);
    for (size_t i = 0; i < norm_values.size(); ++i) norm_values[i] = std::sin(float(i + 1) * 0.31f);
    for (size_t i = 0; i < gated_values.size(); ++i) gated_values[i] = std::cos(float(i + 1) * 0.17f);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(norm_input, norm_values.data(), 0, norm_values.size() * sizeof(float));
    ggml_backend_tensor_set(norm_weight, gamma.data(), 0, gamma.size() * sizeof(float));
    ggml_backend_tensor_set(gated_input, gated_values.data(), 0, gated_values.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, rms);
    ggml_build_forward_expand(graph, gated);
    ggml_build_forward_expand(graph, l2);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> rms_actual(12), gated_actual(12), l2_actual(24);
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(rms, rms_actual.data(), 0, rms_actual.size() * sizeof(float));
        ggml_backend_tensor_get(gated, gated_actual.data(), 0, gated_actual.size() * sizeof(float));
        ggml_backend_tensor_get(l2, l2_actual.data(), 0, l2_actual.size() * sizeof(float));
    }
    float error = 0.0f;
    for (int row = 0; row < 3; ++row) {
        float square_sum = 0.0f;
        for (int i = 0; i < 4; ++i) square_sum += norm_values[row * 4 + i] * norm_values[row * 4 + i];
        const float scale = 1.0f / std::sqrt(square_sum / 4.0f + 1e-5f);
        for (int i = 0; i < 4; ++i) {
            const float expected = norm_values[row * 4 + i] * scale * gamma[i];
            error = std::max(error, std::abs(rms_actual[row * 4 + i] - expected));
        }
    }
    for (int axis_value = 0; axis_value < 4; ++axis_value) for (int inner = 0; inner < 3; ++inner) {
        const float gate = gated_values[inner + 3 * axis_value];
        const float linear = gated_values[inner + 3 * (axis_value + 4)];
        const float expected = gate / (1.0f + std::exp(-gate)) * linear;
        error = std::max(error, std::abs(gated_actual[inner + 3 * axis_value] - expected));
    }
    for (int inner = 0; inner < 3; ++inner) {
        float square_sum = 0.0f;
        for (int axis_value = 0; axis_value < 8; ++axis_value) {
            const float value = gated_values[inner + 3 * axis_value];
            square_sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(square_sum + 1e-12f);
        for (int axis_value = 0; axis_value < 8; ++axis_value) {
            const size_t index = inner + 3 * axis_value;
            error = std::max(error, std::abs(l2_actual[index] - gated_values[index] * scale));
        }
    }
    const bool passed = status == GGML_STATUS_SUCCESS && error < 2e-5f;
    std::cout << name << " foundation error=" << error << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

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
    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        ok = run_backend(backend, name) && ok;
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return ok ? 0 : 1;
}
