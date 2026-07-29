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

bool expected_compare(float a, float b, ggml_ops_ext::ops_compare_mode mode) {
    switch (mode) {
    case ggml_ops_ext::ops_compare_mode::equal: return a == b;
    case ggml_ops_ext::ops_compare_mode::not_equal: return a != b;
    case ggml_ops_ext::ops_compare_mode::less: return a < b;
    case ggml_ops_ext::ops_compare_mode::less_equal: return a <= b;
    case ggml_ops_ext::ops_compare_mode::greater: return a > b;
    case ggml_ops_ext::ops_compare_mode::greater_equal: return a >= b;
    }
    return false;
}

bool run_case(ggml_backend_t backend, const std::string& name,
              ggml_ops_ext::ops_compare_mode mode) {
    const std::vector<float> lhs = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> rhs = {-2, 0, 0, 0, 3, 2, 5, 5, 7, 6, 9, 9};
    ggml_context* context = ggml_init({1024 * 1024, nullptr, true});
    ggml_tensor* a = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 3);
    ggml_tensor* b = ggml_new_tensor_2d(context, GGML_TYPE_F32, 4, 3);
    ggml_tensor* mask = ggml_ops_compare(context, a, b, mode, backend);
    ggml_tensor* inverse = ggml_ops_logical(context, mask, nullptr,
        ggml_ops_ext::ops_logical_mode::logical_not, backend);
    ggml_tensor* chosen = ggml_ops_where(context, inverse, a, b, backend);
    if (!mask || !inverse || !chosen) { ggml_free(context); return false; }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    ggml_backend_tensor_set(a, lhs.data(), 0, lhs.size() * sizeof(float));
    ggml_backend_tensor_set(b, rhs.data(), 0, rhs.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, chosen);
    const ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);
    std::vector<float> actual(lhs.size());
    if (status == GGML_STATUS_SUCCESS) ggml_backend_tensor_get(chosen, actual.data(), 0, actual.size() * sizeof(float));
    bool passed = status == GGML_STATUS_SUCCESS;
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool comparison = expected_compare(lhs[i], rhs[i], mode);
        const float expected = comparison ? rhs[i] : lhs[i];
        passed &= std::abs(actual[i] - expected) < 1e-6f;
    }
    std::cout << name << " selection mode=" << static_cast<int>(mode)
              << (passed ? " PASSED\n" : " FAILED\n");
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
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
    bool passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = device ? ggml_backend_dev_name(device) : "";
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        for (int mode = 0; mode <= static_cast<int>(ggml_ops_ext::ops_compare_mode::greater_equal); ++mode) {
            passed &= run_case(backend, name, static_cast<ggml_ops_ext::ops_compare_mode>(mode));
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
