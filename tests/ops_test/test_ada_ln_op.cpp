#include "ggml.h"
#include "ggml-backend.h"
#include "ops/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <string>
#include <cstring>
#include <algorithm>

namespace gpt_sovits {
    thread_local ggml_backend_t current_vits_backend = nullptr;
}

#ifdef _WIN32
#  define GGML_OPS_EXT_API extern "C" __declspec(dllimport)
#else
#  define GGML_OPS_EXT_API extern "C"
#endif

GGML_OPS_EXT_API void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
GGML_OPS_EXT_API void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
GGML_OPS_EXT_API void ggml_ops_ext_sycl_init();
#endif

// Tolerance checker
bool verify_results(const std::string& op_name, const float* ref, const float* test, size_t count, float tolerance = 1e-4f) {
    float max_diff = 0.0f;
    double sum_sq_err = 0.0;
    size_t err_count = 0;

    float min_val = ref[0];
    float max_val = ref[0];
    double sum_abs_val = 0.0;

    for (size_t i = 0; i < count; ++i) {
        float r = ref[i];
        float t = test[i];
        float diff = std::abs(r - t);
        if (diff > max_diff) {
            max_diff = diff;
        }
        sum_sq_err += diff * diff;
        if (diff > tolerance) {
            err_count++;
        }
        min_val = std::min(min_val, r);
        max_val = std::max(max_val, r);
        sum_abs_val += std::abs(r);
    }
    double rmse = std::sqrt(sum_sq_err / count);
    double mean_abs_val = sum_abs_val / count;
    double rel_max_diff = (mean_abs_val > 1e-8) ? (max_diff / mean_abs_val) : 0.0;

    std::cout << "  [" << op_name << "]\n"
              << "    Val Range: [" << min_val << ", " << max_val << "], MeanAbs: " << mean_abs_val << "\n"
              << "    Max Diff: " << max_diff << ", Rel Max Diff: " << rel_max_diff << ", RMSE: " << rmse << "\n"
              << "    Errors (> " << tolerance << "): " << err_count << "/" << count;
    
    if (err_count == 0) {
        std::cout << " -> PASSED" << std::endl;
        return true;
    } else {
        std::cout << " -> FAILED" << std::endl;
        return false;
    }
}

// Random tensor filler
void fill_random(float* data, size_t count, float min_val = -2.0f, float max_val = 2.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

// Helpers to get/set tensors supporting F16
void set_tensor_data(struct ggml_tensor* tensor, const float* data, size_t count) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, data, 0, count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> temp(count);
        for (size_t i = 0; i < count; ++i) {
            temp[i] = ggml_fp32_to_fp16(data[i]);
        }
        ggml_backend_tensor_set(tensor, temp.data(), 0, count * sizeof(ggml_fp16_t));
    } else {
        std::cerr << "set_tensor_data: unsupported type " << tensor->type << std::endl;
    }
}

void get_tensor_data(struct ggml_tensor* tensor, float* data, size_t count) {
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, data, 0, count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> temp(count);
        ggml_backend_tensor_get(tensor, temp.data(), 0, count * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < count; ++i) {
            data[i] = ggml_fp16_to_fp32(temp[i]);
        }
    } else {
        std::cerr << "get_tensor_data: unsupported type " << tensor->type << std::endl;
    }
}

// AdaLN Test
void run_ada_ln_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type, float eps) {
    int64_t ne0 = 1024; // C
    int64_t ne1 = 512;  // T
    int64_t ne2 = 2;    // B
    
    size_t x_count = ne0 * ne1 * ne2;
    size_t cond_count = ne0 * ne2; // scale/shift shape [ne0, ne2]

    std::vector<float> x_host(x_count);
    std::vector<float> scale_host(cond_count);
    std::vector<float> shift_host(cond_count);

    fill_random(x_host.data(), x_count, -1.5f, 1.5f);
    fill_random(scale_host.data(), cond_count, -0.5f, 0.5f);
    fill_random(shift_host.data(), cond_count, -0.5f, 0.5f);

    std::string prec_name = (type == GGML_TYPE_F32) ? "F32" : "F16";
    std::cout << "\nRunning AdaLN Test (" << prec_name << ", eps=" << eps << ") on " << backend_name << "..." << std::endl;

    // Context for Reference execution (no backend -> fallback primitive nodes)
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, type, ne0, ne1, ne2);
    struct ggml_tensor* scale_ref = ggml_new_tensor_2d(ctx_ref, type, ne0, ne2);
    struct ggml_tensor* shift_ref = ggml_new_tensor_2d(ctx_ref, type, ne0, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_ada_ln(ctx_ref, x_ref, scale_ref, shift_ref, eps, nullptr); // Runs default fallback (composes primitives)

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, x_host.data(), x_count);
    set_tensor_data(scale_ref, scale_host.data(), cond_count);
    set_tensor_data(shift_ref, shift_host.data(), cond_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(x_count);
    get_tensor_data(dst_ref, output_ref.data(), x_count);

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, type, ne0, ne1, ne2);
    struct ggml_tensor* scale_base = ggml_new_tensor_2d(ctx_base, type, ne0, ne2);
    struct ggml_tensor* shift_base = ggml_new_tensor_2d(ctx_base, type, ne0, ne2);
    struct ggml_tensor* dst_base = ggml_ops_ada_ln(ctx_base, x_base, scale_base, shift_base, eps, nullptr); // Runs default fallback

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, x_host.data(), x_count);
    set_tensor_data(scale_base, scale_host.data(), cond_count);
    set_tensor_data(shift_base, shift_host.data(), cond_count);

    struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
    ggml_build_forward_expand(graph_base, dst_base);
    ggml_backend_graph_compute(backend, graph_base); // Warmup

    auto start_base = std::chrono::high_resolution_clock::now();
    int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_base);
    }
    auto end_base = std::chrono::high_resolution_clock::now();
    double base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, type, ne0, ne1, ne2);
    struct ggml_tensor* scale_test = ggml_new_tensor_2d(ctx_test, type, ne0, ne2);
    struct ggml_tensor* shift_test = ggml_new_tensor_2d(ctx_test, type, ne0, ne2);

    ggml_ops_ext::acquire_ops_hook();
    struct ggml_tensor* dst_test = ggml_ops_ada_ln(ctx_test, x_test, scale_test, shift_test, eps, backend); // Calls custom op handler

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, x_host.data(), x_count);
    set_tensor_data(scale_test, scale_host.data(), cond_count);
    set_tensor_data(shift_test, shift_host.data(), cond_count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(x_count);
    get_tensor_data(dst_test, output_test.data(), x_count);

    float tolerance = (type == GGML_TYPE_F32) ? 1e-4f : 2e-3f;
    std::string op_test_name = "AdaLN Custom Op VS Primitive Fallback";
    bool ok = verify_results(op_test_name, output_ref.data(), output_test.data(), x_count, tolerance);

    if (ok) {
        std::cout << "    Performance Comparison:" << std::endl;
        std::cout << "      Primitive Fallback: " << base_avg_time_us << " us" << std::endl;
        std::cout << "      Custom Native Op:   " << opt_avg_time_us << " us" << std::endl;
        std::cout << "      Speedup Factor:     " << (base_avg_time_us / opt_avg_time_us) << "x" << std::endl;
    }

    // Clean up
    ggml_ops_ext::release_ops_hook();
    ggml_backend_buffer_free(ref_buffer);
    ggml_backend_buffer_free(base_buffer);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_ref);
    ggml_free(ctx_base);
    ggml_free(ctx_test);
}

int main() {
    std::cout << "=== GGML Custom AdaLN Operator Testing Suite ===" << std::endl;

    // Load dynamic backends
    ggml_backend_load_all();

    // Find and initialize CPU backend for reference runs
    ggml_backend_t cpu_ref_backend = nullptr;
    size_t n_devs = ggml_backend_dev_count();
    
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* name = ggml_backend_dev_name(dev);
        std::string name_str = name ? name : "";
        std::transform(name_str.begin(), name_str.end(), name_str.begin(), ::tolower);
        if (name_str.find("cpu") != std::string::npos) {
            cpu_ref_backend = ggml_backend_dev_init(dev, nullptr);
            break;
        }
    }

    if (!cpu_ref_backend) {
        std::cerr << "Error: CPU backend is required for reference computations!" << std::endl;
        return 1;
    }

    ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    ggml_ops_ext_sycl_init();
#endif

    // Run tests on all detected device backends (typically CPU first)
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* dev_name = ggml_backend_dev_name(dev);
        std::string name_str = dev_name ? dev_name : "Unnamed";
        std::string name_lower = name_str;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        if (name_lower.find("blas") != std::string::npos) {
            continue;
        }

        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Initializing Device: " << name_str << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        ggml_backend_t test_backend = ggml_backend_dev_init(dev, nullptr);
        if (!test_backend) {
            std::cerr << "Failed to initialize device: " << name_str << std::endl;
            continue;
        }

        run_ada_ln_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, 1e-6f);
        run_ada_ln_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, 1e-6f);

        ggml_backend_free(test_backend);
    }

    ggml_backend_free(cpu_ref_backend);
    std::cout << "\n========================================" << std::endl;
    std::cout << "AdaLN Operator testing completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
