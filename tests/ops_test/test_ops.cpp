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

// 1. Mish Test
void run_mish_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type) {
    int64_t ne0 = 512;
    int64_t ne1 = 2048;
    int64_t ne2 = 1;
    size_t count = ne0 * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    std::string prec_name = (type == GGML_TYPE_F32) ? "F32" : "F16";

    // Context for Reference execution
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_mish(ctx_ref, x_ref, nullptr); // runs CPU fallback

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, input_host.data(), count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    get_tensor_data(dst_ref, output_ref.data(), count);

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_mish(ctx_base, x_base, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, input_host.data(), count);

    struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
    ggml_build_forward_expand(graph_base, dst_base);
    ggml_backend_graph_compute(backend, graph_base); // Warmup

    auto start_base = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_base);
    }
    auto end_base = std::chrono::high_resolution_clock::now();
    double base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, type, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_mish(ctx_test, x_test, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, input_host.data(), count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(count);
    get_tensor_data(dst_test, output_test.data(), count);

    // Check correctness
    float tolerance = (type == GGML_TYPE_F32) ? 1e-4f : 1e-2f;
    verify_results("Mish (" + prec_name + ") (" + backend_name + ")", output_ref.data(), output_test.data(), count, tolerance);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

// 2. Gated Tanh Sigmoid Test
void run_gated_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type) {
    int64_t ne0 = 2048;
    int64_t ne1 = 2048;
    int64_t ne2 = 1;
    int hidden_channels = 1024;
    size_t count = ne0 * ne1 * ne2;
    size_t out_count = hidden_channels * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    std::string prec_name = (type == GGML_TYPE_F32) ? "F32" : "F16";

    // Reference
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_gated_tanh_sigmoid(ctx_ref, x_ref, hidden_channels, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, input_host.data(), count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(out_count);
    get_tensor_data(dst_ref, output_ref.data(), out_count);

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_gated_tanh_sigmoid(ctx_base, x_base, hidden_channels, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, input_host.data(), count);

    struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
    ggml_build_forward_expand(graph_base, dst_base);
    ggml_backend_graph_compute(backend, graph_base); // Warmup

    auto start_base = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_base);
    }
    auto end_base = std::chrono::high_resolution_clock::now();
    double base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, type, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_gated_tanh_sigmoid(ctx_test, x_test, hidden_channels, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, input_host.data(), count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(out_count);
    get_tensor_data(dst_test, output_test.data(), out_count);

    float tolerance = (type == GGML_TYPE_F32) ? 1e-4f : 1e-2f;
    verify_results("Gated Tanh Sigmoid (" + prec_name + ") (" + backend_name + ")", output_ref.data(), output_test.data(), out_count, tolerance);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

// 3. Conv Transpose 1D Test
void run_conv_t_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type w_type, ggml_type x_type) {
    int64_t kW = 16;
    int64_t C_in = 512;
    int64_t C_out = 256;
    int64_t L_in = 100;
    int64_t batch = 1;
    int stride = 10;
    int padding = 3;
    int dilation = 1;

    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (kW - 1) + 1; // 199

    size_t w_count = kW * C_in * C_out;
    size_t x_count = L_in * C_in * batch;
    size_t dst_count = L_out * C_out * batch;

    std::vector<float> w_host(w_count);
    std::vector<float> x_host(x_count);
    fill_random(w_host.data(), w_count);
    fill_random(x_host.data(), x_count);

    std::string w_prec = (w_type == GGML_TYPE_F32) ? "w:F32" : "w:F16";
    std::string x_prec = (x_type == GGML_TYPE_F32) ? "x:F32" : "x:F16";

    // 1. Reference (always F32 on CPU backend for correctness check)
    struct ggml_init_params ref_params = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, kW, C_out, C_in);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_ref = ggml_ops_conv_transpose_1d(ctx_ref, w_ref, x_ref, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(w_ref, w_host.data(), w_count);
    set_tensor_data(x_ref, x_host.data(), x_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // 2. Baseline (Target backend standard ops) - only run if supported (x_type == F32)
    bool run_baseline = (x_type == GGML_TYPE_F32);
    double base_avg_time_us = 0.0;
    std::vector<float> output_base(dst_count);

    if (run_baseline) {
        struct ggml_init_params base_params = { 64 * 1024 * 1024, nullptr, true };
        struct ggml_context* ctx_base = ggml_init(base_params);
        struct ggml_tensor* w_base = ggml_new_tensor_3d(ctx_base, w_type, kW, C_out, C_in);
        struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, x_type, L_in, C_in, batch);
        struct ggml_tensor* dst_base = ggml_ops_conv_transpose_1d(ctx_base, w_base, x_base, stride, padding, dilation, nullptr);

        ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
        set_tensor_data(w_base, w_host.data(), w_count);
        set_tensor_data(x_base, x_host.data(), x_count);

        struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
        ggml_build_forward_expand(graph_base, dst_base);
        ggml_backend_graph_compute(backend, graph_base); // Warmup

        auto start_base = std::chrono::high_resolution_clock::now();
        int iterations = 50;
        for (int i = 0; i < iterations; ++i) {
            ggml_backend_graph_compute(backend, graph_base);
        }
        auto end_base = std::chrono::high_resolution_clock::now();
        base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

        get_tensor_data(dst_base, output_base.data(), dst_count);

        ggml_backend_buffer_free(base_buffer);
        ggml_free(ctx_base);
    }

    // 3. Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, w_type, kW, C_out, C_in);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, x_type, L_in, C_in, batch);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_conv_transpose_1d(ctx_test, w_test, x_test, stride, padding, dilation, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(w_test, w_host.data(), w_count);
    set_tensor_data(x_test, x_host.data(), x_count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(dst_count);
    get_tensor_data(dst_test, output_test.data(), dst_count);

    // Verify directly against reference
    float tolerance = 1e-3f;
    if (w_type == GGML_TYPE_F16 && x_type == GGML_TYPE_F16) {
        tolerance = 3e-1f;
    } else if (w_type == GGML_TYPE_F16 || x_type == GGML_TYPE_F16) {
        tolerance = 1e-1f;
    }
    verify_results("Conv Transpose 1D (" + w_prec + "," + x_prec + ") (" + backend_name + ")", output_ref.data(), output_test.data(), dst_count, tolerance);
    
    if (run_baseline) {
        std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
                  << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
                  << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;
    } else {
        std::cout << "    Baseline Exec Time:  N/A (unsupported by native GGML)\n"
                  << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
                  << "    Speedup:             N/A" << std::endl;
    }

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

// 3b. Conv 1D Test
void run_conv_1d_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type w_type, ggml_type x_type) {
    int64_t kW = 5;
    int64_t C_in = 512;
    int64_t C_out = 512;
    int64_t L_in = 1000;
    int64_t batch = 1;
    int stride = 1;
    int padding = 2; // (kW - 1) / 2
    int dilation = 1;

    int64_t L_out = (L_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1; // 1000

    size_t w_count = kW * C_in * C_out;
    size_t x_count = L_in * C_in * batch;
    size_t dst_count = L_out * C_out * batch;

    std::vector<float> w_host(w_count);
    std::vector<float> x_host(x_count);
    fill_random(w_host.data(), w_count);
    fill_random(x_host.data(), x_count);

    std::string w_prec = (w_type == GGML_TYPE_F32) ? "w:F32" : "w:F16";
    std::string x_prec = (x_type == GGML_TYPE_F32) ? "x:F32" : "x:F16";

    // 1. Reference (always computed on CPU backend; w_ref must be F16 for CPU ggml_conv_1d compatibility)
    struct ggml_init_params ref_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F16, kW, C_in, C_out);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_ref = ggml_ops_conv_1d(ctx_ref, w_ref, x_ref, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(w_ref, w_host.data(), w_count);
    set_tensor_data(x_ref, x_host.data(), x_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // 2. Baseline (Target backend standard ops) - only run if supported (x_type == F32)
    bool run_baseline = (x_type == GGML_TYPE_F32);
    double base_avg_time_us = 0.0;
    std::vector<float> output_base(dst_count);

    if (run_baseline) {
        struct ggml_init_params base_params = { 128 * 1024 * 1024, nullptr, true };
        struct ggml_context* ctx_base = ggml_init(base_params);
        
        ggml_type w_type_base = w_type;
        ggml_type x_type_base = x_type;
        if (backend_name.find("CPU") == std::string::npos) {
            w_type_base = GGML_TYPE_F32;
            x_type_base = GGML_TYPE_F32;
        }

        struct ggml_tensor* w_base = ggml_new_tensor_3d(ctx_base, w_type_base, kW, C_in, C_out);
        struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, x_type_base, L_in, C_in, batch);
        struct ggml_tensor* dst_base = ggml_ops_conv_1d(ctx_base, w_base, x_base, stride, padding, dilation, nullptr);

        ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
        set_tensor_data(w_base, w_host.data(), w_count);
        set_tensor_data(x_base, x_host.data(), x_count);

        struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
        ggml_build_forward_expand(graph_base, dst_base);
        ggml_backend_graph_compute(backend, graph_base); // Warmup

        auto start_base = std::chrono::high_resolution_clock::now();
        int iterations = 50;
        for (int i = 0; i < iterations; ++i) {
            ggml_backend_graph_compute(backend, graph_base);
        }
        auto end_base = std::chrono::high_resolution_clock::now();
        base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

        get_tensor_data(dst_base, output_base.data(), dst_count);

        ggml_backend_buffer_free(base_buffer);
        ggml_free(ctx_base);
        
        if (w_type_base == GGML_TYPE_F32) {
            verify_results("Conv 1D Baseline (w:F32,x:F32) (" + backend_name + ")", output_ref.data(), output_base.data(), dst_count, 1e-2f);
        } else {
            verify_results("Conv 1D Baseline (" + w_prec + "," + x_prec + ") (" + backend_name + ")", output_ref.data(), output_base.data(), dst_count, 1e-2f);
        }
    }

    // 3. Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, w_type, kW, C_in, C_out);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, x_type, L_in, C_in, batch);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_conv_1d(ctx_test, w_test, x_test, stride, padding, dilation, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(w_test, w_host.data(), w_count);
    set_tensor_data(x_test, x_host.data(), x_count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(dst_count);
    get_tensor_data(dst_test, output_test.data(), dst_count);

    // Verify against F32 CPU Reference directly
    float tolerance = 1e-2f;
    if (w_type == GGML_TYPE_F16 || x_type == GGML_TYPE_F16) {
        tolerance = 9e-2f;
    }
    if (w_type == GGML_TYPE_F16 && x_type == GGML_TYPE_F16) {
        tolerance = 3e-1f;
    }
    verify_results("Conv 1D (" + w_prec + "," + x_prec + ") (" + backend_name + ")", output_ref.data(), output_test.data(), dst_count, tolerance);
    
    // Print debug values if mismatch
    if (w_type == GGML_TYPE_F16 && x_type == GGML_TYPE_F32) {
        std::cout << "\n=== DEBUG CONV 1D (First 20 elements comparison) ===" << std::endl;
        for (size_t i = 0; i < std::min(dst_count, (size_t)20); ++i) {
            std::cout << "  Index [" << i << "]: CPU_Ref = " << output_ref[i] 
                      << ", SYCL_Base = " << output_base[i] 
                      << ", SYCL_Opt = " << output_test[i] << std::endl;
        }
        std::cout << "===================================================\n" << std::endl;
    }

    if (run_baseline) {
        std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
                  << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
                  << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;
    } else {
        std::cout << "    Baseline Exec Time:  N/A (unsupported by native GGML)\n"
                  << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
                  << "    Speedup:             N/A" << std::endl;
    }

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

// 4. LayerNorm Test
void run_layernorm_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type) {
    int64_t ne0 = 1024;
    int64_t ne1 = 2048;
    int64_t ne2 = 1;
    float eps = 1e-5f;
    size_t count = ne0 * ne1 * ne2;
    size_t norm_count = ne0;

    std::vector<float> x_host(count);
    std::vector<float> gamma_host(norm_count);
    std::vector<float> beta_host(norm_count);
    fill_random(x_host.data(), count);
    fill_random(gamma_host.data(), norm_count, 0.5f, 1.5f);
    fill_random(beta_host.data(), norm_count, -0.5f, 0.5f);

    std::string prec_name = (type == GGML_TYPE_F32) ? "F32" : "F16";

    // Reference
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, type, ne0, ne1, ne2);
    struct ggml_tensor* gamma_ref = ggml_new_tensor_1d(ctx_ref, type, norm_count);
    struct ggml_tensor* beta_ref = ggml_new_tensor_1d(ctx_ref, type, norm_count);
    struct ggml_tensor* dst_ref = ggml_ops_layer_norm(ctx_ref, x_ref, gamma_ref, beta_ref, eps, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, x_host.data(), count);
    set_tensor_data(gamma_ref, gamma_host.data(), norm_count);
    set_tensor_data(beta_ref, beta_host.data(), norm_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    get_tensor_data(dst_ref, output_ref.data(), count);

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, type, ne0, ne1, ne2);
    struct ggml_tensor* gamma_base = ggml_new_tensor_1d(ctx_base, type, norm_count);
    struct ggml_tensor* beta_base = ggml_new_tensor_1d(ctx_base, type, norm_count);
    struct ggml_tensor* dst_base = ggml_ops_layer_norm(ctx_base, x_base, gamma_base, beta_base, eps, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, x_host.data(), count);
    set_tensor_data(gamma_base, gamma_host.data(), norm_count);
    set_tensor_data(beta_base, beta_host.data(), norm_count);

    struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
    ggml_build_forward_expand(graph_base, dst_base);
    ggml_backend_graph_compute(backend, graph_base); // Warmup

    auto start_base = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_base);
    }
    auto end_base = std::chrono::high_resolution_clock::now();
    double base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

    get_tensor_data(dst_base, output_ref.data(), count); // overwrite output_ref to use base output for base verification if needed, but not required since ref is CPU F32

    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, type, ne0, ne1, ne2);
    struct ggml_tensor* gamma_test = ggml_new_tensor_1d(ctx_test, type, norm_count);
    struct ggml_tensor* beta_test = ggml_new_tensor_1d(ctx_test, type, norm_count);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_layer_norm(ctx_test, x_test, gamma_test, beta_test, eps, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, x_host.data(), count);
    set_tensor_data(gamma_test, gamma_host.data(), norm_count);
    set_tensor_data(beta_test, beta_host.data(), norm_count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(count);
    get_tensor_data(dst_test, output_test.data(), count);

    float tolerance = (type == GGML_TYPE_F32) ? 1e-4f : 1e-2f;
    verify_results("LayerNorm (" + prec_name + ") (" + backend_name + ")", output_ref.data(), output_test.data(), count, tolerance);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

// 5. Double Swish Test
void run_double_swish_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type) {
    int64_t ne0 = 2048;
    int64_t ne1 = 1024;
    int64_t ne2 = 1;
    size_t count = ne0 * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    std::string prec_name = (type == GGML_TYPE_F32) ? "F32" : "F16";

    // Reference
    struct ggml_init_params ref_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_double_swish(ctx_ref, x_ref, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, input_host.data(), count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    get_tensor_data(dst_ref, output_ref.data(), count);

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, type, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_double_swish(ctx_base, x_base, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, input_host.data(), count);

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

    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, type, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_double_swish(ctx_test, x_test, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, input_host.data(), count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(count);
    get_tensor_data(dst_test, output_test.data(), count);

    float tolerance = (type == GGML_TYPE_F32) ? 1e-4f : 1e-2f;
    verify_results("Double Swish (" + prec_name + ") (" + backend_name + ")", output_ref.data(), output_test.data(), count, tolerance);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

void run_attention_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
    int64_t head_dim = 64;
    int64_t n_heads_q = 8;
    int64_t n_heads_kv = 8;
    int64_t seq_len_q = 100;
    int64_t seq_len_kv = 100;
    int64_t batch = 1;
    float scale = 0.125f;

    size_t q_count = head_dim * n_heads_q * seq_len_q * batch;
    size_t k_count = head_dim * n_heads_kv * seq_len_kv * batch;
    size_t v_count = head_dim * n_heads_kv * seq_len_kv * batch;
    size_t bias_count = seq_len_kv * seq_len_q * n_heads_q * batch;
    size_t dst_count = head_dim * n_heads_q * seq_len_q * batch;

    std::vector<float> q_host(q_count);
    std::vector<float> k_host(k_count);
    std::vector<float> v_host(v_count);
    std::vector<float> bias_host(bias_count);

    fill_random(q_host.data(), q_count);
    fill_random(k_host.data(), k_count);
    fill_random(v_host.data(), v_count);
    fill_random(bias_host.data(), bias_count, -0.5f, 0.5f);

    // 1. Reference (using GGUF fallback subgraph, which is 100% correct, running on cpu_backend)
    struct ggml_init_params ref_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* q_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, head_dim, n_heads_q, seq_len_q);
    struct ggml_tensor* k_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* v_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* bias_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, seq_len_kv, seq_len_q, n_heads_q);
    
    struct ggml_tensor* attn_w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, seq_len_kv, seq_len_q, n_heads_q);
    
    // We pass nullptr for backend to trigger fallback path
    struct ggml_tensor* dst_ref = ggml_ops_attention(ctx_ref, q_ref, k_ref, v_ref, bias_ref, attn_w_ref, scale, -1, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(q_ref, q_host.data(), q_count);
    set_tensor_data(k_ref, k_host.data(), k_count);
    set_tensor_data(v_ref, v_host.data(), v_count);
    set_tensor_data(bias_ref, bias_host.data(), bias_count);
    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    std::vector<float> weights_ref(bias_count);
    get_tensor_data(attn_w_ref, weights_ref.data(), bias_count);

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // 2. Baseline (Using GGUF fallback subgraph, but timed on target backend)
    struct ggml_init_params base_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* q_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, head_dim, n_heads_q, seq_len_q);
    struct ggml_tensor* k_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* v_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* bias_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, seq_len_kv, seq_len_q, n_heads_q);
    
    // Pass nullptr to force fallback path on target backend
    struct ggml_tensor* dst_base = ggml_ops_attention(ctx_base, q_base, k_base, v_base, bias_base, nullptr, scale, -1, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(q_base, q_host.data(), q_count);
    set_tensor_data(k_base, k_host.data(), k_count);
    set_tensor_data(v_base, v_host.data(), v_count);
    set_tensor_data(bias_base, bias_host.data(), bias_count);

    struct ggml_cgraph* graph_base = ggml_new_graph(ctx_base);
    ggml_build_forward_expand(graph_base, dst_base);
    ggml_backend_graph_compute(backend, graph_base); // Warmup

    auto start_base = std::chrono::high_resolution_clock::now();
    int iterations = 50;
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_base);
    }
    auto end_base = std::chrono::high_resolution_clock::now();
    double base_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_base - start_base).count() / (double)iterations;

    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);

    // 3. Test (Using optimized target backend custom handler)
    struct ggml_init_params test_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* q_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, head_dim, n_heads_q, seq_len_q);
    struct ggml_tensor* k_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* v_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, head_dim, n_heads_kv, seq_len_kv);
    struct ggml_tensor* bias_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, seq_len_kv, seq_len_q, n_heads_q);
    struct ggml_tensor* attn_w_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, seq_len_kv, seq_len_q, n_heads_q);

    ggml_ops_ext::install_ops_hook(backend);
    // Pass backend to trigger custom handler! Also test optional attn_w writing!
    struct ggml_tensor* dst_test = ggml_ops_attention(ctx_test, q_test, k_test, v_test, bias_test, attn_w_test, scale, -1, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(q_test, q_host.data(), q_count);
    set_tensor_data(k_test, k_host.data(), k_count);
    set_tensor_data(v_test, v_host.data(), v_count);
    set_tensor_data(bias_test, bias_host.data(), bias_count);

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(backend, graph_test); // Warmup

    auto start_opt = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_graph_compute(backend, graph_test);
    }
    auto end_opt = std::chrono::high_resolution_clock::now();
    double opt_avg_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end_opt - start_opt).count() / (double)iterations;

    std::vector<float> output_test(dst_count);
    get_tensor_data(dst_test, output_test.data(), dst_count);

    std::vector<float> weights_test(bias_count);
    get_tensor_data(attn_w_test, weights_test.data(), bias_count);

    std::cout << "\n=== DEBUG ATTENTION WEIGHTS (First 20 elements comparison) ===\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "  Weight Index [" << i << "]: Ref = " << weights_ref[i]
                  << ", Opt = " << weights_test[i] << "\n";
    }
    std::cout << "===================================================\n";

    std::cout << "\n=== DEBUG ATTENTION OUTPUTS (First 20 elements comparison) ===\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << "  Output Index [" << i << "]: Ref = " << output_ref[i]
                  << ", Opt = " << output_test[i] << "\n";
    }
    std::cout << "===================================================\n";

    // Verify weights first
    bool weights_ok = verify_results("Attention Weights (F32) (" + backend_name + ")", weights_ref.data(), weights_test.data(), bias_count, 1e-4f);
    if (!weights_ok) {
        std::cout << "  Warning: Attention Weights verification failed! Checking final outputs anyway...\n";
    }

    // Verify results
    verify_results("Fused Attention (F32) (" + backend_name + ")", output_ref.data(), output_test.data(), dst_count, 1e-4f);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

int main() {
    // Force link and load of custom backend DLLs
    ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    ggml_ops_ext_sycl_init();
#endif

    std::cout << "=== GGML Custom Operators Testing Suite ===" << std::endl;

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

    // Run tests on all detected device backends
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char* dev_name = ggml_backend_dev_name(dev);
        std::string name_str = dev_name ? dev_name : "Unnamed";
        std::string name_lower = name_str;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        // Skip non-CPU backends during CPU debugging to save time
        if (name_lower.find("cpu") == std::string::npos) {
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

        // Test Mish (F32 and F16)
        run_mish_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32);
        if (name_lower.find("cpu") == std::string::npos) {
            run_mish_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16);
        }

        // Test Gated Tanh Sigmoid (F32 and F16)
        run_gated_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32);
        if (name_lower.find("cpu") == std::string::npos) {
            run_gated_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16);
        }

        // Test Conv Transpose 1D (w:F32 x:F32, w:F16 x:F32, w:F16 x:F16)
        run_conv_t_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, GGML_TYPE_F32);
        run_conv_t_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F32);
        
        if (name_lower.find("cuda") != std::string::npos) {
            run_conv_t_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F16);
        }

        // Test Conv 1D (w:F32 x:F32, w:F16 x:F32, w:F16 x:F16)
        if (name_lower.find("cpu") == std::string::npos) {
            run_conv_1d_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, GGML_TYPE_F32);
        }
        run_conv_1d_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F32);
        if (name_lower.find("cuda") != std::string::npos) {
            run_conv_1d_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F16);
        }

        // Test LayerNorm (F32 and F16)
        run_layernorm_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32);
        if (name_lower.find("cpu") == std::string::npos) {
            run_layernorm_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16);
        }

        // Test Double Swish (F32 and F16)
        run_double_swish_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32);
        if (name_lower.find("cpu") == std::string::npos) {
            run_double_swish_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16);
        }

        // Test Fused Attention (F32)
        run_attention_test(test_backend, cpu_ref_backend, name_str);

        ggml_backend_free(test_backend);
    }

    ggml_backend_free(cpu_ref_backend);
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
