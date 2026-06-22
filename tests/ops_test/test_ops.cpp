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

// 1. Mish Test
void run_mish_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
    int64_t ne0 = 512;
    int64_t ne1 = 2048;
    int64_t ne2 = 1;
    size_t count = ne0 * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    // Context for Reference execution
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_mish(ctx_ref, x_ref, nullptr); // runs CPU fallback

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(x_ref, input_host.data(), 0, count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_mish(ctx_base, x_base, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(x_base, input_host.data(), 0, count * sizeof(float));

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
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_mish(ctx_test, x_test, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(x_test, input_host.data(), 0, count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, count * sizeof(float));

    // Check correctness
    verify_results("Mish (" + backend_name + ")", output_ref.data(), output_test.data(), count);
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
void run_gated_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
    int64_t ne0 = 2048;
    int64_t ne1 = 2048;
    int64_t ne2 = 1;
    int hidden_channels = 1024;
    size_t count = ne0 * ne1 * ne2;
    size_t out_count = hidden_channels * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    // Reference
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_gated_tanh_sigmoid(ctx_ref, x_ref, hidden_channels, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(x_ref, input_host.data(), 0, count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(out_count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, out_count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_gated_tanh_sigmoid(ctx_base, x_base, hidden_channels, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(x_base, input_host.data(), 0, count * sizeof(float));

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
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_gated_tanh_sigmoid(ctx_test, x_test, hidden_channels, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(x_test, input_host.data(), 0, count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, out_count * sizeof(float));

    verify_results("Gated Tanh Sigmoid (" + backend_name + ")", output_ref.data(), output_test.data(), out_count);
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
void run_conv_t_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
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

    // Reference
    struct ggml_init_params ref_params = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, kW, C_out, C_in);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_ref = ggml_ops_conv_transpose_1d(ctx_ref, w_ref, x_ref, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(w_ref, w_host.data(), 0, w_count * sizeof(float));
    ggml_backend_tensor_set(x_ref, x_host.data(), 0, x_count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, dst_count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* w_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, kW, C_out, C_in);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_base = ggml_ops_conv_transpose_1d(ctx_base, w_base, x_base, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(w_base, w_host.data(), 0, w_count * sizeof(float));
    ggml_backend_tensor_set(x_base, x_host.data(), 0, x_count * sizeof(float));

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
    struct ggml_init_params test_params = { 64 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, kW, C_out, C_in);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, L_in, C_in, batch);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_conv_transpose_1d(ctx_test, w_test, x_test, stride, padding, dilation, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(w_test, w_host.data(), 0, w_count * sizeof(float));
    ggml_backend_tensor_set(x_test, x_host.data(), 0, x_count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, dst_count * sizeof(float));

    verify_results("Conv Transpose 1D (" + backend_name + ")", output_ref.data(), output_test.data(), dst_count, 1e-3f);
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

// 3b. Conv 1D Test
void run_conv_1d_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
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

    std::vector<float> w_host_f32(w_count);
    std::vector<ggml_fp16_t> w_host(w_count);
    std::vector<float> x_host(x_count);
    fill_random(w_host_f32.data(), w_count);
    for (size_t i = 0; i < w_count; ++i) {
        w_host[i] = ggml_fp32_to_fp16(w_host_f32[i]);
    }
    fill_random(x_host.data(), x_count);

    // Reference
    struct ggml_init_params ref_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F16, kW, C_in, C_out);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_ref = ggml_ops_conv_1d(ctx_ref, w_ref, x_ref, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(w_ref, w_host.data(), 0, w_count * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(x_ref, x_host.data(), 0, x_count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, dst_count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* w_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F16, kW, C_in, C_out);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_base = ggml_ops_conv_1d(ctx_base, w_base, x_base, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(w_base, w_host.data(), 0, w_count * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(x_base, x_host.data(), 0, x_count * sizeof(float));

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

    std::vector<float> output_base(dst_count);
    ggml_backend_tensor_get(dst_base, output_base.data(), 0, dst_count * sizeof(float));
    verify_results("Conv 1D Baseline (" + backend_name + ")", output_ref.data(), output_base.data(), dst_count, 1e-3f);

    // Test (Optimized target backend custom op)
    struct ggml_init_params test_params = { 128 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F16, kW, C_in, C_out);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, L_in, C_in, batch);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_conv_1d(ctx_test, w_test, x_test, stride, padding, dilation, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(w_test, w_host.data(), 0, w_count * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(x_test, x_host.data(), 0, x_count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, dst_count * sizeof(float));

    if (backend_name.find("CPU") != std::string::npos) {
        verify_results("Conv 1D (" + backend_name + ")", output_ref.data(), output_test.data(), dst_count, 1e-3f);
    } else {
        verify_results("Conv 1D (" + backend_name + ")", output_base.data(), output_test.data(), dst_count, 1e-3f);
    }
    
    // Print debug values if mismatch
    if (backend_name.find("CUDA") != std::string::npos) {
        std::cout << "\n=== DEBUG CONV 1D (First 20 elements comparison) ===" << std::endl;
        for (size_t i = 0; i < std::min(dst_count, (size_t)20); ++i) {
            std::cout << "  Index [" << i << "]: CPU_Ref = " << output_ref[i] 
                      << ", CUDA_Base = " << output_base[i] 
                      << ", CUDA_Opt = " << output_test[i] << std::endl;
        }
        std::cout << "===================================================\n" << std::endl;
    }

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

// 4. LayerNorm Test
void run_layernorm_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
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

    // Reference
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* gamma_ref = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, norm_count);
    struct ggml_tensor* beta_ref = ggml_new_tensor_1d(ctx_ref, GGML_TYPE_F32, norm_count);
    struct ggml_tensor* dst_ref = ggml_ops_layer_norm(ctx_ref, x_ref, gamma_ref, beta_ref, eps, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(x_ref, x_host.data(), 0, count * sizeof(float));
    ggml_backend_tensor_set(gamma_ref, gamma_host.data(), 0, norm_count * sizeof(float));
    ggml_backend_tensor_set(beta_ref, beta_host.data(), 0, norm_count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* gamma_base = ggml_new_tensor_1d(ctx_base, GGML_TYPE_F32, norm_count);
    struct ggml_tensor* beta_base = ggml_new_tensor_1d(ctx_base, GGML_TYPE_F32, norm_count);
    struct ggml_tensor* dst_base = ggml_ops_layer_norm(ctx_base, x_base, gamma_base, beta_base, eps, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(x_base, x_host.data(), 0, count * sizeof(float));
    ggml_backend_tensor_set(gamma_base, gamma_host.data(), 0, norm_count * sizeof(float));
    ggml_backend_tensor_set(beta_base, beta_host.data(), 0, norm_count * sizeof(float));

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
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* gamma_test = ggml_new_tensor_1d(ctx_test, GGML_TYPE_F32, norm_count);
    struct ggml_tensor* beta_test = ggml_new_tensor_1d(ctx_test, GGML_TYPE_F32, norm_count);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_layer_norm(ctx_test, x_test, gamma_test, beta_test, eps, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(x_test, x_host.data(), 0, count * sizeof(float));
    ggml_backend_tensor_set(gamma_test, gamma_host.data(), 0, norm_count * sizeof(float));
    ggml_backend_tensor_set(beta_test, beta_host.data(), 0, norm_count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, count * sizeof(float));

    verify_results("LayerNorm (" + backend_name + ")", output_ref.data(), output_test.data(), count);
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

// 5. Double Swish Test
void run_double_swish_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name) {
    int64_t ne0 = 2048;
    int64_t ne1 = 1024;
    int64_t ne2 = 1;
    size_t count = ne0 * ne1 * ne2;

    std::vector<float> input_host(count);
    fill_random(input_host.data(), count);

    // Reference
    struct ggml_init_params ref_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_ref = ggml_ops_double_swish(ctx_ref, x_ref, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    ggml_backend_tensor_set(x_ref, input_host.data(), 0, count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, count * sizeof(float));

    // Baseline (Target backend standard ops)
    struct ggml_init_params base_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_3d(ctx_base, GGML_TYPE_F32, ne0, ne1, ne2);
    struct ggml_tensor* dst_base = ggml_ops_double_swish(ctx_base, x_base, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    ggml_backend_tensor_set(x_base, input_host.data(), 0, count * sizeof(float));

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
    struct ggml_init_params test_params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, ne0, ne1, ne2);

    ggml_ops_ext::install_ops_hook(backend);
    struct ggml_tensor* dst_test = ggml_ops_double_swish(ctx_test, x_test, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    ggml_backend_tensor_set(x_test, input_host.data(), 0, count * sizeof(float));

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
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, count * sizeof(float));

    verify_results("Double Swish (" + backend_name + ")", output_ref.data(), output_test.data(), count);
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

int main() {
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
        if (!dev) continue;
        const char* dev_name = ggml_backend_dev_name(dev);
        std::string name_str = dev_name ? dev_name : "Unnamed";
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Initializing Device: " << name_str << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        ggml_backend_t test_backend = ggml_backend_dev_init(dev, nullptr);
        if (!test_backend) {
            std::cerr << "Failed to initialize device: " << name_str << std::endl;
            continue;
        }

        run_mish_test(test_backend, cpu_ref_backend, name_str);
        run_gated_test(test_backend, cpu_ref_backend, name_str);
        run_conv_t_test(test_backend, cpu_ref_backend, name_str);
        run_conv_1d_test(test_backend, cpu_ref_backend, name_str);
        run_layernorm_test(test_backend, cpu_ref_backend, name_str);
        run_double_swish_test(test_backend, cpu_ref_backend, name_str);

        ggml_backend_free(test_backend);
    }

    ggml_backend_free(cpu_ref_backend);
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
