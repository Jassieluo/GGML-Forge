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

// Helper to set tensor data
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

// 1. GLU Test
void run_glu_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type) {
    int64_t C = 192;
    int64_t T = 128;
    size_t x_count = 2 * C * T;
    size_t dst_count = C * T;

    std::vector<float> x_host(x_count);
    fill_random(x_host.data(), x_count);

    // Reference (Core Ops Fallback Subgraph on CPU backend)
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_2d(ctx_ref, type, 2 * C, T);
    struct ggml_tensor* dst_ref = ggml_ops_glu(ctx_ref, x_ref, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, x_host.data(), x_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    // Baseline (Target backend executing standard subgraph fallback)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_2d(ctx_base, type, 2 * C, T);
    // Passing nullptr backend forces building of standard GGML subgraph
    struct ggml_tensor* dst_base = ggml_ops_glu(ctx_base, x_base, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, x_host.data(), x_count);

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

    // Optimized (using registered backend handler/hook)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_2d(ctx_test, type, 2 * C, T);

    ggml_ops_ext::acquire_ops_hook();
    struct ggml_tensor* dst_test = ggml_ops_glu(ctx_test, x_test, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, x_host.data(), x_count);

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

    std::string type_str = type == GGML_TYPE_F16 ? "F16" : "F32";
    verify_results("GLU (" + backend_name + ") [" + type_str + "]", output_ref.data(), output_test.data(), dst_count, type == GGML_TYPE_F16 ? 1e-2f : 1e-4f);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::release_ops_hook();
}

// 2. Relative Position PE Keys Test
void run_relative_keys_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type, ggml_type emb_type) {
    int64_t d_k = 64;
    int64_t T = 64;
    int64_t n_head = 4;
    int32_t window_size = 4;
    float scale = 1.0f / std::sqrt((float)d_k);

    size_t q_count = d_k * T * n_head;
    size_t emb_count = d_k * (2 * window_size + 1) * n_head;
    size_t dst_count = T * T * n_head;

    std::vector<float> q_host(q_count);
    std::vector<float> emb_host(emb_count);
    fill_random(q_host.data(), q_count);
    fill_random(emb_host.data(), emb_count);

    // Reference (Core Ops Fallback coordinate shift subgraph on CPU backend)
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* q_ref = ggml_new_tensor_3d(ctx_ref, type, d_k, T, n_head);
    struct ggml_tensor* emb_ref = ggml_new_tensor_3d(ctx_ref, emb_type, d_k, 2 * window_size + 1, n_head);
    struct ggml_tensor* dst_ref = ggml_ops_relative_pe_keys(ctx_ref, q_ref, emb_ref, scale, window_size, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(q_ref, q_host.data(), q_count);
    set_tensor_data(emb_ref, emb_host.data(), emb_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    // Baseline (Target backend executing standard subgraph fallback)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* q_base = ggml_new_tensor_3d(ctx_base, type, d_k, T, n_head);
    struct ggml_tensor* emb_base = ggml_new_tensor_3d(ctx_base, emb_type, d_k, 2 * window_size + 1, n_head);
    struct ggml_tensor* dst_base = ggml_ops_relative_pe_keys(ctx_base, q_base, emb_base, scale, window_size, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(q_base, q_host.data(), q_count);
    set_tensor_data(emb_base, emb_host.data(), emb_count);

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

    // Optimized (using registered backend handler/hook)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* q_test = ggml_new_tensor_3d(ctx_test, type, d_k, T, n_head);
    struct ggml_tensor* emb_test = ggml_new_tensor_3d(ctx_test, emb_type, d_k, 2 * window_size + 1, n_head);

    ggml_ops_ext::acquire_ops_hook();
    struct ggml_tensor* dst_test = ggml_ops_relative_pe_keys(ctx_test, q_test, emb_test, scale, window_size, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(q_test, q_host.data(), q_count);
    set_tensor_data(emb_test, emb_host.data(), emb_count);

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

    std::string type_str = type == GGML_TYPE_F16 ? "F16" : "F32";
    std::string emb_str = emb_type == GGML_TYPE_F16 ? "F16" : "F32";
    verify_results("Relative PE Keys (" + backend_name + ") [" + type_str + "/emb:" + emb_str + "]", output_ref.data(), output_test.data(), dst_count, type == GGML_TYPE_F16 ? 1e-2f : 1e-4f);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::release_ops_hook();
}

// 3. Relative Position PE Values Test
void run_relative_values_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type, ggml_type emb_type) {
    int64_t d_k = 64;
    int64_t T = 64;
    int64_t n_head = 4;
    int32_t window_size = 4;

    size_t w_count = T * T * n_head;
    size_t emb_count = d_k * (2 * window_size + 1) * n_head;
    size_t dst_count = d_k * n_head * T;

    std::vector<float> w_host(w_count);
    std::vector<float> emb_host(emb_count);
    fill_random(w_host.data(), w_count, 0.01f, 1.0f); // weights should be non-negative
    fill_random(emb_host.data(), emb_count);

    // Reference (Core Ops Fallback coordinate shift subgraph on CPU backend)
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, type, T, T, n_head);
    struct ggml_tensor* emb_ref = ggml_new_tensor_3d(ctx_ref, emb_type, d_k, 2 * window_size + 1, n_head);
    struct ggml_tensor* dst_ref = ggml_ops_relative_pe_values(ctx_ref, w_ref, emb_ref, nullptr, window_size, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(w_ref, w_host.data(), w_count);
    set_tensor_data(emb_ref, emb_host.data(), emb_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    get_tensor_data(dst_ref, output_ref.data(), dst_count);

    // Baseline (Target backend executing standard subgraph fallback)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* w_base = ggml_new_tensor_3d(ctx_base, type, T, T, n_head);
    struct ggml_tensor* emb_base = ggml_new_tensor_3d(ctx_base, emb_type, d_k, 2 * window_size + 1, n_head);
    struct ggml_tensor* dst_base = ggml_ops_relative_pe_values(ctx_base, w_base, emb_base, nullptr, window_size, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(w_base, w_host.data(), w_count);
    set_tensor_data(emb_base, emb_host.data(), emb_count);

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

    // Optimized (using registered backend handler/hook)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, type, T, T, n_head);
    struct ggml_tensor* emb_test = ggml_new_tensor_3d(ctx_test, emb_type, d_k, 2 * window_size + 1, n_head);

    ggml_ops_ext::acquire_ops_hook();
    struct ggml_tensor* dst_test = ggml_ops_relative_pe_values(ctx_test, w_test, emb_test, nullptr, window_size, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(w_test, w_host.data(), w_count);
    set_tensor_data(emb_test, emb_host.data(), emb_count);

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

    std::string type_str = type == GGML_TYPE_F16 ? "F16" : "F32";
    std::string emb_str = emb_type == GGML_TYPE_F16 ? "F16" : "F32";
    verify_results("Relative PE Values (" + backend_name + ") [" + type_str + "/emb:" + emb_str + "]", output_ref.data(), output_test.data(), dst_count, type == GGML_TYPE_F16 ? 1e-2f : 1e-4f);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::release_ops_hook();
}

void run_instance_norm_test(ggml_backend_t backend, ggml_backend_t cpu_backend, const std::string& backend_name, ggml_type type, ggml_type param_type) {
    int64_t T = 250;
    int64_t C = 512;
    float eps = 1e-5f;

    size_t x_count = T * C;
    size_t w_count = C;

    std::vector<float> x_host(x_count);
    std::vector<float> gamma_host(w_count);
    std::vector<float> beta_host(w_count);

    fill_random(x_host.data(), x_count, -1.0f, 1.0f);
    fill_random(gamma_host.data(), w_count, 0.5f, 1.5f);
    fill_random(beta_host.data(), w_count, -0.5f, 0.5f);

    // Reference (CPU fallback path using standard GGML graph)
    struct ggml_init_params ref_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* x_ref = ggml_new_tensor_2d(ctx_ref, type, T, C);
    struct ggml_tensor* gamma_ref = ggml_new_tensor_1d(ctx_ref, param_type, C);
    struct ggml_tensor* beta_ref = ggml_new_tensor_1d(ctx_ref, param_type, C);
    struct ggml_tensor* dst_ref = ggml_ops_instance_norm(ctx_ref, x_ref, gamma_ref, beta_ref, eps, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    set_tensor_data(x_ref, x_host.data(), x_count);
    set_tensor_data(gamma_ref, gamma_host.data(), w_count);
    set_tensor_data(beta_ref, beta_host.data(), w_count);

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(x_count);
    get_tensor_data(dst_ref, output_ref.data(), x_count);

    // Baseline (Target backend executing standard subgraph fallback)
    struct ggml_init_params base_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_base = ggml_init(base_params);
    struct ggml_tensor* x_base = ggml_new_tensor_2d(ctx_base, type, T, C);
    struct ggml_tensor* gamma_base = ggml_new_tensor_1d(ctx_base, param_type, C);
    struct ggml_tensor* beta_base = ggml_new_tensor_1d(ctx_base, param_type, C);
    struct ggml_tensor* dst_base = ggml_ops_instance_norm(ctx_base, x_base, gamma_base, beta_base, eps, nullptr);

    ggml_backend_buffer_t base_buffer = ggml_backend_alloc_ctx_tensors(ctx_base, backend);
    set_tensor_data(x_base, x_host.data(), x_count);
    set_tensor_data(gamma_base, gamma_host.data(), w_count);
    set_tensor_data(beta_base, beta_host.data(), w_count);

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

    // Optimized (using registered backend handler/hook)
    struct ggml_init_params test_params = { 32 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* x_test = ggml_new_tensor_2d(ctx_test, type, T, C);
    struct ggml_tensor* gamma_test = ggml_new_tensor_1d(ctx_test, param_type, C);
    struct ggml_tensor* beta_test = ggml_new_tensor_1d(ctx_test, param_type, C);

    ggml_ops_ext::acquire_ops_hook();
    struct ggml_tensor* dst_test = ggml_ops_instance_norm(ctx_test, x_test, gamma_test, beta_test, eps, backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, backend);
    set_tensor_data(x_test, x_host.data(), x_count);
    set_tensor_data(gamma_test, gamma_host.data(), w_count);
    set_tensor_data(beta_test, beta_host.data(), w_count);

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

    std::string type_str = type == GGML_TYPE_F16 ? "F16" : "F32";
    std::string param_str = param_type == GGML_TYPE_F16 ? "F16" : "F32";
    verify_results("InstanceNorm (" + backend_name + ") [" + type_str + "/param:" + param_str + "]", output_ref.data(), output_test.data(), x_count, type == GGML_TYPE_F16 ? 1e-2f : 1e-4f);
    std::cout << "    Baseline Exec Time:  " << base_avg_time_us << " us\n"
              << "    Optimized Exec Time: " << opt_avg_time_us << " us\n"
              << "    Speedup:             " << (base_avg_time_us / std::max(opt_avg_time_us, 0.001)) << "x" << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);
    ggml_backend_buffer_free(base_buffer);
    ggml_free(ctx_base);
    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::release_ops_hook();
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

    std::cout << "=== GGML Custom NEW Operators Testing Suite ===" << std::endl;

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

        // Test GLU (F32 and F16)
        run_glu_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32);
        run_glu_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16);

        // Test Relative PE Keys (F32 and F16)
        run_relative_keys_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, GGML_TYPE_F32);
        run_relative_keys_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F16);
        run_relative_keys_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F32);

        // Test Relative PE Values (F32 and F16)
        run_relative_values_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, GGML_TYPE_F32);
        run_relative_values_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F16);
        run_relative_values_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F32);

        // Test InstanceNorm (F32 and F16)
        run_instance_norm_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F32, GGML_TYPE_F32);
        run_instance_norm_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F16);
        run_instance_norm_test(test_backend, cpu_ref_backend, name_str, GGML_TYPE_F16, GGML_TYPE_F32);

        ggml_backend_free(test_backend);
    }

    ggml_backend_free(cpu_ref_backend);
    std::cout << "\n========================================" << std::endl;
    std::cout << "New Operators testing completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
