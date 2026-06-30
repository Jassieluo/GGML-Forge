#include "ops/ops.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

void fill_sequential(float* data, size_t count, float start = 1.0f) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = start + i;
    }
}

extern "C" {
    __declspec(dllimport) void ggml_ops_ext_cpu_init();
}

int main() {
    // Force loading of CPU backend DLL
    ggml_ops_ext_cpu_init();

    // Load dynamic backends
    ggml_backend_load_all();

    // Detect backends
    ggml_backend_dev_t dev_cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* name = ggml_backend_dev_name(dev);
        std::string name_str = name ? name : "";
        for (auto& c : name_str) c = std::tolower(c);
        if (name_str.find("cpu") != std::string::npos) {
            dev_cpu = dev;
            break;
        }
    }
    if (!dev_cpu) {
        std::cerr << "CPU backend not found!" << std::endl;
        return 1;
    }
    ggml_backend_t cpu_backend = ggml_backend_dev_init(dev_cpu, nullptr);

    int64_t C_out = 2;
    int stride = 1;
    int dilation = 1;
    int padding = 0;
    int64_t batch = 2;
    int64_t kW = 3;
    int64_t C_in = 2;
    int64_t L_in = 4;

    int64_t L_out = (L_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
    size_t w_count = kW * C_in * C_out;
    size_t x_count = L_in * C_in * batch;
    size_t dst_count = L_out * C_out * batch;

    std::vector<float> w_host(w_count);
    std::vector<float> x_host(x_count);
    fill_sequential(w_host.data(), w_count, 1.0f);
    fill_sequential(x_host.data(), x_count, 1.0f);

    std::cout << "Weights (w_host):" << std::endl;
    for (size_t i = 0; i < w_count; ++i) std::cout << w_host[i] << " ";
    std::cout << std::endl;

    std::cout << "Input (x_host):" << std::endl;
    for (size_t i = 0; i < x_count; ++i) std::cout << x_host[i] << " ";
    std::cout << std::endl;

    // 1. Reference (GGML CPU Fallback)
    struct ggml_init_params ref_params = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_ref = ggml_init(ref_params);
    struct ggml_tensor* w_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F16, kW, C_in, C_out);
    struct ggml_tensor* x_ref = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, L_in, C_in, batch);
    struct ggml_tensor* dst_ref = ggml_ops_conv_1d(ctx_ref, w_ref, x_ref, stride, padding, dilation, nullptr);

    ggml_backend_buffer_t ref_buffer = ggml_backend_alloc_ctx_tensors(ctx_ref, cpu_backend);
    
    // Set w_ref data
    std::vector<ggml_fp16_t> w_f16(w_count);
    for (size_t i = 0; i < w_count; ++i) w_f16[i] = ggml_fp32_to_fp16(w_host[i]);
    ggml_backend_tensor_set(w_ref, w_f16.data(), 0, w_count * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(x_ref, x_host.data(), 0, x_count * sizeof(float));

    struct ggml_cgraph* graph_ref = ggml_new_graph(ctx_ref);
    ggml_build_forward_expand(graph_ref, dst_ref);
    ggml_backend_graph_compute(cpu_backend, graph_ref);

    std::vector<float> output_ref(dst_count);
    ggml_backend_tensor_get(dst_ref, output_ref.data(), 0, dst_count * sizeof(float));

    std::cout << "Reference Output (shape: " << dst_ref->ne[0] << ", " << dst_ref->ne[1] << ", " << dst_ref->ne[2] << "):" << std::endl;
    std::cout << "dst_ref strides: " << dst_ref->nb[0] << ", " << dst_ref->nb[1] << ", " << dst_ref->nb[2] << std::endl;
    for (size_t i = 0; i < dst_count; ++i) std::cout << output_ref[i] << " ";
    std::cout << std::endl;

    ggml_backend_buffer_free(ref_buffer);
    ggml_free(ctx_ref);

    // 2. Test (Custom OP on CPU)
    struct ggml_init_params test_params = { 16 * 1024 * 1024, nullptr, true };
    struct ggml_context* ctx_test = ggml_init(test_params);
    struct ggml_tensor* w_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, kW, C_in, C_out);
    struct ggml_tensor* x_test = ggml_new_tensor_3d(ctx_test, GGML_TYPE_F32, L_in, C_in, batch);

    ggml_ops_ext::install_ops_hook(cpu_backend);
    struct ggml_tensor* dst_test = ggml_ops_conv_1d(ctx_test, w_test, x_test, stride, padding, dilation, cpu_backend);

    ggml_backend_buffer_t test_buffer = ggml_backend_alloc_ctx_tensors(ctx_test, cpu_backend);
    ggml_backend_tensor_set(w_test, w_host.data(), 0, w_count * sizeof(float));
    ggml_backend_tensor_set(x_test, x_host.data(), 0, x_count * sizeof(float));

    struct ggml_cgraph* graph_test = ggml_new_graph(ctx_test);
    ggml_build_forward_expand(graph_test, dst_test);
    ggml_backend_graph_compute(cpu_backend, graph_test);

    std::vector<float> output_test(dst_count);
    ggml_backend_tensor_get(dst_test, output_test.data(), 0, dst_count * sizeof(float));

    std::cout << "Custom Op Output (shape: " << dst_test->ne[0] << ", " << dst_test->ne[1] << ", " << dst_test->ne[2] << "):" << std::endl;
    for (size_t i = 0; i < dst_count; ++i) std::cout << output_test[i] << " ";
    std::cout << std::endl;

    ggml_backend_buffer_free(test_buffer);
    ggml_free(ctx_test);
    ggml_ops_ext::uninstall_ops_hook(cpu_backend);

    ggml_backend_free(cpu_backend);
    return 0;
}
