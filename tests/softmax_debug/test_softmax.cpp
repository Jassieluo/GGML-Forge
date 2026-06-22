#include "ggml.h"
#include "ggml-backend.h"
#include "ops/ops.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cctype>
#include <string>
#include <cstring>

namespace gpt_sovits {
    thread_local ggml_backend_t current_vits_backend = nullptr;
}

// Helper to calculate statistics of a host buffer
void print_stats(const std::string& label, const float* data, size_t count) {
    float min_val = INFINITY;
    float max_val = -INFINITY;
    double sum = 0.0;
    size_t nan_count = 0;
    size_t inf_count = 0;

    for (size_t i = 0; i < count; ++i) {
        float val = data[i];
        if (std::isnan(val)) {
            nan_count++;
        } else if (std::isinf(val)) {
            inf_count++;
        } else {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            sum += val;
        }
    }

    std::cout << "  " << label << ": count=" << count 
              << ", min=" << min_val << ", max=" << max_val 
              << ", mean=" << (count > (nan_count + inf_count) ? (sum / (count - nan_count - inf_count)) : 0.0)
              << ", NaNs=" << nan_count << ", Infs=" << inf_count << std::endl;
}

// Function to run a specific test case on a backend
void run_test_case(
    ggml_backend_t backend, 
    const std::string& backend_name,
    const std::string& case_name, 
    bool contiguous, 
    bool use_mask,
    float scale,
    float max_bias
) {
    std::cout << "\n========================================\n"
              << "Running [" << case_name << "] on " << backend_name << "\n"
              << "========================================" << std::endl;

    // Install custom operators hook
    ggml_ops_ext::install_ops_hook(backend);

    // 1. Create a GGML context
    struct ggml_init_params params = {
        /*mem_size   =*/ 16 * 1024 * 1024,
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        std::cerr << "Failed to init GGML context" << std::endl;
        return;
    }

    // Shapes: ne0 = columns, ne1 = rows, ne2 = heads, ne3 = batch
    int64_t ne0 = 8;
    int64_t ne1 = 8;
    int64_t ne2 = 2;
    int64_t ne3 = 1;

    struct ggml_tensor* a = nullptr;
    struct ggml_tensor* a_input = nullptr;

    if (contiguous) {
        // Contiguous input tensor: [8, 8, 2, 1]
        a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        a_input = a;
    } else {
        // Non-contiguous input tensor:
        // We create a larger parent tensor [16, 8, 2, 1] and slice a view [8, 8, 2, 1]
        // This gives a non-contiguous stride for columns since the parent stride is 16 elements.
        struct ggml_tensor* parent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, ne1, ne2, ne3);
        
        // Slice a view: width=8, height=8, offset=0
        a = ggml_view_4d(ctx, parent, ne0, ne1, ne2, ne3, parent->nb[1], parent->nb[2], parent->nb[3], 0);
        a_input = parent;
    }

    struct ggml_tensor* mask = nullptr;
    if (use_mask) {
        // Mask tensor: same shape, or broadcastable. We create [8, 8, 2, 1]
        mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    }

    // 2. Build graph first
    struct ggml_tensor* dst = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);

    // 3. Allocate buffers on the backend for all tensors in the context
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        std::cerr << "Failed to allocate backend buffer" << std::endl;
        ggml_free(ctx);
        return;
    }

    // 4. Populate input data on host and upload to backend
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    size_t a_input_elements = ggml_nelements(a_input);
    std::vector<float> a_host(a_input_elements);
    for (size_t i = 0; i < a_input_elements; ++i) {
        a_host[i] = dist(rng);
    }
    ggml_backend_tensor_set(a_input, a_host.data(), 0, a_input_elements * sizeof(float));

    if (use_mask) {
        size_t mask_elements = ggml_nelements(mask);
        std::vector<float> mask_host(mask_elements);
        // Let's set some mask values to 0.0f, and some to -10000.0f (large negative values)
        for (size_t i = 0; i < mask_elements; ++i) {
            mask_host[i] = (i % 2 == 0) ? 0.0f : -10000.0f;
        }
        ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_elements * sizeof(float));
    }

    struct ggml_cgraph* cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, dst);

    // 5. Compute graph
    ggml_status status = ggml_backend_graph_compute(backend, cgraph);
    if (status != GGML_STATUS_SUCCESS) {
        std::cerr << "Graph compute failed with status " << status << std::endl;
        ggml_free(ctx);
        return;
    }

    // 6. Read back results and print stats
    size_t dst_elements = ggml_nelements(dst);
    std::vector<float> dst_host(dst_elements);
    ggml_backend_tensor_get(dst, dst_host.data(), 0, dst_elements * sizeof(float));

    // Print first few outputs
    std::cout << "First 8 outputs of head 0, row 0:" << std::endl;
    std::cout << "  [";
    for (int i = 0; i < 8; ++i) {
        std::cout << dst_host[i] << (i < 7 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    print_stats("Output Stats", dst_host.data(), dst_elements);

    // Clean up
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_ops_ext::uninstall_ops_hook(backend);
}

int main() {
    std::cout << "Starting GGML SoftMax verification suite..." << std::endl;

    // Load all dynamic backends
    ggml_backend_load_all();

    std::cout << "Registered backends count: " << ggml_backend_reg_count() << std::endl;
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        std::cout << "  Backend " << i << ": " << ggml_backend_reg_name(reg) << std::endl;
    }

    // Load CPU and SYCL backends via modern device API
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_sycl = nullptr;

    size_t n_devs = ggml_backend_dev_count();
    std::cout << "Available devices count: " << n_devs << std::endl;
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char* name = ggml_backend_dev_name(dev);
        std::cout << "  Device " << i << ": " << (name ? name : "unnamed") << std::endl;
        
        std::string name_str = name ? name : "";
        std::transform(name_str.begin(), name_str.end(), name_str.begin(), ::tolower);
        
        if (name_str.find("cpu") != std::string::npos && !backend_cpu) {
            backend_cpu = ggml_backend_dev_init(dev, nullptr);
            std::cout << "    --> Initialized CPU backend!" << std::endl;
        } else if (name_str.find("sycl") != std::string::npos && !backend_sycl) {
            backend_sycl = ggml_backend_dev_init(dev, nullptr);
            std::cout << "    --> Initialized SYCL backend!" << std::endl;
        }
    }

    if (!backend_cpu) {
        std::cerr << "CPU backend not found or failed to initialize!" << std::endl;
        return 1;
    }
    if (!backend_sycl) {
        std::cerr << "SYCL backend not found or failed to initialize!" << std::endl;
        return 1;
    }


    // ----------------------------------------------------
    // TEST CASE 1: Contiguous, small values, no mask
    // ----------------------------------------------------
    run_test_case(backend_cpu, "CPU", "TestCase 1: Contiguous, no mask, small values", true, false, 1.0f, 0.0f);
    run_test_case(backend_sycl, "SYCL", "TestCase 1: Contiguous, no mask, small values", true, false, 1.0f, 0.0f);

    // ----------------------------------------------------
    // TEST CASE 2: Contiguous, large negative mask values (-10000.0f)
    // ----------------------------------------------------
    run_test_case(backend_cpu, "CPU", "TestCase 2: Contiguous, large negative mask, small values", true, true, 1.0f, 0.0f);
    run_test_case(backend_sycl, "SYCL", "TestCase 2: Contiguous, large negative mask, small values", true, true, 1.0f, 0.0f);

    // ----------------------------------------------------
    // TEST CASE 3: Non-contiguous, small values, no mask
    // ----------------------------------------------------
    run_test_case(backend_cpu, "CPU", "TestCase 3: Non-contiguous, no mask, small values", false, false, 1.0f, 0.0f);
    run_test_case(backend_sycl, "SYCL", "TestCase 3: Non-contiguous, no mask, small values", false, false, 1.0f, 0.0f);

    // ----------------------------------------------------
    // TEST CASE 4: Non-contiguous, large negative mask
    // ----------------------------------------------------
    run_test_case(backend_cpu, "CPU", "TestCase 4: Non-contiguous, large negative mask, small values", false, true, 1.0f, 0.0f);
    run_test_case(backend_sycl, "SYCL", "TestCase 4: Non-contiguous, large negative mask, small values", false, true, 1.0f, 0.0f);

    // Clean up backends
    ggml_backend_free(backend_cpu);
    ggml_backend_free(backend_sycl);

    std::cout << "\nVerification suite completed!" << std::endl;
    return 0;
}
