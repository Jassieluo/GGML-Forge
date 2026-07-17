#include "ops/ops.h"
#include "ops/cpu.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#ifdef _WIN32
    __declspec(dllimport) void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    __declspec(dllimport) void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    __declspec(dllimport) void ggml_ops_ext_sycl_init();
#endif
#else
    void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    void ggml_ops_ext_sycl_init();
#endif
#endif
}

namespace {

struct WorkerResult {
    std::vector<float> output;
    std::vector<float> attention_output;
    std::string error;
};

bool test_shared_backend_lane(ggml_backend_dev_t device) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return false;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};
    auto worker = [&]() {
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        for (int i = 0; i < 20; ++i) {
            ggml_ops_ext::ops_backend_lane_guard lane(backend);
            const int current = active.fetch_add(1, std::memory_order_acq_rel) + 1;
            int observed = max_active.load(std::memory_order_relaxed);
            while (current > observed && !max_active.compare_exchange_weak(observed, current)) {}
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            active.fetch_sub(1, std::memory_order_acq_rel);
        }
    };
    std::thread first(worker);
    std::thread second(worker);
    while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    first.join();
    second.join();
    ggml_backend_free(backend);
    if (max_active.load(std::memory_order_relaxed) != 1) {
        std::cerr << ggml_backend_dev_name(device) << " shared backend lane overlapped." << std::endl;
        return false;
    }
    return true;
}

void run_conv_worker(
    ggml_backend_dev_t device,
    const std::vector<float>& weights,
    const std::vector<float>& input,
    const std::vector<float>& bias,
    int cpu_threads,
    std::atomic<int>& ready,
    std::atomic<bool>& go,
    WorkerResult& result
) {
    ready.fetch_add(1, std::memory_order_release);
    while (!go.load(std::memory_order_acquire)) std::this_thread::yield();

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        result.error = "backend initialization failed";
        return;
    }
    const std::string device_name = ggml_backend_dev_name(device);
    if (cpu_threads > 0 && device_name.rfind("CPU", 0) == 0) {
        ggml_ops_ext_cpu_set_n_threads(backend, cpu_threads);
    }

    constexpr int64_t kW = 5;
    constexpr int64_t channels = 32;
    constexpr int64_t length = 64;
    constexpr int64_t out_length = length - kW + 1;

    ggml_init_params init_params = { 16 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(init_params);
    ggml_tensor* w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kW, channels, channels);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, length, channels, 1);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    ggml_tensor* dst = ggml_ops_conv_1d(ctx, w, x, 1, 0, 1, 1, backend, b);
    if (!dst) {
        result.error = "Conv1D graph construction failed";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        result.error = "backend tensor allocation failed";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return;
    }

    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(b, bias.data(), 0, bias.size() * sizeof(float));

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    for (int i = 0; i < 20; ++i) {
        if (ggml_ops_ext::ops_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            result.error = "graph execution failed";
            break;
        }
    }

    if (result.error.empty()) {
        ggml_backend_synchronize(backend);
        result.output.resize(out_length * channels);
        ggml_backend_tensor_get(dst, result.output.data(), 0, result.output.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (result.error.empty()) {
        constexpr int64_t head_dim = 32;
        constexpr int64_t sequence = 16;
        constexpr int64_t heads = 4;
        constexpr size_t qkv_count = head_dim * sequence * heads;
        constexpr size_t bias_count = sequence * sequence * heads;

        std::vector<float> q_data(qkv_count);
        std::vector<float> k_data(qkv_count);
        std::vector<float> v_data(qkv_count);
        std::vector<float> bias_data(bias_count);
        for (size_t i = 0; i < qkv_count; ++i) {
            q_data[i] = 0.1f * std::sin((float)i * 0.011f);
            k_data[i] = 0.1f * std::cos((float)i * 0.019f);
            v_data[i] = 0.2f * std::sin((float)i * 0.007f);
        }
        for (size_t i = 0; i < bias_count; ++i) bias_data[i] = 0.01f * std::cos((float)i * 0.023f);

        ggml_init_params attention_params = { 16 * 1024 * 1024, nullptr, true };
        ggml_context* attention_ctx = ggml_init(attention_params);
        ggml_tensor* q = ggml_new_tensor_3d(attention_ctx, GGML_TYPE_F32, head_dim, sequence, heads);
        ggml_tensor* k = ggml_new_tensor_3d(attention_ctx, GGML_TYPE_F32, head_dim, sequence, heads);
        ggml_tensor* v = ggml_new_tensor_3d(attention_ctx, GGML_TYPE_F32, head_dim, sequence, heads);
        ggml_tensor* attn_bias = ggml_new_tensor_3d(attention_ctx, GGML_TYPE_F32, sequence, sequence, heads);
        ggml_tensor* attention_dst = ggml_ops_attention(
            attention_ctx, q, k, v, attn_bias, nullptr,
            1.0f / std::sqrt((float)head_dim), -1, backend
        );
        ggml_backend_buffer_t attention_buffer = attention_dst
            ? ggml_backend_alloc_ctx_tensors(attention_ctx, backend)
            : nullptr;
        if (!attention_buffer) {
            result.error = "attention tensor allocation failed";
        } else {
            ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
            ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
            ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
            ggml_backend_tensor_set(attn_bias, bias_data.data(), 0, bias_data.size() * sizeof(float));

            ggml_cgraph* attention_graph = ggml_new_graph(attention_ctx);
            ggml_build_forward_expand(attention_graph, attention_dst);
            for (int i = 0; i < 20; ++i) {
                if (ggml_ops_ext::ops_backend_graph_compute(backend, attention_graph) != GGML_STATUS_SUCCESS) {
                    result.error = "attention graph execution failed";
                    break;
                }
            }
            if (result.error.empty()) {
                ggml_backend_synchronize(backend);
                result.attention_output.resize(qkv_count);
                ggml_backend_tensor_get(
                    attention_dst, result.attention_output.data(), 0,
                    result.attention_output.size() * sizeof(float)
                );
            }
            ggml_backend_buffer_free(attention_buffer);
        }
        ggml_free(attention_ctx);
    }

    ggml_backend_free(backend);
}

bool test_device(ggml_backend_dev_t device) {
    if (!test_shared_backend_lane(device)) return false;
    constexpr int64_t kW = 5;
    constexpr int64_t channels = 32;
    constexpr int64_t length = 64;
    constexpr int64_t out_length = length - kW + 1;

    std::vector<float> weights(kW * channels * channels);
    std::vector<float> input(length * channels);
    std::vector<float> bias(channels);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = 0.05f * std::sin((float)i * 0.013f);
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.1f * std::cos((float)i * 0.017f);
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.01f * std::sin((float)i);

    std::vector<float> reference(out_length * channels);
    for (int64_t oc = 0; oc < channels; ++oc) {
        for (int64_t ow = 0; ow < out_length; ++ow) {
            float sum = bias[oc];
            for (int64_t ic = 0; ic < channels; ++ic) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    sum += weights[(oc * channels + ic) * kW + kw] * input[ic * length + ow + kw];
                }
            }
            reference[oc * out_length + ow] = sum;
        }
    }

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    WorkerResult results[2];
    std::thread workers[2] = {
        std::thread(run_conv_worker, device, std::cref(weights), std::cref(input), std::cref(bias), 1,
                    std::ref(ready), std::ref(go), std::ref(results[0])),
        std::thread(run_conv_worker, device, std::cref(weights), std::cref(input), std::cref(bias), 2,
                    std::ref(ready), std::ref(go), std::ref(results[1]))
    };
    while (ready.load(std::memory_order_acquire) != 2) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();

    for (int worker = 0; worker < 2; ++worker) {
        if (!results[worker].error.empty()) {
            std::cerr << ggml_backend_dev_name(device) << " worker " << worker << ": "
                      << results[worker].error << std::endl;
            return false;
        }
        float max_diff = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(reference[i] - results[worker].output[i]));
        }
        if (max_diff > 1e-3f) {
            std::cerr << ggml_backend_dev_name(device) << " worker " << worker
                      << " max diff " << max_diff << std::endl;
            return false;
        }
    }

    if (results[0].attention_output.size() != results[1].attention_output.size() ||
        results[0].attention_output.empty()) {
        std::cerr << ggml_backend_dev_name(device) << " attention output is missing." << std::endl;
        return false;
    }
    float attention_max_diff = 0.0f;
    for (size_t i = 0; i < results[0].attention_output.size(); ++i) {
        if (!std::isfinite(results[0].attention_output[i]) || !std::isfinite(results[1].attention_output[i])) {
            std::cerr << ggml_backend_dev_name(device) << " attention produced a non-finite value." << std::endl;
            return false;
        }
        attention_max_diff = std::max(
            attention_max_diff,
            std::abs(results[0].attention_output[i] - results[1].attention_output[i])
        );
    }
    if (attention_max_diff > 1e-5f) {
        std::cerr << ggml_backend_dev_name(device) << " concurrent attention max diff "
                  << attention_max_diff << std::endl;
        return false;
    }

    std::cout << ggml_backend_dev_name(device) << " concurrent Conv1D and Attention sessions passed." << std::endl;
    return true;
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

    bool tested = false;
    bool passed = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const std::string name = device ? ggml_backend_dev_name(device) : "";
        if (name.rfind("CPU", 0) != 0 && name.rfind("CUDA", 0) != 0 && name.rfind("SYCL", 0) != 0) continue;
        tested = true;
        passed = test_device(device) && passed;
    }

    ggml_ops_ext::release_ops_hook();
    if (!tested) {
        std::cerr << "No CPU, CUDA, or SYCL device available." << std::endl;
        return 1;
    }
    return passed ? 0 : 1;
}
