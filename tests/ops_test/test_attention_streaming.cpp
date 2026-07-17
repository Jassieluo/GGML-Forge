#include "ops/ops.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
extern "C" __declspec(dllimport) void ggml_ops_ext_cpu_init();
#else
extern "C" void ggml_ops_ext_cpu_init();
#endif
#ifdef GGML_USE_CUDA
extern "C" void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
extern "C" void ggml_ops_ext_sycl_init();
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> reference_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& bias,
    int head_dim,
    int q_len,
    int kv_len,
    int q_heads,
    int kv_heads,
    float scale) {
    std::vector<float> output(static_cast<size_t>(head_dim * q_len * q_heads));
    std::vector<float> scores(static_cast<size_t>(kv_len));
    const int group_size = q_heads / kv_heads;
    for (int head = 0; head < q_heads; ++head) {
        const int kv_head = head / group_size;
        for (int query = 0; query < q_len; ++query) {
            float maximum = -INFINITY;
            for (int key = 0; key < kv_len; ++key) {
                float score = bias[key + kv_len * (query + q_len * head)];
                for (int dim = 0; dim < head_dim; ++dim) {
                    score += scale * q[dim + head_dim * (query + q_len * head)] *
                                     k[dim + head_dim * (key + kv_len * kv_head)];
                }
                scores[key] = score;
                maximum = std::max(maximum, score);
            }
            float sum = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int dim = 0; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (int key = 0; key < kv_len; ++key) {
                    value += scores[key] / sum * v[dim + head_dim * (key + kv_len * kv_head)];
                }
                output[dim + head_dim * (query + q_len * head)] = value;
            }
        }
    }
    return output;
}

std::vector<uint8_t> quantize_rows(ggml_type type, const std::vector<float>& values, int rows, int row_size) {
    std::vector<uint8_t> result(ggml_row_size(type, row_size) * rows);
    const size_t written = ggml_quantize_chunk(
        type, values.data(), result.data(), 0, rows, row_size, nullptr);
    require(written == result.size(), "failed to quantize attention cache");
    return result;
}

std::vector<float> dequantize_rows(ggml_type type, const std::vector<uint8_t>& values, int rows, int row_size) {
    std::vector<float> result(static_cast<size_t>(rows * row_size));
    const ggml_type_traits* traits = ggml_get_type_traits(type);
    const size_t row_bytes = ggml_row_size(type, row_size);
    for (int row = 0; row < rows; ++row) {
        traits->to_float(values.data() + row * row_bytes, result.data() + row * row_size, row_size);
    }
    return result;
}

std::vector<float> cache_reference(ggml_type type, const std::vector<float>& values, int rows, int row_size) {
    if (type == GGML_TYPE_F32) return values;
    if (type == GGML_TYPE_F16) {
        std::vector<float> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]));
        }
        return result;
    }
    return dequantize_rows(type, quantize_rows(type, values, rows, row_size), rows, row_size);
}

void run_backend(
    ggml_backend_dev_t device,
    ggml_type key_type,
    ggml_type value_type,
    int q_len = 3,
    int kv_len = 7,
    int q_heads = 4,
    int kv_heads = 2) {
    const char* device_name = ggml_backend_dev_name(device);
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    require(backend != nullptr, "failed to initialize backend");

    constexpr int head_dim = 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> q(head_dim * q_len * q_heads);
    std::vector<float> k(head_dim * kv_len * kv_heads);
    std::vector<float> v(head_dim * kv_len * kv_heads);
    std::vector<float> bias(kv_len * q_len * q_heads, 0.0f);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.013f);
    for (size_t i = 0; i < k.size(); ++i) k[i] = std::cos(static_cast<float>(i) * 0.017f);
    for (size_t i = 0; i < v.size(); ++i) v[i] = std::sin(static_cast<float>(i) * 0.019f + 0.3f);
    for (int head = 0; head < q_heads; ++head) {
        for (int query = 0; query < q_len; ++query) {
            bias[(kv_len - 1) + kv_len * (query + q_len * head)] = -INFINITY;
        }
    }
    const int rows = kv_len * kv_heads;
    const std::vector<float> reference_k = cache_reference(key_type, k, rows, head_dim);
    const std::vector<float> reference_v = cache_reference(value_type, v, rows, head_dim);
    const std::vector<float> expected = reference_attention(
        q, reference_k, reference_v, bias, head_dim, q_len, kv_len, q_heads, kv_heads, scale);

    ggml_context* context = ggml_init({8 * 1024 * 1024, nullptr, true});
    require(context != nullptr, "failed to create graph context");
    ggml_tensor* q_tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, q_len, q_heads, 1);
    ggml_tensor* k_input = ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, kv_len, kv_heads, 1);
    ggml_tensor* v_input = ggml_new_tensor_4d(context, GGML_TYPE_F32, head_dim, kv_len, kv_heads, 1);
    ggml_tensor* k_tensor = ggml_new_tensor_4d(context, key_type, head_dim, kv_len, kv_heads, 1);
    ggml_tensor* v_tensor = ggml_new_tensor_4d(context, value_type, head_dim, kv_len, kv_heads, 1);
    ggml_tensor* bias_tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, kv_len, q_len, q_heads, 1);
    ggml_tensor* output = ggml_ops_attention(
        context, q_tensor, k_tensor, v_tensor, bias_tensor, nullptr, scale, -1, backend);
    require(output != nullptr, "failed to build streaming attention node");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, "failed to allocate graph tensors");
    ggml_backend_tensor_set(q_tensor, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(k_input, k.data(), 0, k.size() * sizeof(float));
    ggml_backend_tensor_set(v_input, v.data(), 0, v.size() * sizeof(float));
    ggml_backend_tensor_set(bias_tensor, bias.data(), 0, bias.size() * sizeof(float));

    ggml_cgraph* graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, ggml_cpy(context, k_input, k_tensor));
    ggml_build_forward_expand(graph, ggml_cpy(context, v_input, v_tensor));
    ggml_build_forward_expand(graph, output);
    require(ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "streaming attention execution failed");

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    float maximum_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum_error = std::max(maximum_error, std::abs(actual[i] - expected[i]));
    }
    require(maximum_error < 2e-4f,
            std::string(device_name ? device_name : "unknown") + " K=" + ggml_type_name(key_type) +
            " V=" + ggml_type_name(value_type) +
            " streaming attention error: " +
            std::to_string(maximum_error));

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    std::cout << (device_name ? device_name : "unknown") << " K=" << ggml_type_name(key_type)
              << " V=" << ggml_type_name(value_type)
              << " streaming attention passed\n";
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
    try {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t device = ggml_backend_dev_get(i);
            const std::string name = ggml_backend_dev_name(device) ? ggml_backend_dev_name(device) : "";
            if (name.find("BLAS") != std::string::npos) continue;
            const std::pair<ggml_type, ggml_type> cache_types[] = {
                {GGML_TYPE_F32, GGML_TYPE_F32},
                {GGML_TYPE_F16, GGML_TYPE_F16},
                {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
                {GGML_TYPE_Q4_0, GGML_TYPE_Q4_0},
                {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0},
            };
            for (const auto& [key_type, value_type] : cache_types) {
                run_backend(device, key_type, value_type);
            }
            // Exercises the tiled quantized prefill path rather than decode-only dispatch.
            run_backend(device, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, 17, 65);
            // Exercises the allocation-free strided MHA path used by DiT/ViT blocks.
            run_backend(device, GGML_TYPE_F32, GGML_TYPE_F32, 17, 65, 4, 4);
        }
    } catch (...) {
        ggml_ops_ext::release_ops_hook();
        throw;
    }
    ggml_ops_ext::release_ops_hook();
    return 0;
}
