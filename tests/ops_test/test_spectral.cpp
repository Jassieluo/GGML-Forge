// Tests for GGML_OP_OPS_VIRT_FFT / STFT / ISTFT against naive host DFT
// references. Suites are selectable so edits to one op re-run only its tests:
//   test_spectral            all suites
//   test_spectral fft        complex forward/inverse FFT only
//   test_spectral stft       analysis only
//   test_spectral istft      synthesis + round-trip only
//   test_spectral cabs       complex magnitude / power spectrum only
#include "ops/ops.h"

#include <cmath>
#include <cstring>
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

namespace {

constexpr double kPi = 3.14159265358979323846;

bool check_error(const std::string& label, const std::vector<float>& actual,
                 const std::vector<float>& expected, float tolerance) {
    float maximum = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    const bool passed = maximum <= tolerance;
    std::cout << label << " error=" << maximum << (passed ? " PASSED\n" : " FAILED\n");
    return passed;
}

bool skip_unsupported(const std::string& backend_name, const char* op_name) {
    const bool is_cpu = backend_name.rfind("CPU", 0) == 0;
    std::cout << backend_name << " " << op_name
              << (is_cpu ? " FAILED (missing CPU kernel)\n" : " SKIPPED (no kernel)\n");
    return !is_cpu;
}

std::vector<float> test_signal(size_t count, float step) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = std::sin(float(i) * step) + 0.5f * std::cos(float(i) * step * 2.7f + 0.4f);
    }
    return values;
}

std::vector<float> hann_window(int n_fft) {
    std::vector<float> window(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[i] = 0.5f - 0.5f * static_cast<float>(std::cos(2.0 * kPi * i / n_fft));
    }
    return window;
}

// O(n^2) DFT: independent from the radix-2 kernel implementation on purpose.
void naive_dft(const float* re_in, const float* im_in, float* re_out, float* im_out,
               int n, bool inverse) {
    const double sign = inverse ? 1.0 : -1.0;
    for (int k = 0; k < n; ++k) {
        double sum_re = 0.0, sum_im = 0.0;
        for (int i = 0; i < n; ++i) {
            const double angle = sign * 2.0 * kPi * k * i / n;
            const double c = std::cos(angle), s = std::sin(angle);
            sum_re += re_in[i] * c - im_in[i] * s;
            sum_im += re_in[i] * s + im_in[i] * c;
        }
        const double scale = inverse ? 1.0 / n : 1.0;
        re_out[k] = static_cast<float>(sum_re * scale);
        im_out[k] = static_cast<float>(sum_im * scale);
    }
}

bool run_fft(ggml_backend_t backend, const std::string& name, bool inverse) {
    constexpr int n = 64;
    constexpr int64_t rows = 3;
    constexpr int64_t batch = 2;
    const auto re_data = test_signal(static_cast<size_t>(n * rows * batch), 0.11f);
    const auto im_data = test_signal(static_cast<size_t>(n * rows * batch), 0.23f);

    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n, rows, 2, batch);
    ggml_tensor* out = ggml_ops_fft(ctx, x, inverse, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name, "FFT"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            const size_t offset = static_cast<size_t>((b * rows + r) * n);
            ggml_backend_tensor_set(x, re_data.data() + offset,
                                    ((b * 2 + 0) * rows + r) * n * sizeof(float),
                                    n * sizeof(float));
            ggml_backend_tensor_set(x, im_data.data() + offset,
                                    ((b * 2 + 1) * rows + r) * n * sizeof(float),
                                    n * sizeof(float));
        }
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> actual(static_cast<size_t>(n * rows * 2 * batch));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    std::vector<float> expected(actual.size());
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            const size_t offset = static_cast<size_t>((b * rows + r) * n);
            naive_dft(re_data.data() + offset, im_data.data() + offset,
                      expected.data() + ((b * 2 + 0) * rows + r) * n,
                      expected.data() + ((b * 2 + 1) * rows + r) * n, n, inverse);
        }
    }
    passed &= check_error(name + " FFT" + (inverse ? " inverse" : " forward"),
                          actual, expected, 5e-4f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

bool run_stft(ggml_backend_t backend, const std::string& name) {
    constexpr int n_fft = 128;
    constexpr int hop = 32;
    constexpr int64_t batch = 2;
    constexpr int64_t samples = 512;
    constexpr int64_t frames = (samples - n_fft) / hop + 1;
    constexpr int64_t bins = n_fft / 2 + 1;
    const auto signal = test_signal(static_cast<size_t>(samples * batch), 0.083f);
    const auto window = hann_window(n_fft);

    ggml_context* ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_tensor* signal_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, samples, batch);
    ggml_tensor* window_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);
    ggml_tensor* out = ggml_ops_stft(ctx, signal_t, window_t, n_fft, hop, backend);
    if (!out) { ggml_free(ctx); return skip_unsupported(name, "STFT"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(signal_t, signal.data(), 0, signal.size() * sizeof(float));
    ggml_backend_tensor_set(window_t, window.data(), 0, window.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> actual(static_cast<size_t>(bins * frames * 2 * batch));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    std::vector<float> expected(actual.size());
    std::vector<float> frame_re(n_fft), frame_im(n_fft, 0.0f), dft_re(n_fft), dft_im(n_fft);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < frames; ++f) {
            for (int i = 0; i < n_fft; ++i) {
                frame_re[i] = signal[b * samples + f * hop + i] * window[i];
                frame_im[i] = 0.0f;
            }
            naive_dft(frame_re.data(), frame_im.data(), dft_re.data(), dft_im.data(),
                      n_fft, false);
            for (int64_t k = 0; k < bins; ++k) {
                expected[((b * 2 + 0) * frames + f) * bins + k] = dft_re[k];
                expected[((b * 2 + 1) * frames + f) * bins + k] = dft_im[k];
            }
        }
    }
    passed &= check_error(name + " STFT", actual, expected, 5e-4f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

// istft(stft(x)) == x wherever the window overlap envelope is positive: with
// NOLA normalization the round trip is exact, so compare away from the edges.
bool run_istft_roundtrip(ggml_backend_t backend, const std::string& name) {
    constexpr int n_fft = 128;
    constexpr int hop = 32;
    constexpr int64_t batch = 2;
    constexpr int64_t samples = 512;
    const auto signal = test_signal(static_cast<size_t>(samples * batch), 0.061f);
    const auto window = hann_window(n_fft);

    ggml_context* ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_tensor* signal_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, samples, batch);
    ggml_tensor* window_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_fft);
    ggml_tensor* spectrum = ggml_ops_stft(ctx, signal_t, window_t, n_fft, hop, backend);
    if (!spectrum) { ggml_free(ctx); return skip_unsupported(name, "iSTFT"); }
    ggml_tensor* rebuilt = ggml_ops_istft(ctx, spectrum, window_t, n_fft, hop, backend);
    if (!rebuilt) { ggml_free(ctx); return skip_unsupported(name, "iSTFT"); }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(signal_t, signal.data(), 0, signal.size() * sizeof(float));
    ggml_backend_tensor_set(window_t, window.data(), 0, window.size() * sizeof(float));
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, rebuilt);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> actual(signal.size());
    ggml_backend_tensor_get(rebuilt, actual.data(), 0, actual.size() * sizeof(float));
    std::vector<float> trimmed_actual, trimmed_expected;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = n_fft; t < samples - n_fft; ++t) {
            trimmed_actual.push_back(actual[b * samples + t]);
            trimmed_expected.push_back(signal[b * samples + t]);
        }
    }
    passed &= check_error(name + " iSTFT roundtrip", trimmed_actual, trimmed_expected, 5e-5f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

// complex_abs degrades to a native composition, so it must PASS on every
// backend — never SKIP.
bool run_complex_abs(ggml_backend_t backend, const std::string& name, bool squared) {
    constexpr int64_t n = 33;
    constexpr int64_t rows = 4;
    constexpr int64_t batch = 2;
    const auto re_data = test_signal(static_cast<size_t>(n * rows * batch), 0.19f);
    const auto im_data = test_signal(static_cast<size_t>(n * rows * batch), 0.31f);

    ggml_context* ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_tensor* x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n, rows, 2, batch);
    ggml_tensor* out = ggml_ops_complex_abs(ctx, x, squared, backend);
    if (!out) {
        std::cout << name << " ComplexAbs FAILED (builder returned null)\n";
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            const size_t offset = static_cast<size_t>((b * rows + r) * n);
            ggml_backend_tensor_set(x, re_data.data() + offset,
                                    ((b * 2 + 0) * rows + r) * n * sizeof(float),
                                    n * sizeof(float));
            ggml_backend_tensor_set(x, im_data.data() + offset,
                                    ((b * 2 + 1) * rows + r) * n * sizeof(float),
                                    n * sizeof(float));
        }
    }
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    bool passed = ggml_ops_ext::ops_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;

    std::vector<float> actual(static_cast<size_t>(n * rows * batch));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    std::vector<float> expected(actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const float power = re_data[i] * re_data[i] + im_data[i] * im_data[i];
        expected[i] = squared ? power : std::sqrt(power);
    }
    passed &= check_error(name + " ComplexAbs" + (squared ? " power" : " magnitude"),
                          actual, expected, 3e-6f);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

} // namespace

int main(int argc, char** argv) {
    const std::string suite = argc > 1 ? argv[1] : "all";
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
    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        const std::string name = ggml_backend_dev_name(device);
        if (name.rfind("CPU", 0) && name.rfind("CUDA", 0) && name.rfind("SYCL", 0)) continue;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        if (suite == "all" || suite == "fft") {
            passed &= run_fft(backend, name, false);
            passed &= run_fft(backend, name, true);
        }
        if (suite == "all" || suite == "stft") {
            passed &= run_stft(backend, name);
        }
        if (suite == "all" || suite == "istft") {
            passed &= run_istft_roundtrip(backend, name);
        }
        if (suite == "all" || suite == "cabs") {
            passed &= run_complex_abs(backend, name, false);
            passed &= run_complex_abs(backend, name, true);
        }
        ggml_backend_free(backend);
    }
    ggml_ops_ext::release_ops_hook();
    return passed ? 0 : 1;
}
