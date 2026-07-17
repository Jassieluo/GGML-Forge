#include "models/models.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ops/ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
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

struct wav_data {
    int sample_rate = 0;
    std::vector<float> samples;
};

static wav_data load_wav_file(const std::string & filename) {
    wav_data out;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[HuBERT Align] Failed to open WAV file: " << filename << "\n";
        return out;
    }

    char chunk_id[4];
    file.read(chunk_id, 4);
    if (std::strncmp(chunk_id, "RIFF", 4) != 0) {
        std::cerr << "[HuBERT Align] Invalid RIFF header\n";
        return out;
    }

    file.seekg(8, std::ios::beg);
    char format_id[4];
    file.read(format_id, 4);
    if (std::strncmp(format_id, "WAVE", 4) != 0) {
        std::cerr << "[HuBERT Align] Not a WAVE file\n";
        return out;
    }

    short bits_per_sample = 0;
    short num_channels = 0;

    while (file) {
        char subchunk_id[4];
        file.read(subchunk_id, 4);
        if (!file) {
            break;
        }

        int subchunk_size = 0;
        file.read(reinterpret_cast<char *>(&subchunk_size), 4);
        if (!file) {
            break;
        }

        if (std::strncmp(subchunk_id, "fmt ", 4) == 0) {
            short audio_format = 0;
            file.read(reinterpret_cast<char *>(&audio_format), 2);
            file.read(reinterpret_cast<char *>(&num_channels), 2);
            file.read(reinterpret_cast<char *>(&out.sample_rate), 4);
            file.seekg(6, std::ios::cur);
            file.read(reinterpret_cast<char *>(&bits_per_sample), 2);
            if (subchunk_size > 16) {
                file.seekg(subchunk_size - 16, std::ios::cur);
            }
        } else if (std::strncmp(subchunk_id, "data", 4) == 0) {
            if (num_channels != 1) {
                std::cerr << "[HuBERT Align] Expected mono WAV, got channels=" << num_channels << "\n";
                return {};
            }
            if (bits_per_sample == 16) {
                const int num_samples = subchunk_size / 2;
                std::vector<int16_t> raw_samples(num_samples);
                file.read(reinterpret_cast<char *>(raw_samples.data()), subchunk_size);
                out.samples.resize(num_samples);
                for (int i = 0; i < num_samples; ++i) {
                    out.samples[i] = raw_samples[i] / 32768.0f;
                }
            } else if (bits_per_sample == 32) {
                const int num_samples = subchunk_size / 4;
                out.samples.resize(num_samples);
                file.read(reinterpret_cast<char *>(out.samples.data()), subchunk_size);
            } else {
                std::cerr << "[HuBERT Align] Unsupported bits per sample: " << bits_per_sample << "\n";
                return {};
            }
            return out;
        } else {
            file.seekg(subchunk_size, std::ios::cur);
        }
    }

    std::cerr << "[HuBERT Align] Failed to find data chunk in WAV file\n";
    return {};
}

static void normalize_audio(std::vector<float> & audio) {
    if (audio.empty()) {
        return;
    }
    double sum = 0.0;
    for (float v : audio) {
        sum += v;
    }
    const double mean = sum / audio.size();

    double sq_sum = 0.0;
    for (float v : audio) {
        const double d = v - mean;
        sq_sum += d * d;
    }
    const double var = sq_sum / audio.size();
    const float scale = 1.0f / std::sqrt((float)var + 1e-7f);

    for (float & v : audio) {
        v = (v - (float)mean) * scale;
    }
}

static void write_f32_binary(const std::string & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[HuBERT Align] Failed to open binary output: " << path << "\n";
        return;
    }
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
}

static void write_summary(
    const std::string & path,
    const std::vector<float> & normalized_audio,
    const std::vector<float> & ssl_content,
    int64_t channels,
    int64_t frames) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[HuBERT Align] Failed to open summary output: " << path << "\n";
        return;
    }

    float audio_min = std::numeric_limits<float>::max();
    float audio_max = std::numeric_limits<float>::lowest();
    for (float v : normalized_audio) {
        audio_min = std::min(audio_min, v);
        audio_max = std::max(audio_max, v);
    }

    float ssl_min = std::numeric_limits<float>::max();
    float ssl_max = std::numeric_limits<float>::lowest();
    for (float v : ssl_content) {
        ssl_min = std::min(ssl_min, v);
        ssl_max = std::max(ssl_max, v);
    }

    file << "normalized_audio_len=" << normalized_audio.size() << "\n";
    file << "normalized_audio_min=" << audio_min << "\n";
    file << "normalized_audio_max=" << audio_max << "\n";
    file << "normalized_audio_first16=";
    for (size_t i = 0; i < std::min<size_t>(16, normalized_audio.size()); ++i) {
        if (i) file << ",";
        file << normalized_audio[i];
    }
    file << "\n";

    file << "ssl_content_shape=" << channels << "x" << frames << "\n";
    file << "ssl_content_min=" << ssl_min << "\n";
    file << "ssl_content_max=" << ssl_max << "\n";
    file << "ssl_content_ch0_first16=";
    for (int64_t i = 0; i < std::min<int64_t>(16, frames); ++i) {
        if (i) file << ",";
        file << ssl_content[i * channels];
    }
    file << "\n";
}

static ggml_backend_t pick_backend(bool use_gpu) {
    ggml_backend_load_all();
    if (!use_gpu) {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("CUDA", 0) == 0 || name.rfind("SYCL", 0) == 0) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                return backend;
            }
        }
    }
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

} // namespace

int main(int argc, char ** argv) {
    ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    ggml_ops_ext_sycl_init();
#endif
    ggml_ops_ext::acquire_ops_hook();
    struct OpsHookGuard {
        ~OpsHookGuard() { ggml_ops_ext::release_ops_hook(); }
    } ops_hook_guard;

    std::string hubert_path = "models/gpt_sovits/weights/cnhubert/cnhubert_f16.gguf";
    std::string wav_path = "models/gpt_sovits/reference_audios/firekeeper/gentle.wav";
    std::string out_prefix = "scratch/hubert_alignment_cpp";
    bool use_gpu = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hubert" && i + 1 < argc) {
            hubert_path = argv[++i];
        } else if (arg == "--wav" && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (arg == "--out-prefix" && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        }
    }

    ggml_backend_t backend = pick_backend(use_gpu);
    if (!backend) {
        std::cerr << "[HuBERT Align] Failed to initialize backend\n";
        return 1;
    }

    gpt_sovits::HubertModel hubert;
    if (!hubert.load(hubert_path, backend)) {
        std::cerr << "[HuBERT Align] Failed to load model: " << hubert_path << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    wav_data wav = load_wav_file(wav_path);
    if (wav.samples.empty()) {
        ggml_backend_free(backend);
        return 1;
    }
    if (wav.sample_rate != 16000) {
        std::cerr << "[HuBERT Align] Expected 16k reference WAV, got " << wav.sample_rate << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    std::vector<float> normalized_audio = wav.samples;
    normalized_audio.resize(normalized_audio.size() + 4800, 0.0f);
    normalize_audio(normalized_audio);

    ggml_init_params init_params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(init_params);
    ggml_tensor * ssl_content = hubert.forward(ctx, normalized_audio.data(), (int)normalized_audio.size(), backend);
    if (!ssl_content) {
        std::cerr << "[HuBERT Align] Hubert forward failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t channels = ssl_content->ne[0];
    const int64_t frames = ssl_content->ne[1];
    std::vector<float> ssl_content_host((size_t)(channels * frames));
    std::memcpy(ssl_content_host.data(), ssl_content->data, ssl_content_host.size() * sizeof(float));

    write_f32_binary(out_prefix + "_normalized_audio.f32", normalized_audio);
    write_f32_binary(out_prefix + "_ssl_content.f32", ssl_content_host);
    write_summary(out_prefix + "_summary.txt", normalized_audio, ssl_content_host, channels, frames);

    std::cout << "[HuBERT Align] Wrote:\n"
              << "  " << out_prefix << "_normalized_audio.f32\n"
              << "  " << out_prefix << "_ssl_content.f32\n"
              << "  " << out_prefix << "_summary.txt\n";

    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
