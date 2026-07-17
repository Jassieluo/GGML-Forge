#include "gpt_sovits.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

namespace {

static std::vector<float> read_f32_file(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[VITS Align] Failed to open input: " << path << "\n";
        return {};
    }

    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0 || (size % sizeof(float)) != 0) {
        std::cerr << "[VITS Align] Invalid f32 file size: " << path << "\n";
        return {};
    }

    std::vector<float> data((size_t)size / sizeof(float));
    file.read(reinterpret_cast<char *>(data.data()), size);
    return data;
}

static void write_f32_file(const std::string & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[VITS Align] Failed to open output: " << path << "\n";
        return;
    }
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
}

} // namespace

int main(int argc, char ** argv) {
    std::string dict_dir = "models/gpt_sovits/dict";
    std::string hubert_path = "models/gpt_sovits/weights/cnhubert/cnhubert_f16.gguf";
    std::string bert_path = "models/gpt_sovits/weights/bert/bert_f16.gguf";
    std::string t2s_path = "models/gpt_sovits/weights/t2s/t2s_f16.gguf";
    std::string vits_path;
    std::string latent_path;
    std::string speaker_path;
    std::string mel_path;
    std::string tokens_path;   // int32 binary file with semantic token IDs
    std::string phones_path;   // int32 binary file with phone IDs
    std::string out_path = "scratch/vits_alignment_audio.f32";
    int threads = 4;
    std::string device = "cpu";

    bool is_cfm = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = argv[++i];
        } else if (arg == "--cfm" || arg == "--v3" || arg == "--v4") {
            is_cfm = true;
        } else if (arg == "--dict" && i + 1 < argc) {
            dict_dir = argv[++i];
        } else if (arg == "--hubert" && i + 1 < argc) {
            hubert_path = argv[++i];
        } else if (arg == "--bert" && i + 1 < argc) {
            bert_path = argv[++i];
        } else if (arg == "--t2s" && i + 1 < argc) {
            t2s_path = argv[++i];
        } else if (arg == "--vits" && i + 1 < argc) {
            vits_path = argv[++i];
        } else if (arg == "--latent" && i + 1 < argc) {
            latent_path = argv[++i];
        } else if (arg == "--speaker" && i + 1 < argc) {
            speaker_path = argv[++i];
        } else if (arg == "--mel" && i + 1 < argc) {
            mel_path = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            tokens_path = argv[++i];
        } else if (arg == "--phones" && i + 1 < argc) {
            phones_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        }
    }

    if (vits_path.empty() || (latent_path.empty() && mel_path.empty() && tokens_path.empty())) {
        std::cerr << "Usage: vits_alignment [--device cpu|CUDA0|SYCL0] [--dict <dir>] ... --vits <model.gguf> (--latent <latent.f32> | --mel <mel.f32> | --tokens <tokens.bin> --phones <phones.bin>) [--speaker <speaker.f32>] [--out <out.f32>]\n";
        return 1;
    }

    const int channels = is_cfm ? 100 : 192;

    // Latent loading (only for VITS generator test, skip for full pipeline or ref_enc)
    std::vector<float> latent_data;
    if (!latent_path.empty() && tokens_path.empty()) {
        latent_data = read_f32_file(latent_path);
        if (latent_data.empty() || (latent_data.size() % channels) != 0) {
            std::cerr << "[VITS Align] Latent file must contain a whole number of " << channels << "-channel frames.\n";
            return 1;
        }
        std::cout << "[VITS Align] Loaded latent frames=" << (latent_data.size() / channels) << std::endl;
    }

    std::vector<float> speaker_data;
    if (!speaker_path.empty()) {
        speaker_data = read_f32_file(speaker_path);
        std::cout << "[VITS Align] Loaded speaker embedding size=" << speaker_data.size() << std::endl;
    }

    std::cout << "[VITS Align] Initializing GPT-SoVITS engine..." << std::endl;
    gpt_sovits_engine_t engine = gpt_sovits_init_with_device(
        dict_dir.c_str(),
        hubert_path.c_str(),
        bert_path.c_str(),
        t2s_path.c_str(),
        vits_path.c_str(),
        threads,
        device == "cpu" ? 0 : 1,
        device.c_str()
    );
    if (!engine) {
        std::cerr << "[VITS Align] Failed to initialize engine.\n";
        return 1;
    }

    // Full pipeline test mode (enc_p + flow)
    if (!tokens_path.empty() && !phones_path.empty()) {
        std::vector<int> token_ids;
        {
            std::ifstream f(tokens_path, std::ios::binary);
            if (!f) { std::cerr << "Failed to open tokens file\n"; return 1; }
            f.seekg(0, std::ios::end);
            size_t n = f.tellg() / sizeof(int32_t);
            f.seekg(0, std::ios::beg);
            token_ids.resize(n);
            f.read(reinterpret_cast<char*>(token_ids.data()), n * sizeof(int32_t));
        }
        std::vector<int> phone_ids;
        {
            std::ifstream f(phones_path, std::ios::binary);
            if (!f) { std::cerr << "Failed to open phones file\n"; return 1; }
            f.seekg(0, std::ios::end);
            size_t n = f.tellg() / sizeof(int32_t);
            f.seekg(0, std::ios::beg);
            phone_ids.resize(n);
            f.read(reinterpret_cast<char*>(phone_ids.data()), n * sizeof(int32_t));
        }
        std::vector<float> speaker_data;
        if (!speaker_path.empty()) {
            speaker_data = read_f32_file(speaker_path);
        }

        std::cout << "[VITS Align] Full pipeline: tokens=" << token_ids.size()
                  << " phones=" << phone_ids.size() << " ge=" << speaker_data.size() << std::endl;

        int out_samples = 0;
        auto t_start = std::chrono::high_resolution_clock::now();
        const float* audio_ptr = gpt_sovits_debug_full_pipeline(
            engine, token_ids.data(), token_ids.size(),
            phone_ids.data(), phone_ids.size(),
            speaker_data.empty() ? nullptr : speaker_data.data(), speaker_data.size(),
            1.0f, &out_samples);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::cout << "[VITS Align] Pure VITS computation time: " << elapsed_ms << " ms" << std::endl;

        if (!audio_ptr || out_samples <= 0) {
            std::cerr << "[VITS Align] Full pipeline failed\n";
            gpt_sovits_free(engine);
            return 1;
        }
        std::vector<float> audio_data(audio_ptr, audio_ptr + out_samples);
        write_f32_file(out_path, audio_data);
        std::cout << "[VITS Align] Full pipeline: " << out_samples << " samples -> " << out_path << "\n";
        gpt_sovits_free(engine);
        return 0;
    }

    // Ref_enc test mode
    if (!mel_path.empty()) {
        std::vector<float> mel_data = read_f32_file(mel_path);
        if (mel_data.empty() || (mel_data.size() % 704) != 0) {
            std::cerr << "[VITS Align] Mel file must contain multiples of 704 floats.\n";
            gpt_sovits_free(engine);
            return 1;
        }
        std::cout << "[VITS Align] Running ref_enc with mel shape [704, " << (mel_data.size() / 704) << "]" << std::endl;
        int out_dim = 0;
        const float* ge_ptr = gpt_sovits_debug_ref_enc(engine, mel_data.data(), mel_data.size(), &out_dim);
        if (!ge_ptr || out_dim <= 0) {
            std::cerr << "[VITS Align] Ref enc failed.\n";
            gpt_sovits_free(engine);
            return 1;
        }
        std::vector<float> ge_data(ge_ptr, ge_ptr + out_dim);
        write_f32_file(out_path, ge_data);
        std::cout << "[VITS Align] Ref enc output " << out_dim << " dims -> " << out_path << "\n";
        gpt_sovits_free(engine);
        return 0;
    }

    std::cout << "[VITS Align] Running engine-backed VITS forward..." << std::endl;
    int out_samples = 0;
    const float* audio_ptr = gpt_sovits_debug_vits_from_latent(
        engine,
        latent_data.data(),
        latent_data.size(),
        speaker_data.empty() ? nullptr : speaker_data.data(),
        speaker_data.empty() ? 0 : speaker_data.size(),
        &out_samples
    );
    if (!audio_ptr || out_samples <= 0) {
        std::cerr << "[VITS Align] VITS debug forward failed.\n";
        gpt_sovits_free(engine);
        return 1;
    }

    std::vector<float> audio_data(audio_ptr, audio_ptr + out_samples);
    write_f32_file(out_path, audio_data);
    std::cout << "[VITS Align] Generated " << out_samples << " samples -> " << out_path << "\n";

    gpt_sovits_free(engine);
    return 0;
}
