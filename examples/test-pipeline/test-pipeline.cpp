#include "gpt_sovits.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <chrono>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

static std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}
#endif

namespace gpt_sovits {
    std::vector<float> voice_manager_load_wav_file(const std::string& filename, int& sample_rate);
}


// WAV Writer Helper
static void write_wav_file(const std::string& filename, const float* data, size_t num_samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Pipeline Test] Failed to open output WAV file: " << filename << std::endl;
        return;
    }

    char chunk_id[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size = 36 + num_samples * sizeof(int16_t);
    char format[4] = {'W', 'A', 'V', 'E'};
    char subchunk1_id[4] = {'f', 'm', 't', ' '};
    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate_val = sample_rate;
    uint32_t byte_rate = sample_rate * num_channels * sizeof(int16_t);
    uint16_t block_align = num_channels * sizeof(int16_t);
    uint16_t bits_per_sample = 16;
    char subchunk2_id[4] = {'d', 'a', 't', 'a'};
    uint32_t subchunk2_size = num_samples * sizeof(int16_t);

    file.write(chunk_id, 4);
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write(format, 4);
    file.write(subchunk1_id, 4);
    file.write(reinterpret_cast<const char*>(&subchunk1_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate_val), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    file.write(subchunk2_id, 4);
    file.write(reinterpret_cast<const char*>(&subchunk2_size), 4);

    for (size_t i = 0; i < num_samples; ++i) {
        float sample = data[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        int16_t pcm_sample = static_cast<int16_t>(sample * 32767.0f);
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(int16_t));
    }
    std::cout << "[Pipeline Test] Saved " << num_samples << " samples to " << filename << " (Sample Rate: " << sample_rate << "Hz)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "[Pipeline Test] Main function started..." << std::endl;
    std::vector<std::string> args;
#ifdef _WIN32
    int wargc;
    wchar_t** wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        for (int i = 0; i < wargc; ++i) {
            args.push_back(utf16_to_utf8(wargv[i]));
        }
        LocalFree(wargv);
    }
#else
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
#endif

    // Default paths
    std::string dict_dir = "models/gpt_sovits/dict";
    std::string hubert_path = "models/gpt_sovits/weights/cnhubert/cnhubert_q4_0.gguf";
    {
        std::ifstream f(hubert_path);
        if (!f.good()) {
            hubert_path = "models/gpt_sovits/weights/cnhubert/cnhubert_fp16.gguf";
        }
    }
    std::string bert_path = "models/gpt_sovits/weights/bert/bert_q4_0.gguf";
    {
        std::ifstream f(bert_path);
        if (!f.good()) {
            bert_path = "models/gpt_sovits/weights/bert/bert_fp16.gguf";
        }
    }
    std::string t2s_path = "models/gpt_sovits/weights/t2s/t2s_q4_0.gguf";
    {
        std::ifstream f(t2s_path);
        if (!f.good()) {
            t2s_path = "models/gpt_sovits/weights/t2s/t2s_fp16.gguf";
        }
    }
    std::string vits_path = "models/gpt_sovits/weights/vits/vits_fp16.gguf";
    std::string voices_root = "models/gpt_sovits/reference_audios";
    std::string character_id = "doubao";
    std::string emotion = "";
    std::string text = "你好，欢迎使用纯C加加推理的语音合成系统。项目整体架构设计干净优雅！";
    std::string lang = "zh";
    std::string out_wav = "scratch/output.wav";
    int threads = 4;
    bool use_gpu = true;
    std::string device_name = "";
    std::string ref_audio_path = "";
    std::string ref_text = "";
    std::string ref_lang = "";

    int args_size = (int)args.size();
    for (int i = 1; i < args_size; ++i) {
        const std::string arg = args[i];
        if (arg == "--dict" && i + 1 < args_size) {
            dict_dir = args[++i];
        } else if (arg == "--hubert" && i + 1 < args_size) {
            hubert_path = args[++i];
        } else if (arg == "--bert" && i + 1 < args_size) {
            bert_path = args[++i];
        } else if (arg == "--t2s" && i + 1 < args_size) {
            t2s_path = args[++i];
        } else if (arg == "--vits" && i + 1 < args_size) {
            vits_path = args[++i];
        } else if (arg == "--voices" && i + 1 < args_size) {
            voices_root = args[++i];
        } else if (arg == "--character" && i + 1 < args_size) {
            character_id = args[++i];
        } else if (arg == "--emotion" && i + 1 < args_size) {
            emotion = args[++i];
        } else if (arg == "--text" && i + 1 < args_size) {
            text = args[++i];
        } else if (arg == "--lang" && i + 1 < args_size) {
            lang = args[++i];
        } else if (arg == "--ref-audio" && i + 1 < args_size) {
            ref_audio_path = args[++i];
        } else if (arg == "--ref-text" && i + 1 < args_size) {
            ref_text = args[++i];
        } else if (arg == "--ref-lang" && i + 1 < args_size) {
            ref_lang = args[++i];
        } else if (arg == "--out" && i + 1 < args_size) {
            out_wav = args[++i];
        } else if (arg == "--threads" && i + 1 < args_size) {
            threads = std::stoi(args[++i]);
        } else if (arg == "--device" && i + 1 < args_size) {
            device_name = args[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: gpt-sovits-test-pipeline [options]\n"
                      << "Options:\n"
                      << "  --dict <path>        Path to text frontend dictionary directory (default: " << dict_dir << ")\n"
                      << "  --hubert <path>      Path to Hubert GGUF model (default: " << hubert_path << ")\n"
                      << "  --bert <path>        Path to BERT GGUF model (default: " << bert_path << ")\n"
                      << "  --t2s <path>         Path to T2S GGUF model (default: " << t2s_path << ")\n"
                      << "  --vits <path>        Path to VITS GGUF model (default: " << vits_path << ")\n"
                      << "  --voices <dir>       Path to voices root directory (default: " << voices_root << ")\n"
                      << "  --character <name>   Character ID to use (default: " << character_id << ")\n"
                      << "  --emotion <name>     Emotion to use (e.g., 平静, 兴奋, Comfort, etc.) (default: default)\n"
                      << "  --text <string>      Text to synthesize (default: " << text << ")\n"
                      << "  --lang <string>      Language of text (default: " << lang << ")\n"
                      << "  --ref-audio <path>   Path to reference audio file\n"
                      << "  --ref-text <string>  Text of the reference audio\n"
                      << "  --ref-lang <string>  Language of the reference audio\n"
                      << "  --out <path>         Output WAV file path (default: " << out_wav << ")\n"
                      << "  --threads <num>      Number of threads (default: " << threads << ")\n"
                      << "  --device <name>      Specific GPU device name to use (e.g. CUDA0, SYCL0)\n"
                      << "  --cpu                Force CPU-only mode (default: use GPU)\n";
            return 0;
        }
    }

    std::cout << "[Pipeline Test] Initializing GPT-SoVITS Engine...\n";
    std::cout << "  Dictionary Dir:   " << dict_dir << "\n"
              << "  Hubert Model:     " << hubert_path << "\n"
              << "  BERT Model:       " << bert_path << "\n"
              << "  T2S Model:        " << t2s_path << "\n"
              << "  VITS Model:       " << vits_path << "\n"
              << "  Threads:          " << threads << "\n"
              << "  GPU Enabled:      " << (use_gpu ? "Yes" : "No") << "\n";
    if (!device_name.empty()) {
        std::cout << "  Target Device:    " << device_name << "\n";
    }

    gpt_sovits_engine_t engine = gpt_sovits_init_with_device(
        dict_dir.c_str(),
        hubert_path.c_str(),
        bert_path.c_str(),
        t2s_path.c_str(),
        vits_path.c_str(),
        threads,
        use_gpu ? 1 : 0,
        device_name.c_str()
    );

    if (!engine) {
        std::cerr << "[Pipeline Test] Error: Failed to initialize GPT-SoVITS engine.\n";
        return 1;
    }
    std::cout << "[Pipeline Test] Engine initialized successfully.\n";

    const float* audio_data = nullptr;
    int out_num_samples = 0;
    gpt_sovits_voice_manager_t manager = nullptr;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (!ref_audio_path.empty()) {
        std::cout << "[Pipeline Test] Loading reference audio from: " << ref_audio_path << "\n";
        int ref_sr = 0;
        std::vector<float> ref_audio = gpt_sovits::voice_manager_load_wav_file(ref_audio_path, ref_sr);
        if (ref_audio.empty()) {
            std::cerr << "[Pipeline Test] Error: Failed to load reference audio.\n";
            gpt_sovits_free(engine);
            return 1;
        }
        std::cout << "[Pipeline Test] Reference audio loaded: " << ref_audio.size() << " samples, sample rate " << ref_sr << "Hz.\n";
        std::cout << "[Pipeline Test] Synthesizing speech with on-the-fly reference:\n"
                  << "  Text:     \"" << text << "\"\n"
                  << "  Language: \"" << lang << "\"\n"
                  << "  Ref Text: \"" << ref_text << "\"\n"
                  << "  Ref Lang: \"" << ref_lang << "\"\n";

        audio_data = gpt_sovits_synthesize(
            engine,
            text.c_str(),
            lang.c_str(),
            ref_audio.data(),
            ref_audio.size(),
            ref_text.c_str(),
            ref_lang.c_str(),
            1.0f, // speed
            &out_num_samples
        );
    } else {
        std::cout << "[Pipeline Test] Initializing Voice Manager...\n";
        manager = gpt_sovits_voice_manager_init(engine);
        if (!manager) {
            std::cerr << "[Pipeline Test] Error: Failed to initialize Voice Manager.\n";
            gpt_sovits_free(engine);
            return 1;
        }

        std::cout << "[Pipeline Test] Registering character \"" << character_id << "\" under \"" << voices_root << "\"...\n";
        std::string char_dir = voices_root + "/" + character_id;
        bool reg_ok = gpt_sovits_voice_manager_register_character(manager, char_dir.c_str(), character_id.c_str());
        if (!reg_ok) {
            std::cerr << "[Pipeline Test] Error: Failed to register character \"" << character_id << "\".\n";
            gpt_sovits_voice_manager_free(manager);
            gpt_sovits_free(engine);
            return 1;
        }
        std::cout << "[Pipeline Test] Character registered successfully.\n";

        std::string synth_char_id = character_id;
        if (!emotion.empty()) {
            synth_char_id = character_id + "/" + emotion;
        }

        std::cout << "[Pipeline Test] Synthesizing speech:\n"
                  << "  Text:     \"" << text << "\"\n"
                  << "  Language: \"" << lang << "\"\n"
                  << "  Target Character/Emotion: \"" << synth_char_id << "\"\n";

        audio_data = gpt_sovits_voice_manager_synthesize(
            manager,
            synth_char_id.c_str(),
            text.c_str(),
            lang.c_str(),
            1.0f, // speed
            &out_num_samples
        );
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    if (!audio_data || out_num_samples <= 0) {
        std::cerr << "[Pipeline Test] Error: Synthesis failed.\n";
        if (manager) gpt_sovits_voice_manager_free(manager);
        gpt_sovits_free(engine);
        return 1;
    }
    double audio_len_sec = (double)out_num_samples / 32000.0;
    double rtf = (duration_ms / 1000.0) / audio_len_sec;
    std::cout << "[Pipeline Test] Synthesis completed successfully. Generated " << out_num_samples << " samples.\n";
    std::cout << "[Pipeline Test] Time taken: " << duration_ms << " ms\n";
    std::cout << "[Pipeline Test] Audio length: " << audio_len_sec << " s\n";
    std::cout << "[Pipeline Test] Real-Time Factor (RTF): " << rtf << "\n";

    // Ensure output folder exists
    size_t last_slash = out_wav.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        std::string out_dir = out_wav.substr(0, last_slash);
#ifdef _WIN32
        CreateDirectoryA(out_dir.c_str(), NULL);
#else
        std::string mkdir_cmd = "mkdir -p " + out_dir;
        system(mkdir_cmd.c_str());
#endif
    }

    // VITS audio sample rate is 32000Hz (32kHz)
    write_wav_file(out_wav, audio_data, out_num_samples, 32000);

    // Free resources
    std::cout << "[Pipeline Test] Cleaning up resources...\n";
    if (manager) gpt_sovits_voice_manager_free(manager);
    gpt_sovits_free(engine);
    std::cout << "[Pipeline Test] Finished!\n";

    return 0;
}
