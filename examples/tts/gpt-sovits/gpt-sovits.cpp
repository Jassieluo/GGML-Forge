#include "categories/tts/tts.h"
#include "common/wav.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

static std::string utf16_to_utf8(const std::wstring& value) {
    if (value.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), size, nullptr, nullptr);
    return result;
}
#endif

namespace {

struct RuntimeDeleter { void operator()(tts_runtime_ptr value) const { tts_runtime_free(value); } };
struct ModelDeleter { void operator()(tts_model_ptr value) const { tts_free_model(value); } };
struct SessionDeleter { void operator()(tts_session_ptr value) const { tts_free_session(value); } };

using Runtime = std::unique_ptr<tts_runtime, RuntimeDeleter>;
using Model = std::unique_ptr<tts_model, ModelDeleter>;
using Session = std::unique_ptr<tts_session, SessionDeleter>;

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "  --model <json>       Portable TTS model composition\n"
        << "  --ref-audio <wav>    Reference voice audio (required)\n"
        << "  --ref-text <text>    Transcript of the reference audio\n"
        << "  --ref-lang <lang>    Reference language (default: zh)\n"
        << "  --text <text>        Text to synthesize\n"
        << "  --lang <lang>        Target language (default: zh)\n"
        << "  --device <name>      auto, cpu, or an exact backend device name\n"
        << "  --threads <count>    CPU workers per execution lane\n"
        << "  --cfm-steps <count>  Provider option for CFM-based models\n"
        << "  --speed <factor>     Speech speed (default: 1.0)\n"
        << "  --out <wav>          Output path (default: scratch/output.wav)\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    int wide_argc = 0;
    wchar_t** wide_argv = CommandLineToArgvW(GetCommandLineW(), &wide_argc);
    if (wide_argv) {
        for (int i = 0; i < wide_argc; ++i) args.push_back(utf16_to_utf8(wide_argv[i]));
        LocalFree(wide_argv);
    }
#else
    for (int i = 0; i < argc; ++i) args.emplace_back(argv[i]);
#endif

    std::string model_path = "models/tts/gpt_sovits/configs/v3-q4.json";
    std::string reference_path;
    std::string reference_text;
    std::string reference_language = "zh";
    std::string text = "你好，欢迎使用 GGML-Forge TTS 推理。";
    std::string language = "zh";
    std::string device = "auto";
    std::string output_path = "scratch/output.wav";
    uint32_t threads = 4;
    float speed = 1.0f;
    int cfm_steps = 0;

    for (size_t i = 1; i < args.size(); ++i) {
        const auto take = [&](std::string& output) -> bool {
            if (i + 1 >= args.size()) return false;
            output = args[++i];
            return true;
        };
        if (args[i] == "--model") { if (!take(model_path)) return 2; }
        else if (args[i] == "--ref-audio") { if (!take(reference_path)) return 2; }
        else if (args[i] == "--ref-text") { if (!take(reference_text)) return 2; }
        else if (args[i] == "--ref-lang") { if (!take(reference_language)) return 2; }
        else if (args[i] == "--text") { if (!take(text)) return 2; }
        else if (args[i] == "--lang") { if (!take(language)) return 2; }
        else if (args[i] == "--device") { if (!take(device)) return 2; }
        else if (args[i] == "--out") { if (!take(output_path)) return 2; }
        else if (args[i] == "--threads" && i + 1 < args.size()) threads = static_cast<uint32_t>(std::stoul(args[++i]));
        else if (args[i] == "--cfm-steps" && i + 1 < args.size()) cfm_steps = std::stoi(args[++i]);
        else if (args[i] == "--speed" && i + 1 < args.size()) speed = std::stof(args[++i]);
        else if (args[i] == "--cpu") device = "cpu";
        else if (args[i] == "--help" || args[i] == "-h") { print_usage(args[0].c_str()); return 0; }
        else { std::cerr << "Unknown or incomplete option: " << args[i] << '\n'; return 2; }
    }

    if (reference_path.empty()) {
        std::cerr << "--ref-audio is required; references belong to sessions, not provider engines.\n";
        return 2;
    }
    const example::Audio reference = example::load_wav(reference_path);
    if (reference.samples.empty()) {
        std::cerr << "Failed to load reference WAV: " << reference_path << '\n';
        return 1;
    }

    tts_runtime_params params = tts_runtime_default_params();
    params.device = device.c_str();
    params.n_threads = threads;
    Runtime runtime(tts_runtime_create(params));
    if (!runtime) return 1;

    Model model(tts_load_model(runtime.get(), model_path.c_str()));
    if (!model) return 1;
    std::cout << "Loaded " << tts_model_get_name(model.get())
              << " with provider " << tts_model_get_provider(model.get())
              << " on " << tts_runtime_get_device(runtime.get()) << '\n';

    Session session(tts_create_session(model.get()));
    if (!session || !tts_session_set_reference(
            session.get(), reference.samples.data(), reference.samples.size(), reference.sample_rate,
            reference_text.c_str(), reference_language.c_str())) {
        std::cerr << "Failed to create the TTS session or attach its reference.\n";
        return 1;
    }
    if (cfm_steps > 0) tts_session_set_float_option(session.get(), "cfm_steps", static_cast<float>(cfm_steps));

    int32_t sample_count = 0;
    const auto start = std::chrono::steady_clock::now();
    const float* audio = tts_synthesize(session.get(), text.c_str(), language.c_str(), speed, &sample_count);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const int32_t sample_rate = tts_session_get_output_sample_rate(session.get());
    if (!audio || sample_count <= 0 || sample_rate <= 0) {
        std::cerr << "Synthesis failed.\n";
        return 1;
    }

    const std::filesystem::path output = std::filesystem::u8path(output_path);
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
    if (!example::write_wav(output_path, audio, static_cast<size_t>(sample_count), sample_rate)) {
        std::cerr << "Failed to write output WAV: " << output_path << '\n';
        return 1;
    }
    const double audio_seconds = static_cast<double>(sample_count) / sample_rate;
    std::cout << "Wrote " << output_path << " (" << audio_seconds << " s, RTF "
              << seconds / audio_seconds << ")\n";
    return 0;
}
