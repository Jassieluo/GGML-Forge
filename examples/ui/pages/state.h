#pragma once

#include "eui/signal.h"

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

namespace app {

enum class Tool { Chat, Speech, Image };

struct ChatMessage {
    std::string role;
    std::string text;
};

struct StudioState {
    Tool tool = Tool::Chat;
    int backend = 0;
    bool busy = false;
    bool has_error = false;
    std::atomic<float> progress{0.0f};
    std::string status = "Ready";

    std::vector<ChatMessage> messages = {
        {"assistant", "你好，我是运行在 GGML-Forge 上的本地助手。你可以对话、合成语音或生成图片。"},
    };
    std::string chat_input;
    std::string speech_text = "欢迎使用 GGML-Forge 本地语音合成演示。";
    float speech_speed = 1.0f;
    std::string audio_path;
    double audio_seconds = 0.0;

    std::string image_prompt = "A quiet futuristic library, warm natural light, cinematic, highly detailed";
    int image_steps = 20;
    std::string image_path;
    int image_width = 0;
    int image_height = 0;

    eui::Signal<float> workspace_scroll{0.0f};
};

constexpr const char* kBackendNames[] = {"CUDA", "SYCL", "CPU"};
constexpr const char* kLlmModel = "models/llm/llama_cpp/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf";
constexpr const char* kTtsConfig = "models/tts/gpt_sovits/configs/v2-q4.json";
constexpr const char* kImageModel = "models/visual_generation/stable_diffusion_cpp/sd1.5/DreamShaper_8_pruned.safetensors";

inline std::string resolveProjectPath(const std::string& relative) {
    namespace fs = std::filesystem;
    const fs::path input = fs::u8path(relative);
    if (input.is_absolute()) return input.string();
    for (fs::path base = fs::current_path(); !base.empty(); base = base.parent_path()) {
        const fs::path candidate = base / input;
        if (fs::exists(candidate)) return fs::absolute(candidate).string();
        if (base == base.parent_path()) break;
    }
    return fs::absolute(input).string();
}

inline std::string outputPath(const std::string& filename) {
    namespace fs = std::filesystem;
    fs::path root = fs::current_path();
    for (fs::path base = root; !base.empty(); base = base.parent_path()) {
        if (fs::exists(base / "CMakeLists.txt") && fs::exists(base / "models")) {
            root = base;
            break;
        }
        if (base == base.parent_path()) break;
    }
    const fs::path directory = root / "outputs" / "ui-demo";
    fs::create_directories(directory);
    return (directory / filename).string();
}

} // namespace app
