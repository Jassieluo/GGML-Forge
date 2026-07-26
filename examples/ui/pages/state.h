#pragma once

#include "eui/signal.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace app {

enum class Tool { Chat, Speech, Image };

enum Backend : int {
    kBackendCuda = 0,
    kBackendSycl = 1,
    kBackendCpu = 2,
    kBackendCount = 3,
};

constexpr const char* kBackendNames[kBackendCount] = {"CUDA", "SYCL", "CPU"};

constexpr const char* kLlmModel = "models/llm/llama_cpp/qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf";
constexpr const char* kTtsConfig = "models/tts/gpt_sovits/configs/v2-q4.json";
constexpr const char* kImageModel = "models/visual_generation/stable_diffusion_cpp/sd1.5/DreamShaper_8_pruned.safetensors";

// Default voice for GPT-SoVITS: a session must own a reference (audio +
// transcript) before it can synthesize; without one the provider fails with
// "Prompt cache ID not found".
constexpr const char* kTtsVoiceWav = "models/tts/gpt_sovits/voices/doubao/audios/平静.wav";
constexpr const char* kTtsVoiceText = "凡事顺其自然，放平心态，好好生活就够了。";
constexpr const char* kTtsVoiceLang = "zh";

// Roles: "user", "assistant", and "tool" (tool results rendered with
// optional image/audio attachments; forwarded to the LLM as user text).
struct ChatMessage {
    std::string role;
    std::string text;
    std::string thinking;        // <think> reasoning, shown behind a toggle.
    bool think_expanded = false;
    std::string image_path;
    int image_width = 0;
    int image_height = 0;
    std::string audio_path;
};

// Worker -> UI bridge for streamed LLM tokens. The worker thread appends under
// the mutex; the UI thread polls once per frame and copies out on change.
class StreamBuffer {
public:
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        text_.clear();
        ++version_;
    }

    void append(const char* bytes, std::size_t length) {
        std::lock_guard<std::mutex> lock(mutex_);
        text_.append(bytes, length);
        ++version_;
    }

    // Returns true (and refreshes `out`) only when the buffer changed since
    // `seen`, so the UI avoids re-copying the string every frame.
    bool snapshot(std::uint64_t& seen, std::string& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (version_ == seen) return false;
        seen = version_;
        out = text_;
        return true;
    }

private:
    std::mutex mutex_;
    std::string text_;
    std::uint64_t version_ = 0;
};

struct StudioState {
    Tool tool = Tool::Chat;
    int backend = kBackendCuda;
    bool busy = false;
    bool has_error = false;
    std::atomic<float> progress{0.0f};
    std::atomic<bool> cancel_requested{false};
    std::string status = "就绪";

    // Chat
    std::vector<ChatMessage> messages = {
        {"assistant", "你好，我是运行在 GGML-Forge 上的本地助手。你可以对话、合成语音或生成图片。"},
    };
    std::string chat_input;
    bool streaming = false;         // UI flag: a chat generation is in flight.
    StreamBuffer stream;            // Shared with the worker thread.
    std::string streaming_text;     // UI-thread copy of the partial answer.
    std::uint64_t stream_seen = 0;  // Last StreamBuffer version copied.
    int agent_rounds = 0;           // Tool calls consumed by the current turn.
    eui::Signal<float> chat_scroll{0.0f};
    // Stick-to-bottom: while set, the chat list renders as a bottom-justified
    // clipped column (always showing the newest content); scrolling up hands
    // control to a scroll view. The epoch remounts that scroll view so its
    // seeded offset is honored (the runtime ignores seeds on live instances).
    bool chat_stick_bottom = true;
    int chat_scroll_epoch = 0;
    int chat_layout_epoch = 0;      // Bumped when a think toggle changes bubble heights.

    // Speech
    std::string speech_text = "欢迎使用 GGML-Forge 本地语音合成演示。";
    float speech_speed = 1.0f;
    std::string audio_path;
    double audio_seconds = 0.0;

    // Image
    std::string image_prompt = "A quiet futuristic library, warm natural light, cinematic, highly detailed";
    int image_steps = 20;
    std::string image_path;
    int image_width = 0;
    int image_height = 0;
};

inline StudioState state;

// Single source of truth for locating the repository root: the nearest parent
// directory that contains both CMakeLists.txt and models/. Resolved once.
inline const std::filesystem::path& projectRoot() {
    static const std::filesystem::path root = [] {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path current = fs::current_path(ec);
        for (fs::path base = current; !base.empty(); base = base.parent_path()) {
            if (fs::exists(base / "CMakeLists.txt", ec) && fs::exists(base / "models", ec)) {
                return base;
            }
            if (base == base.parent_path()) break;
        }
        return current;
    }();
    return root;
}

// All path strings in the app are UTF-8 (sources are /utf-8, and every file
// open goes through u8path). path::string() would re-encode through the
// active code page and corrupt non-ASCII names on Windows — u8string() keeps
// the round trip lossless.
inline std::string resolveProjectPath(const std::string& relative) {
    const std::filesystem::path input = std::filesystem::u8path(relative);
    if (input.is_absolute()) return input.u8string();
    return (projectRoot() / input).u8string();
}

inline std::string outputPath(const std::string& filename) {
    namespace fs = std::filesystem;
    const fs::path directory = projectRoot() / "outputs" / "ui-demo";
    std::error_code ec;
    fs::create_directories(directory, ec);
    return (directory / std::filesystem::u8path(filename)).u8string();
}

inline std::string filenameOnly(const std::string& path) {
    return path.empty() ? std::string{} : std::filesystem::u8path(path).filename().u8string();
}

} // namespace app
