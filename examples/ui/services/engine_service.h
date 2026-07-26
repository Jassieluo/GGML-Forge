#pragma once

// Engine access for the ui-demo. Runtimes and models are cached per backend so
// switching pages or sending another message does not reload gigabytes from
// disk. All ensure/free/generate functions run on the core::async worker; the
// UI thread serializes access through the single state.busy guard, so no lock
// is needed around the caches themselves.

#include "categories/llm/llm.h"
#include "categories/tts/tts.h"
#include "categories/visual_generation/visual_generation.h"
#include "core/platform/async.h"
#include "ggml-backend.h"
#include "pages/state.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#endif

namespace app {

struct LlmResult {
    std::string text;
    bool canceled = false;
};
struct SpeechResult {
    std::string path;
    double seconds = 0.0;
};
struct ImageResult {
    std::string path;
    int width = 0;
    int height = 0;
    bool canceled = false;
};

class EngineService {
public:
    // Lock-free flags the UI reads to render "loaded" chips in the sidebar.
    struct LoadedFlags {
        std::atomic<bool> llm{false};
        std::atomic<bool> tts{false};
        std::atomic<bool> visual{false};
        bool any() const { return llm.load() || tts.load() || visual.load(); }
    };

    static LoadedFlags& loaded() {
        static LoadedFlags flags;
        return flags;
    }

    // True when the ggml device registry actually offers the backend. SYCL in
    // particular silently drops out when ggml-sycl.dll cannot load its oneAPI
    // runtime; probing once at startup lets the sidebar refuse dead options
    // instead of failing later inside a generation.
    static bool backendAvailable(int backend) {
        struct Probe {
            bool cuda = false;
            bool sycl = false;
        };
        static const Probe probe = [] {
            ggml_backend_load_all();
            Probe result;
            const size_t count = ggml_backend_dev_count();
            for (size_t i = 0; i < count; ++i) {
                const char* name = ggml_backend_dev_name(ggml_backend_dev_get(i));
                if (!name) continue;
                if (std::strncmp(name, "CUDA", 4) == 0) result.cuda = true;
                if (std::strncmp(name, "SYCL", 4) == 0) result.sycl = true;
            }
            return result;
        }();
        if (backend == kBackendCuda) return probe.cuda;
        if (backend == kBackendSycl) return probe.sycl;
        return true;  // CPU
    }

    // ---- Chat (streams tokens into `stream`; honors `cancel`) ----
    static core::async::Result<LlmResult> chat(
        const std::vector<ChatMessage>& history, int backend,
        std::atomic<float>* progress, StreamBuffer* stream,
        const std::atomic<bool>* cancel) {
        setProgress(progress, 0.05f);
        std::string error;
        if (!ensureLlm(backend, error)) return core::async::failure<LlmResult>(error);
        setProgress(progress, 0.30f);

        // A fresh session per request keeps KV state deterministic; the
        // expensive parts (runtime + weights) stay cached in llmEngine().
        llm_session_ptr session = llm_create_session(llmEngine().model);
        if (!session) return core::async::failure<LlmResult>("Failed to create the LLM session");

        std::vector<llm_chat_message> api_messages;
        api_messages.reserve(history.size());
        for (const auto& message : history)
            api_messages.push_back({message.role.c_str(), message.text.c_str()});

        struct StreamContext {
            LlmResult* result;
            StreamBuffer* stream;
            const std::atomic<bool>* cancel;
        };
        LlmResult result;
        StreamContext context{&result, stream, cancel};

        llm_generation_params generation = llm_generation_default_params();
        generation.max_tokens = 1024;
        const bool ok = llm_generate_chat(
            session, api_messages.data(), api_messages.size(), generation,
            [](const char* bytes, size_t length, void* user_data) {
                auto* ctx = static_cast<StreamContext*>(user_data);
                if (ctx->cancel && ctx->cancel->load()) return false;
                ctx->result->text.append(bytes, length);
                if (ctx->stream) ctx->stream->append(bytes, length);
                return true;
            },
            &context);
        llm_free_session(session);
        setProgress(progress, 1.0f);

        result.canceled = cancel && cancel->load();
        if (result.canceled) return core::async::success(std::move(result));
        if (!ok || result.text.empty())
            return core::async::failure<LlmResult>("LLM generation failed");
        return core::async::success(std::move(result));
    }

    // ---- Speech ----
    static core::async::Result<SpeechResult> synthesize(
        const std::string& text, float speed, int backend, std::atomic<float>* progress) {
        setProgress(progress, 0.05f);
        std::string error;
        if (!ensureTts(backend, error)) return core::async::failure<SpeechResult>(error);
        setProgress(progress, 0.40f);

        // Provider progress [0,1] lands in the 0.40..1.0 window that remains
        // after the load phase. state.progress is a stable global.
        tts_session_set_progress_callback(
            ttsEngine().session,
            [](float value, void* data) {
                static_cast<std::atomic<float>*>(data)->store(
                    std::clamp(0.40f + 0.60f * value, 0.0f, 1.0f));
            },
            &state.progress);

        int32_t sample_count = 0;
        const float* samples = tts_synthesize(ttsEngine().session, text.c_str(), "zh", speed, &sample_count);
        const int32_t sample_rate = tts_session_get_output_sample_rate(ttsEngine().session);

        SpeechResult result;
        result.path = outputPath("speech-" + timestamp() + ".wav");
        result.seconds = sample_rate > 0 ? static_cast<double>(sample_count) / sample_rate : 0.0;
        const bool written = samples && sample_count > 0 &&
            writeWav(result.path, samples, static_cast<size_t>(sample_count), sample_rate);
        setProgress(progress, 1.0f);
        if (!written) return core::async::failure<SpeechResult>("Speech synthesis or WAV output failed");
        return core::async::success(std::move(result));
    }

    // ---- Image ----
    static core::async::Result<ImageResult> generateImage(
        const std::string& prompt, int steps, int backend,
        std::atomic<float>* progress, const std::atomic<bool>* cancel) {
        setProgress(progress, 0.03f);
        std::string error;
        if (!ensureVisual(backend, error)) return core::async::failure<ImageResult>(error);
        setProgress(progress, 0.20f);

        visual_image_request request = visual_image_request_default_params();
        request.prompt = prompt.c_str();
        request.negative_prompt = "low quality, blurry, deformed, text, watermark";
        request.width = 512;
        request.height = 512;
        request.seed = -1;
        request.sample.steps = steps;
        request.sample.text_guidance = 7.0f;

        visual_image* images = nullptr;
        size_t image_count = 0;
        activeVisualSession().store(visualEngine().session);
        const bool generated = visual_generate_images(visualEngine().session, &request, &images, &image_count);
        activeVisualSession().store(nullptr);

        ImageResult result;
        result.canceled = cancel && cancel->load();
        if (result.canceled) {
            visual_free_images(images, image_count);
            setProgress(progress, 1.0f);
            return core::async::success(std::move(result));
        }

        if (generated && image_count > 0) {
            result.path = outputPath("image-" + timestamp() + ".bmp");
            result.width = static_cast<int>(images[0].width);
            result.height = static_cast<int>(images[0].height);
        }
        const bool written = generated && image_count > 0 && writeBmp(result.path, images[0]);
        visual_free_images(images, image_count);
        setProgress(progress, 1.0f);
        if (!written) return core::async::failure<ImageResult>("Image generation or file output failed");
        return core::async::success(std::move(result));
    }

    // Called from the UI thread while an image generation is in flight. The
    // atomic only holds a session between the store/clear pair inside
    // generateImage, and sessions are freed exclusively on the (busy-guarded)
    // worker, so the pointer read here cannot dangle.
    static void requestImageCancel() {
        if (visual_session_ptr session = activeVisualSession().load()) {
            visual_session_cancel(session, VISUAL_CANCEL_ALL);
        }
    }

    // ---- Lifecycle ----
    static core::async::Result<void> releaseAll() {
        freeLlm();
        freeTts();
        freeVisual();
        return core::async::success();
    }

    static void playAudio(const std::string& path) {
#ifdef _WIN32
        // PlaySound(SND_ASYNC) may read the name after returning; keep the
        // wide string alive. UI-thread only, so a static is sufficient.
        static std::wstring keep_alive;
        keep_alive = std::filesystem::u8path(path).wstring();
        PlaySoundW(keep_alive.c_str(), nullptr, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
#else
        (void)path;
#endif
    }

private:
    struct LlmEngine {
        int backend = -1;
        llm_runtime_ptr runtime = nullptr;
        llm_model_ptr model = nullptr;
    };
    struct TtsEngine {
        int backend = -1;
        tts_runtime_ptr runtime = nullptr;
        tts_model_ptr model = nullptr;
        tts_session_ptr session = nullptr;
    };
    struct VisualEngine {
        int backend = -1;
        visual_runtime_ptr runtime = nullptr;
        visual_model_ptr model = nullptr;
        visual_session_ptr session = nullptr;
    };

    // Function-local statics: a nested class cannot be brace-initialized in an
    // inline static member before the enclosing class is complete.
    static LlmEngine& llmEngine() { static LlmEngine engine; return engine; }
    static TtsEngine& ttsEngine() { static TtsEngine engine; return engine; }
    static VisualEngine& visualEngine() { static VisualEngine engine; return engine; }
    static std::atomic<visual_session_ptr>& activeVisualSession() {
        static std::atomic<visual_session_ptr> session{nullptr};
        return session;
    }

    static uint32_t workerThreads() {
        return std::max(1u, std::thread::hardware_concurrency() / 2u);
    }

    static void freeLlm() {
        llm_free_model(llmEngine().model);
        llm_runtime_free(llmEngine().runtime);
        llmEngine() = {};
        loaded().llm.store(false);
    }

    static void freeTts() {
        tts_free_session(ttsEngine().session);
        tts_free_model(ttsEngine().model);
        tts_runtime_free(ttsEngine().runtime);
        ttsEngine() = {};
        loaded().tts.store(false);
    }

    static void freeVisual() {
        visual_free_session(visualEngine().session);
        visual_free_model(visualEngine().model);
        visual_runtime_free(visualEngine().runtime);
        visualEngine() = {};
        loaded().visual.store(false);
    }

    static bool ensureLlm(int backend, std::string& error) {
        if (llmEngine().model && llmEngine().backend == backend) return true;
        freeLlm();

        const std::string model_path = resolveProjectPath(kLlmModel);
        if (!std::filesystem::exists(std::filesystem::u8path(model_path))) {
            error = "LLM model not found: " + model_path;
            return false;
        }

        llm_runtime_params params = llm_runtime_default_params();
        params.n_ctx = 4096;
        params.n_threads = workerThreads();
        // The llama.cpp provider exposes no per-device selection; CPU means no
        // offload, anything else offloads to the provider-selected GPU.
        params.n_gpu_layers = backend == kBackendCpu ? 0 : -1;
        llmEngine().runtime = llm_runtime_create(params);
        if (!llmEngine().runtime) {
            error = "Failed to create the LLM runtime";
            return false;
        }
        llmEngine().model = llm_load_model(llmEngine().runtime, model_path.c_str());
        if (!llmEngine().model) {
            freeLlm();
            error = "Failed to load the LLM model";
            return false;
        }
        llmEngine().backend = backend;
        loaded().llm.store(true);
        return true;
    }

    static bool ensureTts(int backend, std::string& error) {
        if (ttsEngine().session && ttsEngine().backend == backend) return true;
        freeTts();

        const std::string config_path = resolveProjectPath(kTtsConfig);
        if (!std::filesystem::exists(std::filesystem::u8path(config_path))) {
            error = "TTS config not found: " + config_path;
            return false;
        }

        tts_runtime_params params = tts_runtime_default_params();
        const char* device = backend == kBackendCuda ? "CUDA0"
                           : backend == kBackendSycl ? "SYCL0"
                                                     : "cpu";
        params.device = device;  // Resolved and copied inside tts_runtime_create.
        params.n_threads = workerThreads();
        ttsEngine().runtime = tts_runtime_create(params);
        ttsEngine().model = ttsEngine().runtime ? tts_load_model(ttsEngine().runtime, config_path.c_str()) : nullptr;
        ttsEngine().session = ttsEngine().model ? tts_create_session(ttsEngine().model) : nullptr;
        if (!ttsEngine().session) {
            freeTts();
            error = "Failed to load the TTS model";
            return false;
        }

        // GPT-SoVITS requires a reference voice per session; without one every
        // synthesis fails ("Prompt cache ID not found").
        const std::string voice_path = resolveProjectPath(kTtsVoiceWav);
        int voice_rate = 0;
        const std::vector<float> voice = readWav(voice_path, voice_rate);
        if (voice.empty() ||
            !tts_session_set_reference(ttsEngine().session, voice.data(), voice.size(),
                                       voice_rate, kTtsVoiceText, kTtsVoiceLang)) {
            freeTts();
            error = "Failed to load the reference voice: " + voice_path;
            return false;
        }
        ttsEngine().backend = backend;
        loaded().tts.store(true);
        return true;
    }

    static bool ensureVisual(int backend, std::string& error) {
        if (visualEngine().session && visualEngine().backend == backend) return true;
        freeVisual();

        const std::string model_path = resolveProjectPath(kImageModel);
        if (!std::filesystem::exists(std::filesystem::u8path(model_path))) {
            error = "Image model not found: " + model_path;
            return false;
        }

        visual_runtime_params runtime_params = visual_runtime_default_params();
        const char* backend_name = backend == kBackendCuda ? "cuda"
                                 : backend == kBackendSycl ? "sycl"
                                                           : "cpu";
        runtime_params.backend = backend_name;
        runtime_params.n_threads = workerThreads();
        visualEngine().runtime = visual_runtime_create(runtime_params);

        visual_model_params model_params = visual_model_default_params();
        model_params.model = model_path.c_str();
        visualEngine().model = visualEngine().runtime ? visual_load_model(visualEngine().runtime, &model_params) : nullptr;
        visualEngine().session = visualEngine().model ? visual_create_session(visualEngine().model) : nullptr;
        if (!visualEngine().session || !visual_model_get_capabilities(visualEngine().model).text_to_image) {
            freeVisual();
            error = "Failed to load a text-to-image model";
            return false;
        }
        // state.progress is a stable global, so binding it once is safe for
        // the lifetime of the cached session.
        visual_session_set_callbacks(
            visualEngine().session,
            [](int32_t step, int32_t total, float, void* data) {
                if (total > 0) {
                    static_cast<std::atomic<float>*>(data)->store(
                        std::clamp(0.20f + 0.78f * static_cast<float>(step) / static_cast<float>(total),
                                   0.0f, 1.0f));
                }
            },
            nullptr, &state.progress);
        visualEngine().backend = backend;
        loaded().visual.store(true);
        return true;
    }

    static void setProgress(std::atomic<float>* value, float progress) {
        if (value) value->store(std::clamp(progress, 0.0f, 1.0f));
    }

    static std::string timestamp() {
        const auto value = std::chrono::system_clock::now().time_since_epoch();
        return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(value).count());
    }

    // Minimal RIFF reader for the bundled reference voices: 16-bit PCM or
    // float32, any channel count (downmixed to mono). u8path keeps non-ASCII
    // voice filenames working on Windows.
    static std::vector<float> readWav(const std::string& path, int& sample_rate) {
        sample_rate = 0;
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
        char id[4] = {};
        uint32_t chunk_size = 0;
        if (!file.read(id, 4) || std::memcmp(id, "RIFF", 4) != 0 ||
            !file.read(reinterpret_cast<char*>(&chunk_size), 4) ||
            !file.read(id, 4) || std::memcmp(id, "WAVE", 4) != 0) {
            return {};
        }
        uint16_t format = 0, channels = 0, bits = 0;
        uint32_t rate = 0;
        std::vector<char> data;
        while (file.read(id, 4) && file.read(reinterpret_cast<char*>(&chunk_size), 4)) {
            if (std::memcmp(id, "fmt ", 4) == 0 && chunk_size >= 16) {
                file.read(reinterpret_cast<char*>(&format), 2);
                file.read(reinterpret_cast<char*>(&channels), 2);
                file.read(reinterpret_cast<char*>(&rate), 4);
                file.seekg(6, std::ios::cur);
                file.read(reinterpret_cast<char*>(&bits), 2);
                file.seekg(chunk_size - 16, std::ios::cur);
            } else if (std::memcmp(id, "data", 4) == 0) {
                data.resize(chunk_size);
                file.read(data.data(), chunk_size);
                break;
            } else {
                file.seekg(chunk_size + (chunk_size & 1), std::ios::cur);
            }
        }
        if (data.empty() || channels == 0 || rate == 0) return {};
        std::vector<float> mono;
        if (format == 1 && bits == 16) {
            const auto* pcm = reinterpret_cast<const int16_t*>(data.data());
            const size_t frames = data.size() / sizeof(int16_t) / channels;
            mono.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < channels; ++c) sum += pcm[i * channels + c] / 32768.0f;
                mono[i] = sum / channels;
            }
        } else if (format == 3 && bits == 32) {
            const auto* f32 = reinterpret_cast<const float*>(data.data());
            const size_t frames = data.size() / sizeof(float) / channels;
            mono.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                float sum = 0.0f;
                for (uint16_t c = 0; c < channels; ++c) sum += f32[i * channels + c];
                mono[i] = sum / channels;
            }
        } else {
            return {};
        }
        sample_rate = static_cast<int>(rate);
        return mono;
    }

    static bool writeWav(const std::string& path, const float* samples, size_t count, int sample_rate) {
        if (!samples || count == 0 || sample_rate <= 0) return false;
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
        if (!file) return false;
        const uint16_t format = 1, channels = 1, bits = 16;
        const uint16_t block_align = channels * bits / 8;
        const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
        const uint32_t data_size = static_cast<uint32_t>(count * sizeof(int16_t));
        const uint32_t riff_size = 36 + data_size, fmt_size = 16;
        file.write("RIFF", 4); file.write(reinterpret_cast<const char*>(&riff_size), 4);
        file.write("WAVEfmt ", 8); file.write(reinterpret_cast<const char*>(&fmt_size), 4);
        file.write(reinterpret_cast<const char*>(&format), 2);
        file.write(reinterpret_cast<const char*>(&channels), 2);
        file.write(reinterpret_cast<const char*>(&sample_rate), 4);
        file.write(reinterpret_cast<const char*>(&byte_rate), 4);
        file.write(reinterpret_cast<const char*>(&block_align), 2);
        file.write(reinterpret_cast<const char*>(&bits), 2);
        file.write("data", 4); file.write(reinterpret_cast<const char*>(&data_size), 4);
        for (size_t i = 0; i < count; ++i) {
            const int16_t pcm = static_cast<int16_t>(std::clamp(samples[i], -1.0f, 1.0f) * 32767.0f);
            file.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
        }
        return static_cast<bool>(file);
    }

    static bool writeBmp(const std::string& path, const visual_image& image) {
        if (!image.data || image.width == 0 || image.height == 0 || image.channels < 3) return false;
        std::ofstream file(std::filesystem::u8path(path), std::ios::binary);
        if (!file) return false;
        const uint32_t row_size = (image.width * 3u + 3u) & ~3u;
        const uint32_t pixel_size = row_size * image.height;
        const uint32_t file_size = 54u + pixel_size;
        const uint16_t signature = 0x4D42, planes = 1, bits = 24;
        const uint32_t zero = 0, offset = 54, dib_size = 40, compression = 0;
        const int32_t width = static_cast<int32_t>(image.width);
        const int32_t height = static_cast<int32_t>(image.height);
        file.write(reinterpret_cast<const char*>(&signature), 2);
        file.write(reinterpret_cast<const char*>(&file_size), 4);
        file.write(reinterpret_cast<const char*>(&zero), 4);
        file.write(reinterpret_cast<const char*>(&offset), 4);
        file.write(reinterpret_cast<const char*>(&dib_size), 4);
        file.write(reinterpret_cast<const char*>(&width), 4);
        file.write(reinterpret_cast<const char*>(&height), 4);
        file.write(reinterpret_cast<const char*>(&planes), 2);
        file.write(reinterpret_cast<const char*>(&bits), 2);
        file.write(reinterpret_cast<const char*>(&compression), 4);
        file.write(reinterpret_cast<const char*>(&pixel_size), 4);
        file.write(reinterpret_cast<const char*>(&zero), 4);
        file.write(reinterpret_cast<const char*>(&zero), 4);
        file.write(reinterpret_cast<const char*>(&zero), 4);
        file.write(reinterpret_cast<const char*>(&zero), 4);
        const std::vector<char> padding(row_size - image.width * 3u, 0);
        for (int32_t y = height - 1; y >= 0; --y) {
            for (uint32_t x = 0; x < image.width; ++x) {
                const uint8_t* pixel = image.data + (static_cast<size_t>(y) * image.width + x) * image.channels;
                const char bgr[3] = {static_cast<char>(pixel[2]), static_cast<char>(pixel[1]), static_cast<char>(pixel[0])};
                file.write(bgr, 3);
            }
            if (!padding.empty()) file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        }
        return static_cast<bool>(file);
    }
};

} // namespace app
