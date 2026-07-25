#pragma once

#include "categories/llm/llm.h"
#include "categories/tts/tts.h"
#include "categories/visual_generation/visual_generation.h"
#include "core/platform/async.h"
#include "pages/state.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <mmsystem.h>
#endif

namespace app {

struct LlmResult { std::string text; };
struct SpeechResult { std::string path; double seconds = 0.0; };
struct ImageResult { std::string path; int width = 0; int height = 0; };

class EngineService {
public:
    static core::async::Result<LlmResult> chat(
        const std::vector<ChatMessage>& history, int backend, std::atomic<float>* progress) {
        setProgress(progress, 0.08f);
        const std::string model_path = resolveProjectPath(kLlmModel);
        if (!std::filesystem::exists(model_path))
            return core::async::failure<LlmResult>("LLM model not found: " + model_path);

        llm_runtime_params runtime_params = llm_runtime_default_params();
        runtime_params.n_ctx = 4096;
        runtime_params.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
        runtime_params.n_gpu_layers = backend == 2 ? 0 : -1;
        llm_runtime_ptr runtime = llm_runtime_create(runtime_params);
        if (!runtime) return core::async::failure<LlmResult>("Failed to create LLM runtime");
        setProgress(progress, 0.18f);

        llm_model_ptr model = llm_load_model(runtime, model_path.c_str());
        llm_session_ptr session = model ? llm_create_session(model) : nullptr;
        if (!session) {
            llm_free_model(model);
            llm_runtime_free(runtime);
            return core::async::failure<LlmResult>("Failed to load the LLM model");
        }
        setProgress(progress, 0.35f);

        std::vector<llm_chat_message> api_messages;
        api_messages.reserve(history.size());
        for (const auto& message : history)
            api_messages.push_back({message.role.c_str(), message.text.c_str()});

        LlmResult result;
        llm_generation_params generation = llm_generation_default_params();
        generation.max_tokens = 512;
        const bool ok = llm_generate_chat(
            session, api_messages.data(), api_messages.size(), generation,
            [](const char* bytes, size_t length, void* user_data) {
                static_cast<LlmResult*>(user_data)->text.append(bytes, length);
                return true;
            }, &result);

        llm_free_session(session);
        llm_free_model(model);
        llm_runtime_free(runtime);
        setProgress(progress, 1.0f);
        if (!ok || result.text.empty()) return core::async::failure<LlmResult>("LLM generation failed");
        return core::async::success(std::move(result));
    }

    static core::async::Result<SpeechResult> synthesize(
        std::string text, float speed, int backend, std::atomic<float>* progress) {
        setProgress(progress, 0.08f);
        const std::string config_path = resolveProjectPath(kTtsConfig);
        if (!std::filesystem::exists(config_path))
            return core::async::failure<SpeechResult>("TTS config not found: " + config_path);

        tts_runtime_params params = tts_runtime_default_params();
        const std::string device = backend == 0 ? "CUDA0" : backend == 1 ? "SYCL0" : "cpu";
        params.device = device.c_str();
        params.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
        tts_runtime_ptr runtime = tts_runtime_create(params);
        tts_model_ptr model = runtime ? tts_load_model(runtime, config_path.c_str()) : nullptr;
        tts_session_ptr session = model ? tts_create_session(model) : nullptr;
        if (!session) {
            tts_free_model(model);
            tts_runtime_free(runtime);
            return core::async::failure<SpeechResult>("Failed to load the TTS model");
        }
        setProgress(progress, 0.42f);

        int32_t sample_count = 0;
        const float* samples = tts_synthesize(session, text.c_str(), "zh", speed, &sample_count);
        const int32_t sample_rate = tts_session_get_output_sample_rate(session);
        SpeechResult result;
        result.path = outputPath("speech-" + timestamp() + ".wav");
        result.seconds = sample_rate > 0 ? static_cast<double>(sample_count) / sample_rate : 0.0;
        const bool written = samples && sample_count > 0 &&
            writeWav(result.path, samples, static_cast<size_t>(sample_count), sample_rate);

        tts_free_session(session);
        tts_free_model(model);
        tts_runtime_free(runtime);
        setProgress(progress, 1.0f);
        if (!written) return core::async::failure<SpeechResult>("Speech synthesis or WAV output failed");
        return core::async::success(std::move(result));
    }

    static core::async::Result<ImageResult> generateImage(
        std::string prompt, int steps, int backend, std::atomic<float>* progress) {
        setProgress(progress, 0.05f);
        const std::string model_path = resolveProjectPath(kImageModel);
        if (!std::filesystem::exists(model_path))
            return core::async::failure<ImageResult>("Image model not found: " + model_path);

        visual_runtime_params runtime_params = visual_runtime_default_params();
        const std::string backend_name = backend == 0 ? "cuda" : backend == 1 ? "sycl" : "cpu";
        runtime_params.backend = backend_name.c_str();
        runtime_params.n_threads = std::max(1u, std::thread::hardware_concurrency() / 2u);
        visual_runtime_ptr runtime = visual_runtime_create(runtime_params);
        visual_model_params model_params = visual_model_default_params();
        model_params.model = model_path.c_str();
        visual_model_ptr model = runtime ? visual_load_model(runtime, &model_params) : nullptr;
        visual_session_ptr session = model ? visual_create_session(model) : nullptr;
        if (!session || !visual_model_get_capabilities(model).text_to_image) {
            visual_free_session(session);
            visual_free_model(model);
            visual_runtime_free(runtime);
            return core::async::failure<ImageResult>("Failed to load a text-to-image model");
        }
        setProgress(progress, 0.22f);
        visual_session_set_callbacks(session,
            [](int32_t step, int32_t total, float, void* data) {
                if (total > 0) setProgress(static_cast<std::atomic<float>*>(data),
                    0.22f + 0.73f * static_cast<float>(step) / static_cast<float>(total));
            }, nullptr, progress);

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
        const bool generated = visual_generate_images(session, &request, &images, &image_count);

        ImageResult result;
        if (generated && image_count > 0) {
            result.path = outputPath("image-" + timestamp() + ".bmp");
            result.width = static_cast<int>(images[0].width);
            result.height = static_cast<int>(images[0].height);
        }
        const bool written = generated && image_count > 0 && writeBmp(result.path, images[0]);
        visual_free_images(images, image_count);
        visual_free_session(session);
        visual_free_model(model);
        visual_runtime_free(runtime);
        setProgress(progress, 1.0f);
        if (!written) return core::async::failure<ImageResult>("Image generation or file output failed");
        return core::async::success(std::move(result));
    }

    static void playAudio(const std::string& path) {
#ifdef _WIN32
        const std::wstring wide = std::filesystem::u8path(path).wstring();
        PlaySoundW(wide.c_str(), nullptr, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
#else
        (void)path;
#endif
    }

private:
    static void setProgress(std::atomic<float>* value, float progress) {
        if (value) value->store(std::clamp(progress, 0.0f, 1.0f));
    }

    static std::string timestamp() {
        const auto value = std::chrono::system_clock::now().time_since_epoch();
        return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(value).count());
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
