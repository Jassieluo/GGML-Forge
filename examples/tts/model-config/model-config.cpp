#include "categories/tts/tts.h"
#include "audio/audio_io.h"

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const char* config_path = argc > 1
        ? argv[1]
        : "models/tts/gpt_sovits/configs/v3-q4.json";

    tts_runtime_params params = tts_runtime_default_params();
    if (argc > 2) params.device = argv[2];
    params.n_threads = 4;
    params.max_concurrency = 2;

    tts_runtime_ptr runtime = tts_runtime_create(params);
    if (!runtime) {
        std::cerr << "Failed to create TTS runtime." << std::endl;
        return 1;
    }
    std::cout << "Runtime device: " << tts_runtime_get_device(runtime)
              << ", threads per lane: " << tts_runtime_get_thread_count(runtime)
              << ", max concurrency: " << tts_runtime_get_max_concurrency(runtime) << std::endl;

    // Component IDs are provider-defined. Device=nullptr inherits the runtime
    // default; on-demand components are unloaded after each request.
    if (!tts_runtime_set_component_policy(
            runtime, "hubert", nullptr, TTS_COMPONENT_ON_DEMAND)) {
        std::cerr << "Failed to configure HuBERT residency." << std::endl;
        tts_runtime_free(runtime);
        return 1;
    }

    tts_model_ptr model = tts_load_model(runtime, config_path);
    if (!model) {
        std::cerr << "Failed to load model composition: " << config_path << std::endl;
        tts_runtime_free(runtime);
        return 1;
    }

    std::cout << "Loaded model composition: " << config_path << std::endl;
    tts_session_ptr first_session = tts_create_session(model);
    tts_session_ptr second_session = tts_create_session(model);
    if (!first_session || !second_session) {
        std::cerr << "Failed to create independent TTS sessions." << std::endl;
        tts_free_session(first_session);
        tts_free_session(second_session);
        tts_free_model(model);
        tts_runtime_free(runtime);
        return 1;
    }
    std::cout << "Created two independent sessions sharing one loaded model." << std::endl;
    tts_free_model(model);
    model = nullptr;
    std::cout << "Sessions retained shared model state after releasing the model handle." << std::endl;
    std::vector<float> first_reference(160, 0.0f);
    std::vector<float> second_reference(160, 0.25f);
    if (!tts_session_set_reference(first_session, first_reference.data(), first_reference.size(), 16000, "a", "zh") ||
        !tts_session_set_reference(second_session, second_reference.data(), second_reference.size(), 16000, "b", "zh")) {
        std::cerr << "Failed to assign independent session references." << std::endl;
        tts_free_session(first_session);
        tts_free_session(second_session);
        tts_runtime_free(runtime);
        return 1;
    }
    std::cout << "Assigned independent voice references to both sessions." << std::endl;

    if (argc > 3) {
        std::vector<float> reference;
        uint32_t reference_rate = 0;
        std::string media_error;
        if (!forge::media::load_wav_mono(
                std::filesystem::u8path(argv[3]), reference, reference_rate, media_error) ||
            !tts_session_set_reference(first_session, reference.data(), reference.size(), static_cast<int32_t>(reference_rate), "hello", "en") ||
            !tts_session_set_reference(second_session, reference.data(), reference.size(), static_cast<int32_t>(reference_rate), "hello", "en")) {
            std::cerr << "Failed to prepare concurrency reference." << std::endl;
            tts_free_session(first_session);
            tts_free_session(second_session);
            tts_runtime_free(runtime);
            return 1;
        }

        int32_t first_samples = 0;
        int32_t second_samples = 0;
        const auto start = std::chrono::steady_clock::now();
        std::thread first([&] { tts_synthesize(first_session, "hello world", "en", 1.0f, &first_samples); });
        std::thread second([&] { tts_synthesize(second_session, "testing speech", "en", 1.0f, &second_samples); });
        first.join();
        second.join();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "Concurrent synthesis: first=" << first_samples
                  << ", second=" << second_samples << ", wall_ms=" << elapsed << std::endl;
        if (first_samples <= 0 || second_samples <= 0) {
            tts_free_session(first_session);
            tts_free_session(second_session);
            tts_runtime_free(runtime);
            return 1;
        }
    }

    tts_free_session(first_session);
    tts_free_session(second_session);
    tts_runtime_free(runtime);
    return 0;
}
