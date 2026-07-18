#include "categories/asr/asr.h"
#include "common/wav.h"

#include <iostream>

namespace {

bool print_event(const asr_event* event, void*) {
    if (!event) return true;
    if (event->type == ASR_EVENT_LANGUAGE && event->language) {
        std::cerr << "Detected language: " << event->language << '\n';
    } else if (event->type == ASR_EVENT_SEGMENT && event->text) {
        std::cout.write(event->text, static_cast<std::streamsize>(event->text_length));
        std::cout.flush();
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.bin> <audio.wav> [language|auto] [device|auto]\n";
        return 2;
    }

    const example::Audio audio = example::load_wav(argv[2]);
    if (audio.samples.empty()) {
        std::cerr << "Failed to read PCM16/float32 WAV: " << argv[2] << '\n';
        return 1;
    }

    asr_runtime_params runtime_params = asr_runtime_default_params();
    runtime_params.device = argc > 4 ? argv[4] : "auto";
    asr_runtime_ptr runtime = asr_runtime_create(runtime_params);
    asr_model_ptr model = runtime ? asr_load_model(runtime, argv[1]) : nullptr;
    asr_session_ptr session = model ? asr_create_session(model) : nullptr;
    if (!session) {
        std::cerr << "Failed to initialize ASR.\n";
        asr_free_session(session);
        asr_free_model(model);
        asr_runtime_free(runtime);
        return 1;
    }

    const asr_capabilities capabilities = asr_model_get_capabilities(model);
    std::cerr << "Provider: " << asr_model_get_provider(model)
              << ", device: " << asr_runtime_get_device(runtime)
              << ", language detection: " << (capabilities.language_detection ? "yes" : "no")
              << '\n';

    asr_request_params request = asr_request_default_params();
    request.language = argc > 3 ? argv[3] : "auto";
    const bool ok = asr_transcribe(
        session, audio.samples.data(), audio.samples.size(), audio.sample_rate,
        request, print_event, nullptr);
    std::cout << '\n';

    asr_free_session(session);
    asr_free_model(model);
    asr_runtime_free(runtime);
    return ok ? 0 : 1;
}
