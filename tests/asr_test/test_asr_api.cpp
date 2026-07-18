#include "categories/asr/asr.h"

#include <cstring>
#include <iostream>

namespace {

bool sink(const asr_event*, void*) { return true; }

} // namespace

int main(int argc, char** argv) {
    asr_runtime_params defaults = asr_runtime_default_params();
    if (!defaults.device || std::strcmp(defaults.device, "auto") != 0 || defaults.n_threads == 0) {
        std::cerr << "invalid ASR runtime defaults\n";
        return 1;
    }

    asr_runtime_ptr runtime = asr_runtime_create({"cpu", 2});
    if (!runtime || std::strcmp(asr_runtime_get_device(runtime), "cpu") != 0 ||
        asr_runtime_get_thread_count(runtime) != 2) {
        std::cerr << "ASR runtime did not retain configuration\n";
        return 1;
    }

    const asr_request_params request = asr_request_default_params();
    if (request.task != ASR_TASK_TRANSCRIBE || !request.language ||
        std::strcmp(request.language, "auto") != 0 || request.token_timestamps) {
        std::cerr << "invalid ASR request defaults\n";
        return 1;
    }

    const float sample = 0.0f;
    if (asr_load_model(nullptr, "missing.bin") ||
        asr_load_model(runtime, nullptr) ||
        asr_create_session(nullptr) ||
        asr_session_reset(nullptr) ||
        asr_transcribe(nullptr, &sample, 1, 16000, request, sink, nullptr)) {
        std::cerr << "ASR null-handle contract failed\n";
        return 1;
    }

    const asr_capabilities empty = asr_model_get_capabilities(nullptr);
    if (empty.streaming || empty.translation || empty.language_detection ||
        empty.segment_timestamps || empty.token_timestamps) {
        std::cerr << "null ASR model reported capabilities\n";
        return 1;
    }

    if (argc > 1) {
        asr_model_ptr model = asr_load_model(runtime, argv[1]);
        if (!model || !asr_model_get_provider(model) ||
            std::strcmp(asr_model_get_provider(model), "whisper.cpp") != 0) {
            std::cerr << "whisper.cpp provider failed to load its test model\n";
            asr_free_model(model);
            asr_runtime_free(runtime);
            return 1;
        }
        const asr_capabilities capabilities = asr_model_get_capabilities(model);
        if (capabilities.streaming || capabilities.translation ||
            capabilities.language_detection || !capabilities.segment_timestamps ||
            !capabilities.token_timestamps) {
            std::cerr << "whisper.cpp provider reported invalid capabilities\n";
            asr_free_model(model);
            asr_runtime_free(runtime);
            return 1;
        }
        asr_session_ptr session = asr_create_session(model);
        if (!session || !asr_session_reset(session)) {
            std::cerr << "whisper.cpp provider failed to create/reset a session\n";
            asr_free_session(session);
            asr_free_model(model);
            asr_runtime_free(runtime);
            return 1;
        }
        asr_free_session(session);
        asr_free_model(model);
    }

    asr_runtime_free(runtime);
    std::cout << "ASR API lifecycle checks passed\n";
    return 0;
}
