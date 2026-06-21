#include "src/models/model.h"
#include <iostream>

namespace tts {

std::unique_ptr<ModelArch> create_model_arch(const std::string& arch_name) {
    if (arch_name == "vits") {
        std::cout << "[TTS Core] Loading VITS model architecture..." << std::endl;
        // In the next step, we will instantiate the VITS model class
        // return std::make_unique<VITSModelArch>();
    } else if (arch_name == "gpt-t2s") {
        std::cout << "[TTS Core] Loading GPT Text-to-Semantic model architecture..." << std::endl;
        // return std::make_unique<GPTT2SModelArch>();
    } else {
        std::cerr << "[TTS Core] Warning: Unknown model architecture: " << arch_name << std::endl;
    }
    return nullptr;
}

} // namespace tts
