#include "phonemizer.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "Starting test_phonemizer_hang..." << std::endl;
    auto phonemizer = std::make_unique<phonemizer::Phonemizer>("models/speech/dict");
    std::cout << "Phonemizer loaded successfully!" << std::endl;
    
    std::string ref_text = "欢迎来到营火，无火的余灰。";
    std::cout << "Processing ref_text: " << ref_text << std::endl;
    
    auto res = phonemizer->process(ref_text, "zh");
    std::cout << "Success! Phones count: " << res.phones.size() << std::endl;
    for (const auto& ph : res.phones) {
        std::cout << ph << " ";
    }
    std::cout << std::endl;
    
    std::string target_text = "你好，欢迎使用纯C加加推理的语音合成系统";
    std::cout << "Processing target_text: " << target_text << std::endl;
    auto res2 = phonemizer->process(target_text, "zh");
    std::cout << "Success! Phones count: " << res2.phones.size() << std::endl;
    for (const auto& ph : res2.phones) {
        std::cout << ph << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
