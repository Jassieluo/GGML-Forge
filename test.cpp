#include <iostream>
#include <fstream>
int main() {
    std::ifstream f("D:/Projects/CMake Projects/GPT-SoVITS.cpp/scratch/py_prompt_semantics.bin", std::ios::binary);
    if (!f.is_open()) {
        std::cout << "failed to open" << std::endl;
        return 1;
    }
    f.seekg(0, std::ios::end);
    std::cout << "tellg size: " << f.tellg() << std::endl;
    return 0;
}
