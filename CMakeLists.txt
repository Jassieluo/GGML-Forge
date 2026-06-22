# CMakeLists.txt for GPT-SoVITS.cpp

cmake_minimum_required(VERSION 3.14)
project(gpt_sovits C CXX ASM)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (GGML_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)

add_subdirectory(src)

add_subdirectory(examples)

add_subdirectory(tests/softmax_debug)
add_subdirectory(tests/ops_test)


