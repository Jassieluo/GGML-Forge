# CMakeLists.txt for GPT-SoVITS.cpp

cmake_minimum_required(VERSION 3.14)
project(gpt_sovits C CXX ASM)

option(INFERENCE_BUILD_TTS "Build the TTS category" ON)
option(INFERENCE_BUILD_EXAMPLES "Build category examples" ON)
option(INFERENCE_BUILD_TESTS "Build tests" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (GGML_CUDA)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)

add_subdirectory(src)

if (INFERENCE_BUILD_EXAMPLES AND INFERENCE_BUILD_TTS)
    add_subdirectory(examples)
endif()

if (INFERENCE_BUILD_TESTS)
    add_subdirectory(tests/ops_test)
    add_subdirectory(tests/nn_test)
    if (INFERENCE_BUILD_TTS)
        add_subdirectory(tests/model_config_test)
    endif()
endif()
