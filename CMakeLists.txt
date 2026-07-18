# CMakeLists.txt for GGML-Forge

cmake_minimum_required(VERSION 3.14)
project(ggml_forge C CXX ASM)

option(INFERENCE_BUILD_TTS "Build the TTS category" ON)
option(INFERENCE_BUILD_LLM "Build the LLM category with llama.cpp" ON)
option(INFERENCE_BUILD_ASR "Build the ASR category with whisper.cpp" ON)
option(INFERENCE_BUILD_VISUAL_GENERATION "Build visual generation with stable-diffusion.cpp" ON)
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

# stable-diffusion.cpp requires longer tensor names. This is an ABI-affecting
# GGML definition, so every GGML consumer must compile with the same value.
add_compile_definitions(GGML_MAX_NAME=160)

add_subdirectory(ggml)

add_subdirectory(src)

if (INFERENCE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (INFERENCE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/ops_test)
    add_subdirectory(tests/nn_test)
    if (INFERENCE_BUILD_TTS)
        add_subdirectory(tests/model_config_test)
    endif()
    if (INFERENCE_BUILD_ASR)
        add_subdirectory(tests/asr_test)
    endif()
    if (INFERENCE_BUILD_LLM)
        add_subdirectory(tests/llm_test)
    endif()
    if (INFERENCE_BUILD_VISUAL_GENERATION)
        add_subdirectory(tests/visual_test)
    endif()

    find_package(Python3 QUIET COMPONENTS Interpreter)
    if (Python3_Interpreter_FOUND)
        foreach(script_test IN ITEMS artifact quantization model_schema maintenance_updates)
            add_test(
                NAME scripts.${script_test}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_${script_test}.py
            )
            set_tests_properties(scripts.${script_test} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                LABELS "scripts;unit"
            )
        endforeach()
        if (INFERENCE_BUILD_TTS)
            add_test(
                NAME scripts.gpt_sovits_provider
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/scripts/test_gpt_sovits_provider.py
            )
            set_tests_properties(scripts.gpt_sovits_provider PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                LABELS "scripts;tts;provider;unit"
            )
        endif()
    endif()
endif()
