# CMakeLists.txt for GGML-Forge

cmake_minimum_required(VERSION 3.14)
project(ggml_forge C CXX ASM)

if (GGML_BACKEND_DL)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries for dynamic backends" FORCE)
endif()

option(INFERENCE_BUILD_TTS "Build the TTS category" ON)
option(INFERENCE_BUILD_LLM "Build the LLM category with llama.cpp" ON)
option(INFERENCE_BUILD_ASR "Build the ASR category with whisper.cpp" ON)
option(INFERENCE_BUILD_VISUAL_GENERATION "Build visual generation with stable-diffusion.cpp" ON)
option(INFERENCE_BUILD_OBJECT_DETECTION "Build the object detection category" ON)
option(INFERENCE_BUILD_DEPTH_ESTIMATION "Build the depth estimation category" ON)
option(INFERENCE_BUILD_SEMANTIC_SEGMENTATION "Build the semantic segmentation category" ON)
option(INFERENCE_BUILD_UI "Build the EUI-NEO UI framework" ON)
option(INFERENCE_BUILD_EXAMPLES "Build category examples" ON)
option(INFERENCE_BUILD_SERVER "Build the provider-neutral forge-server tool" OFF)
option(INFERENCE_BUILD_TESTS "Build tests" ON)
set(EUI_RENDER_BACKEND "vulkan" CACHE STRING
    "EUI render backend: auto, opengl, or vulkan")
set_property(CACHE EUI_RENDER_BACKEND PROPERTY STRINGS auto opengl vulkan)
if (NOT EUI_RENDER_BACKEND MATCHES "^(auto|opengl|vulkan)$")
    message(FATAL_ERROR
        "Invalid EUI_RENDER_BACKEND=${EUI_RENDER_BACKEND}; expected auto, opengl, or vulkan")
endif()
set(FORGE_SYCL_AOT_LINK_JOBS "0" CACHE STRING
    "Maximum parallel Intel SYCL AOT device-link jobs (0 uses compiler default)")
set(FORGE_SYCL_DEVICE_CODE_SPLIT "auto" CACHE STRING
    "Intel SYCL device-code split mode: auto, off, per_kernel, or per_source")
set_property(CACHE FORGE_SYCL_DEVICE_CODE_SPLIT PROPERTY STRINGS
    auto off per_kernel per_source)
if (NOT FORGE_SYCL_DEVICE_CODE_SPLIT MATCHES "^(auto|off|per_kernel|per_source)$")
    message(FATAL_ERROR
        "Invalid FORGE_SYCL_DEVICE_CODE_SPLIT=${FORGE_SYCL_DEVICE_CODE_SPLIT}")
endif()

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

if (GGML_SYCL AND GGML_SYCL_DEVICE_ARCH AND TARGET ggml-sycl AND
    FORGE_SYCL_AOT_LINK_JOBS GREATER 0)
    target_link_options(ggml-sycl PRIVATE
        "-fsycl-max-parallel-link-jobs=${FORGE_SYCL_AOT_LINK_JOBS}")
endif()
if (GGML_SYCL AND GGML_SYCL_DEVICE_ARCH AND TARGET ggml-sycl)
    target_link_options(ggml-sycl PRIVATE
        "-fsycl-device-code-split=${FORGE_SYCL_DEVICE_CODE_SPLIT}")
endif()

# ggml-sycl.dll links against the Intel oneAPI runtime. Copy those DLLs next
# to the binaries so SYCL devices are available without an oneAPI shell
# (missing files are skipped; ze_loader.dll ships with the GPU driver).
if (GGML_SYCL AND WIN32 AND DEFINED ENV{ONEAPI_ROOT})
    set(FORGE_ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
    set(FORGE_SYCL_RUNTIME_DLLS
        "compiler/latest/bin/sycl8.dll"
        "compiler/latest/bin/sycl-jit.dll"
        "compiler/latest/bin/ur_loader.dll"
        "compiler/latest/bin/ur_win_proxy_loader.dll"
        "compiler/latest/bin/ur_adapter_level_zero.dll"
        "compiler/latest/bin/ur_adapter_level_zero_v2.dll"
        "compiler/latest/bin/ur_adapter_opencl.dll"
        "compiler/latest/bin/svml_dispmd.dll"
        "compiler/latest/bin/libmmd.dll"
        "mkl/latest/bin/mkl_sycl_blas.5.dll"
        "mkl/latest/bin/mkl_core.2.dll"
        "mkl/latest/bin/mkl_tbb_thread.2.dll"
        "dnnl/latest/bin/dnnl.dll"
        "tbb/latest/bin/tbb12.dll"
    )
    file(MAKE_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    foreach(dll_path IN LISTS FORGE_SYCL_RUNTIME_DLLS)
        get_filename_component(dll_name "${dll_path}" NAME)
        if (EXISTS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${dll_name}")
            # Already deployed — don't re-copy on every configure.
        elseif (EXISTS "${FORGE_ONEAPI_ROOT}/${dll_path}")
            file(COPY "${FORGE_ONEAPI_ROOT}/${dll_path}"
                 DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        else()
            message(STATUS "SYCL runtime DLL not found (skipped): ${dll_path}")
        endif()
    endforeach()
endif()

add_subdirectory(src)

if (INFERENCE_BUILD_UI)
    add_subdirectory(ui)
endif()

if (INFERENCE_BUILD_SERVER)
    add_subdirectory(tools/server)
endif()

if (INFERENCE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (INFERENCE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/media_test)
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
    if (INFERENCE_BUILD_OBJECT_DETECTION)
        add_subdirectory(tests/object_detection_test)
    endif()
    if (INFERENCE_BUILD_DEPTH_ESTIMATION)
        add_subdirectory(tests/depth_estimation_test)
    endif()
    if (INFERENCE_BUILD_SEMANTIC_SEGMENTATION)
        add_subdirectory(tests/semantic_segmentation_test)
    endif()
    if (INFERENCE_BUILD_SERVER)
        add_subdirectory(tests/server_test)
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
