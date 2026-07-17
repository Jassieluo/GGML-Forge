# src/CMakeLists.txt

add_subdirectory(models)
add_subdirectory(ops)
add_subdirectory(nn)
add_subdirectory(frontends)

add_library(tts SHARED
    pipelines/gpt_sovits/gpt_sovits_pipeline.cpp
    pipelines/gpt_sovits/dsp.cpp
    pipelines/gpt_sovits/voice_manager.cpp
    model_config.cpp
    tts_core.cpp
    tts_frontend.cpp
)

target_compile_definitions(tts PRIVATE TTS_BUILD_SHARED GPT_SOVITS_BUILD_SHARED)
target_compile_definitions(tts PUBLIC TTS_USE_SHARED GPT_SOVITS_USE_SHARED)
set_target_properties(tts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/frontends/gpt_sovits>
)

target_link_libraries(tts PRIVATE
    ggml
    tts_models
    ggml_ops_ext
    ggml_ops_ext_cpu
    ggml_nn_ext
    tts_frontends
)

if (GGML_CUDA)
    target_link_libraries(tts PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(tts PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    target_link_libraries(tts PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(tts PRIVATE GGML_USE_SYCL)
endif()

if (GGML_BLAS)
    target_compile_definitions(tts PRIVATE GGML_USE_BLAS)
endif()

if (MSVC)
    target_compile_options(tts PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(tts PRIVATE -Wall -Wextra)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(tts PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(tts PRIVATE "-fsycl")
        target_link_options(tts PRIVATE "-fsycl")
    endif()
endif()
