# src/CMakeLists.txt

add_subdirectory(models)
add_subdirectory(ops)
add_subdirectory(frontends)

add_library(tts SHARED
    pipelines/gpt_sovits/gpt_sovits_pipeline.cpp
    pipelines/gpt_sovits/dsp.cpp
    pipelines/gpt_sovits/voice_manager.cpp
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
    tts_ops
    tts_frontends
)

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
