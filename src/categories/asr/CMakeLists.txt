set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
set(WHISPER_COREML OFF CACHE BOOL "" FORCE)
set(WHISPER_OPENVINO OFF CACHE BOOL "" FORCE)
set(WHISPER_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)

if (NOT TARGET ggml)
    message(FATAL_ERROR "ASR requires the project root ggml target")
endif()

# Upstream explicitly skips its bundled ggml when the parent already provides
# a target named 'ggml'. The root llama.cpp-derived GGML remains authoritative.
add_subdirectory(providers/whisper_cpp/whisper.cpp whisper_cpp_upstream EXCLUDE_FROM_ALL)

add_library(asr SHARED
    asr_core.cpp
    providers/whisper_cpp/provider.cpp
)
target_compile_definitions(asr PRIVATE ASR_BUILD_SHARED)
target_compile_definitions(asr PUBLIC ASR_USE_SHARED)
set_target_properties(asr PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(asr PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(asr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/whisper_cpp/whisper.cpp/include
)
target_link_libraries(asr PRIVATE whisper ggml forge_audio_dsp)

if (MSVC)
    target_compile_options(asr PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(asr PRIVATE -Wall -Wextra)
endif()
