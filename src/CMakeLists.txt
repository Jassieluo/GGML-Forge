# src/CMakeLists.txt

add_subdirectory(models)

add_library(tts SHARED
    tts_core.cpp
    tts_frontend.cpp
)

target_compile_definitions(tts PRIVATE TTS_BUILD_SHARED)
target_compile_definitions(tts PUBLIC TTS_USE_SHARED)
set_target_properties(tts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include>
)

target_link_libraries(tts PRIVATE
    ggml
    phonemizer
    tts_models
)

if (MSVC)
    target_compile_options(tts PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(tts PRIVATE -Wall -Wextra)
endif()
