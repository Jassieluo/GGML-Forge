# src/CMakeLists.txt

add_subdirectory(ops)
add_subdirectory(nn)
if (INFERENCE_BUILD_TTS)
    add_subdirectory(categories/tts)
endif()
