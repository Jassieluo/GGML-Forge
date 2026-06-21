# src/models/CMakeLists.txt

add_library(tts_models STATIC
    model.cpp
    # Future model architectures source files will be appended here
)

target_include_directories(tts_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(tts_models PUBLIC ggml)
