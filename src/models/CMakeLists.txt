# src/models/CMakeLists.txt

add_library(tts_models STATIC
    model.cpp
    model_loader.cpp
    text/bert/bert.cpp
    ssl/hubert/hubert.cpp
    lm/gpt_t2s/gpt_t2s.cpp
    synthesis/vits/vits.cpp
)

target_include_directories(tts_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(tts_models PUBLIC ggml ggml_nn_ext)
