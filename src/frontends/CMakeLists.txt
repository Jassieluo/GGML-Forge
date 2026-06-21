# src/frontends/CMakeLists.txt

add_library(tts_frontends STATIC
    gpt_sovits/text_utils.cpp
    gpt_sovits/gpt_sovits_frontend.cpp
    gpt_sovits/phonemizer.cpp
)

target_include_directories(tts_frontends PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cpp-jieba/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cpp-jieba/deps/limonp/include
)

target_link_libraries(tts_frontends PUBLIC ggml tts_models)
