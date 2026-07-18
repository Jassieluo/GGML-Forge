add_library(gpt_sovits_frontend STATIC
    text_utils.cpp
    gpt_sovits_frontend.cpp
    phonemizer.cpp
)

target_include_directories(gpt_sovits_frontend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/categories/tts
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/categories/tts
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/src/categories/tts/frontends/gpt_sovits/vendor/cppjieba/include
)

target_link_libraries(gpt_sovits_frontend PUBLIC ggml gpt_sovits_models)
