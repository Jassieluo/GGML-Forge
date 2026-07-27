add_library(gpt_sovits_models STATIC
    bert/bert.cpp
    hubert/hubert.cpp
    speaker_encoder/eres2net_v2.cpp
    t2s/gpt_t2s.cpp
    vits/vits.cpp
    vits/encoder.cpp
    vits/conditioning.cpp
    vits/cfm_modules.cpp
    vits/generator.cpp
    vits/vits_modules.cpp
    vits/vits_classic.cpp
    vits/vits_cfm.cpp
)

target_include_directories(gpt_sovits_models PUBLIC
    ${CMAKE_SOURCE_DIR}/src/categories/tts
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/categories/tts
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(gpt_sovits_models PUBLIC ggml ggml_nn_ext)
