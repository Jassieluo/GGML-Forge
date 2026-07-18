add_library(tts_models STATIC
    model.cpp
    text/bert/bert.cpp
    ssl/hubert/hubert.cpp
    lm/gpt_t2s/gpt_t2s.cpp
    synthesis/vits/vits.cpp
    synthesis/vits/encoder.cpp
    synthesis/vits/conditioning.cpp
    synthesis/vits/cfm_modules.cpp
    synthesis/vits/generator.cpp
    synthesis/vits/vits_modules.cpp
    synthesis/vits/vits_classic.cpp
    synthesis/vits/vits_cfm.cpp
)

target_include_directories(tts_models PUBLIC
    ${CMAKE_SOURCE_DIR}/src/categories/tts
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/categories/tts
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(tts_models PUBLIC ggml ggml_nn_ext)

add_executable(nn-model-schema model_schema_cli.cpp)
target_link_libraries(nn-model-schema PRIVATE tts_models)

if (MSVC)
    target_compile_options(nn-model-schema PRIVATE /EHsc /utf-8)
else()
    target_compile_options(nn-model-schema PRIVATE -Wall -Wextra)
endif()
