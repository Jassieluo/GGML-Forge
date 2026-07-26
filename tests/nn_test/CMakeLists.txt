add_executable(test_nn_context test_nn_context.cpp)
add_executable(test_nn_loading test_nn_loading.cpp)
add_executable(test_nn_kv_cache test_nn_kv_cache.cpp)

target_include_directories(test_nn_context PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_nn_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_nn_kv_cache PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(test_nn_context PRIVATE ggml_nn_ext ggml)
target_link_libraries(test_nn_loading PRIVATE ggml_nn_ext ggml)
target_link_libraries(test_nn_kv_cache PRIVATE ggml_nn_ext ggml_ops_ext ggml_ops_ext_cpu ggml)

add_test(NAME nn.context COMMAND test_nn_context)
add_test(NAME nn.loading COMMAND test_nn_loading)
add_test(NAME nn.kv_cache COMMAND test_nn_kv_cache)
set_tests_properties(nn.context nn.loading nn.kv_cache PROPERTIES LABELS "nn;unit")

if (INFERENCE_BUILD_TTS)
    foreach(model IN ITEMS t2s bert hubert vits)
        add_executable(test_${model}_loading test_${model}_loading.cpp)
        target_include_directories(test_${model}_loading PRIVATE
            ${CMAKE_SOURCE_DIR}/src/categories/tts
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/ggml/include
        )
        target_link_libraries(test_${model}_loading PRIVATE gpt_sovits_models ggml_nn_ext ggml)
    endforeach()
    add_executable(test_speaker_encoder
        test_speaker_encoder.cpp
        ${CMAKE_SOURCE_DIR}/src/categories/tts/providers/gpt_sovits/dsp.cpp)
    target_include_directories(test_speaker_encoder PRIVATE
        ${CMAKE_SOURCE_DIR}/src/categories/tts
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/examples)
    target_link_libraries(test_speaker_encoder PRIVATE
        gpt_sovits_models ggml_nn_ext ggml_ops_ext ggml)
    foreach(model IN ITEMS t2s bert hubert)
        add_test(NAME tts.${model}_loading COMMAND test_${model}_loading)
        set_tests_properties(tts.${model}_loading PROPERTIES LABELS "tts;provider;unit")
    endforeach()
    # test_vits_loading and test_speaker_encoder require real model weights
    # (plus reference audio/embedding fixtures) passed on the command line, so
    # they are built but intentionally not registered with CTest. Run manually:
    #   test_vits_loading <vits.gguf>
    #   test_speaker_encoder <model.gguf> <reference.wav> <expected.sv.bin> [expected.fbank.bin]
endif()

if (MSVC)
    target_compile_options(test_nn_context PRIVATE /EHsc /utf-8)
    target_compile_options(test_nn_loading PRIVATE /EHsc /utf-8)
    target_compile_options(test_nn_kv_cache PRIVATE /EHsc /utf-8)
    if (INFERENCE_BUILD_TTS)
        foreach(model IN ITEMS t2s bert hubert vits)
            target_compile_options(test_${model}_loading PRIVATE /EHsc /utf-8)
        endforeach()
        target_compile_options(test_speaker_encoder PRIVATE /EHsc /utf-8)
    endif()
else()
    target_compile_options(test_nn_context PRIVATE -Wall -Wextra)
    target_compile_options(test_nn_loading PRIVATE -Wall -Wextra)
    target_compile_options(test_nn_kv_cache PRIVATE -Wall -Wextra)
    if (INFERENCE_BUILD_TTS)
        foreach(model IN ITEMS t2s bert hubert vits)
            target_compile_options(test_${model}_loading PRIVATE -Wall -Wextra)
        endforeach()
        target_compile_options(test_speaker_encoder PRIVATE -Wall -Wextra)
    endif()
endif()
