add_executable(test_nn_context test_nn_context.cpp)
add_executable(test_nn_loading test_nn_loading.cpp)
add_executable(test_t2s_loading test_t2s_loading.cpp)
add_executable(test_bert_loading test_bert_loading.cpp)
add_executable(test_hubert_loading test_hubert_loading.cpp)
add_executable(test_vits_loading test_vits_loading.cpp)

target_include_directories(test_nn_context PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_nn_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_t2s_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_bert_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_hubert_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_vits_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)

target_link_libraries(test_nn_context PRIVATE ggml_nn_ext ggml)
target_link_libraries(test_nn_loading PRIVATE ggml_nn_ext ggml)
target_link_libraries(test_t2s_loading PRIVATE tts_models ggml_nn_ext ggml)
target_link_libraries(test_bert_loading PRIVATE tts_models ggml_nn_ext ggml)
target_link_libraries(test_hubert_loading PRIVATE tts_models ggml_nn_ext ggml)
target_link_libraries(test_vits_loading PRIVATE tts_models ggml_nn_ext ggml)

if (MSVC)
    target_compile_options(test_nn_context PRIVATE /EHsc /utf-8)
    target_compile_options(test_nn_loading PRIVATE /EHsc /utf-8)
    target_compile_options(test_t2s_loading PRIVATE /EHsc /utf-8)
    target_compile_options(test_bert_loading PRIVATE /EHsc /utf-8)
    target_compile_options(test_hubert_loading PRIVATE /EHsc /utf-8)
    target_compile_options(test_vits_loading PRIVATE /EHsc /utf-8)
else()
    target_compile_options(test_nn_context PRIVATE -Wall -Wextra)
    target_compile_options(test_nn_loading PRIVATE -Wall -Wextra)
    target_compile_options(test_t2s_loading PRIVATE -Wall -Wextra)
    target_compile_options(test_bert_loading PRIVATE -Wall -Wextra)
    target_compile_options(test_hubert_loading PRIVATE -Wall -Wextra)
    target_compile_options(test_vits_loading PRIVATE -Wall -Wextra)
endif()
