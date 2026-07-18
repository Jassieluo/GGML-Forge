add_executable(test_nn_context test_nn_context.cpp)
add_executable(test_nn_loading test_nn_loading.cpp)

target_include_directories(test_nn_context PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_include_directories(test_nn_loading PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(test_nn_context PRIVATE ggml_nn_ext ggml)
target_link_libraries(test_nn_loading PRIVATE ggml_nn_ext ggml)

if (INFERENCE_BUILD_TTS)
    foreach(model IN ITEMS t2s bert hubert vits)
        add_executable(test_${model}_loading test_${model}_loading.cpp)
        target_include_directories(test_${model}_loading PRIVATE
            ${CMAKE_SOURCE_DIR}/src/categories/tts
            ${CMAKE_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/ggml/include
        )
        target_link_libraries(test_${model}_loading PRIVATE tts_models ggml_nn_ext ggml)
    endforeach()
endif()

if (MSVC)
    target_compile_options(test_nn_context PRIVATE /EHsc /utf-8)
    target_compile_options(test_nn_loading PRIVATE /EHsc /utf-8)
    if (INFERENCE_BUILD_TTS)
        foreach(model IN ITEMS t2s bert hubert vits)
            target_compile_options(test_${model}_loading PRIVATE /EHsc /utf-8)
        endforeach()
    endif()
else()
    target_compile_options(test_nn_context PRIVATE -Wall -Wextra)
    target_compile_options(test_nn_loading PRIVATE -Wall -Wextra)
    if (INFERENCE_BUILD_TTS)
        foreach(model IN ITEMS t2s bert hubert vits)
            target_compile_options(test_${model}_loading PRIVATE -Wall -Wextra)
        endforeach()
    endif()
endif()
