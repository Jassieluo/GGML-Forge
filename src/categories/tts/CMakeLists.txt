add_subdirectory(providers/gpt_sovits/models)
add_subdirectory(providers/gpt_sovits/frontend)

add_library(tts SHARED
    providers/gpt_sovits/provider.cpp
    providers/gpt_sovits/gpt_sovits_provider.cpp
    providers/gpt_sovits/dsp.cpp
    providers/gpt_sovits/voice_manager.cpp
    model_config.cpp
    tts_core.cpp
)

target_compile_definitions(tts PRIVATE TTS_BUILD_SHARED GPT_SOVITS_BUILD_SHARED)
target_compile_definitions(tts PUBLIC TTS_USE_SHARED GPT_SOVITS_USE_SHARED)
set_target_properties(tts PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)
target_include_directories(tts PRIVATE
    ${CMAKE_SOURCE_DIR}/include/categories/tts
    ${CMAKE_SOURCE_DIR}/src/categories/tts
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/gpt_sovits/frontend
)

target_link_libraries(tts PRIVATE
    ggml
    gpt_sovits_models
    ggml_ops_ext
    ggml_ops_ext_cpu
    ggml_nn_ext
    gpt_sovits_frontend
)

if (GGML_CUDA)
    target_link_libraries(tts PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(tts PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    target_link_libraries(tts PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(tts PRIVATE GGML_USE_SYCL)
endif()

if (GGML_BLAS)
    target_compile_definitions(tts PRIVATE GGML_USE_BLAS)
endif()

if (MSVC)
    target_compile_options(tts PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(tts PRIVATE -Wall -Wextra)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(tts PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(tts PRIVATE "-fsycl")
        target_link_options(tts PRIVATE "-fsycl")
    endif()
endif()
