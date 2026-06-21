# src/ops/CMakeLists.txt

add_library(tts_ops STATIC
    ops.cpp
)

target_include_directories(tts_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src  # For ggml-backend-impl.h
)

target_link_libraries(tts_ops PUBLIC ggml)

if (GGML_CUDA)
    add_subdirectory(ops-cuda)
    target_link_libraries(tts_ops PRIVATE tts_ops_cuda)
    target_compile_definitions(tts_ops PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    add_subdirectory(ops-sycl)
    target_link_libraries(tts_ops PRIVATE tts_ops_sycl)
    target_compile_definitions(tts_ops PRIVATE GGML_USE_SYCL)
endif()
