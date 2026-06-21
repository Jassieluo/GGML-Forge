# src/ops/sycl/CMakeLists.txt

add_library(tts_ops_sycl STATIC
    ops_sycl.cpp
)

target_include_directories(tts_ops_sycl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
)

target_link_libraries(tts_ops_sycl PUBLIC ggml)
target_compile_definitions(tts_ops_sycl PRIVATE GGML_USE_SYCL)

set_source_files_properties(ops_sycl.cpp PROPERTIES COMPILE_FLAGS "-fsycl")
