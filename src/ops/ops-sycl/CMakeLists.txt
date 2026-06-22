# src/ops/sycl/CMakeLists.txt

add_library(ggml_ops_ext_sycl
    ops_sycl.cpp
    conv_1d_builder.cpp
    conv_transpose_1d.cpp
    softmax.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
)

target_include_directories(ggml_ops_ext_sycl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src/ggml-sycl
)

target_link_libraries(ggml_ops_ext_sycl PUBLIC ggml ggml_ops_ext)
target_compile_definitions(ggml_ops_ext_sycl PRIVATE GGML_USE_SYCL GGML_SYCL_WARP_SIZE=32)

set_source_files_properties(
    ops_sycl.cpp
    conv_1d_builder.cpp
    conv_transpose_1d.cpp
    softmax.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
    PROPERTIES COMPILE_FLAGS "-fsycl"
)

# Export all symbols on Windows for dynamic DLL linking
set_target_properties(ggml_ops_ext_sycl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
