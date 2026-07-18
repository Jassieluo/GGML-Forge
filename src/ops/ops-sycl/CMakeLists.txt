# src/ops/sycl/CMakeLists.txt

add_library(ggml_ops_ext_sycl
    ops_sycl.cpp
    conv_1d.cpp
    conv_transpose_1d.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
    attention.cpp
    kv_cache.cpp
    glu.cpp
    relative_pe_keys.cpp
    relative_pe_values.cpp
    instance_norm.cpp
    snake.cpp
    alias_free_activation.cpp
)

target_include_directories(ggml_ops_ext_sycl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src/ggml-sycl
)

target_link_libraries(ggml_ops_ext_sycl PUBLIC ggml ggml_ops_ext ${CMAKE_DL_LIBS})
find_package(MKL REQUIRED)
target_link_libraries(ggml_ops_ext_sycl PRIVATE MKL::MKL_SYCL::BLAS)
target_compile_definitions(ggml_ops_ext_sycl PRIVATE GGML_USE_SYCL GGML_SYCL_WARP_SIZE=32)

set_source_files_properties(
    ops_sycl.cpp
    conv_1d.cpp
    conv_transpose_1d.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
    attention.cpp
    kv_cache.cpp
    snake.cpp
    alias_free_activation.cpp
    PROPERTIES COMPILE_FLAGS "-fsycl"
)

set_target_properties(ggml_ops_ext_sycl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
