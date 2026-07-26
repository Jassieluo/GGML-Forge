# src/ops/sycl/CMakeLists.txt

add_library(ggml_ops_ext_sycl
    ops_sycl.cpp
    conv_1d.cpp
    conv_transpose_1d.cpp
    conv_nd.cpp
    pool_nd.cpp
    pad_nd.cpp
    adaptive_pool_nd.cpp
    resize_nd.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
    attention.cpp
    kv_cache.cpp
    glu.cpp
    gated_activation.cpp
    relative_pe_keys.cpp
    relative_pe_values.cpp
    instance_norm.cpp
    snake.cpp
    alias_free_activation.cpp
    fused_norm_act.cpp
    pos_encoding.cpp
    sample_dist.cpp
    stft.cpp
    length_regulate.cpp
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

if (GGML_SYCL_DEVICE_ARCH)
    message(STATUS
        "Forge SYCL AOT target: spir64_gen/${GGML_SYCL_DEVICE_ARCH}")
    target_compile_options(ggml_ops_ext_sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
    )
    target_link_options(ggml_ops_ext_sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
        "-fsycl-device-code-split=${FORGE_SYCL_DEVICE_CODE_SPLIT}"
    )
    if (FORGE_SYCL_AOT_LINK_JOBS GREATER 0)
        target_link_options(ggml_ops_ext_sycl PRIVATE
            "-fsycl-max-parallel-link-jobs=${FORGE_SYCL_AOT_LINK_JOBS}")
    endif()
else()
    message(STATUS "Forge SYCL JIT build enabled for rapid iteration.")
endif()

set_source_files_properties(
    ops_sycl.cpp
    conv_1d.cpp
    conv_transpose_1d.cpp
    conv_nd.cpp
    pool_nd.cpp
    pad_nd.cpp
    adaptive_pool_nd.cpp
    resize_nd.cpp
    mish.cpp
    gated_tanh_sigmoid.cpp
    layer_norm.cpp
    double_swish.cpp
    attention.cpp
    kv_cache.cpp
    snake.cpp
    alias_free_activation.cpp
    gated_activation.cpp
    glu.cpp
    relative_pe_keys.cpp
    relative_pe_values.cpp
    instance_norm.cpp
    fused_norm_act.cpp
    pos_encoding.cpp
    sample_dist.cpp
    stft.cpp
    length_regulate.cpp
    PROPERTIES COMPILE_FLAGS "-fsycl"
)

set_target_properties(ggml_ops_ext_sycl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
