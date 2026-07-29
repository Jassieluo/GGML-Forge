# src/ops/CMakeLists.txt

add_library(ggml_ops_ext
    ops.cpp
    core_ops/ops_conv_1d.cpp
    core_ops/ops_conv_transpose_1d.cpp
    core_ops/ops_conv_nd.cpp
    core_ops/ops_pool_nd.cpp
    core_ops/ops_pad_nd.cpp
    core_ops/ops_adaptive_pool_nd.cpp
    core_ops/ops_resize_nd.cpp
    core_ops/ops_foundation.cpp
    core_ops/ops_mish.cpp
    core_ops/ops_gated_tanh_sigmoid.cpp
    core_ops/ops_layer_norm.cpp
    core_ops/ops_double_swish.cpp
    core_ops/ops_embedding.cpp
    core_ops/ops_attention.cpp
    core_ops/ops_kv_cache.cpp
    core_ops/ops_glu.cpp
    core_ops/ops_relative_pe_keys.cpp
    core_ops/ops_relative_pe_values.cpp
    core_ops/ops_instance_norm.cpp
    core_ops/ops_snake.cpp
    core_ops/ops_snake_beta.cpp
    core_ops/ops_alias_free_activation.cpp
    core_ops/ops_ada_ln.cpp
    core_ops/ops_fused_norm_act.cpp
    core_ops/ops_pos_encoding.cpp
    core_ops/ops_sample.cpp
    core_ops/ops_stft.cpp
    core_ops/ops_complex_abs.cpp
    core_ops/ops_length_regulate.cpp
    core_ops/ops_recurrent.cpp
    core_ops/ops_reduce_nd.cpp
    core_ops/ops_selection.cpp
    core_ops/ops_grid_sample.cpp
)

target_include_directories(ggml_ops_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src  # For ggml-backend-impl.h
)

target_link_libraries(ggml_ops_ext PUBLIC ggml)

# Export all symbols on Windows for dynamic DLL linking
set_target_properties(ggml_ops_ext PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Add CPU Backend Subdirectory
add_subdirectory(ops-cpu)

if (GGML_CUDA)
    add_subdirectory(ops-cuda)
    target_compile_definitions(ggml_ops_ext PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    add_subdirectory(ops-sycl)
    target_compile_definitions(ggml_ops_ext PRIVATE GGML_USE_SYCL)
endif()
