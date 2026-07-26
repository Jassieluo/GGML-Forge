# src/ops/cuda/CMakeLists.txt

add_library(ggml_ops_ext_cuda
    ops_cuda.cu
    conv_1d.cu
    conv_transpose_1d.cu
    conv_nd.cu
    pool_nd.cu
    pad_nd.cu
    adaptive_pool_nd.cu
    resize_nd.cu
    mish.cu
    gated_tanh_sigmoid.cu
    layer_norm.cu
    double_swish.cu
    attention.cu
    kv_cache.cu
    glu.cu
    gated_activation.cu
    alias_free_activation.cu
    relative_pe_keys.cu
    relative_pe_values.cu
    instance_norm.cu
    snake.cu
    fused_norm_act.cu
    pos_encoding.cu
    sample_dist.cu
    stft.cu
    length_regulate.cu
    recurrent.cu
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(ggml_ops_ext_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
)

target_link_libraries(ggml_ops_ext_cuda PUBLIC ggml ggml_ops_ext CUDA::cublas CUDA::cudart ${CMAKE_DL_LIBS})

target_compile_definitions(ggml_ops_ext_cuda PRIVATE GGML_USE_CUDA)

target_compile_definitions(ggml_ops_ext_cuda PRIVATE GGML_OPS_EXT_CUDA_SHARED)
