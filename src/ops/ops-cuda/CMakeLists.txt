# src/ops/cuda/CMakeLists.txt

add_library(ggml_ops_ext_cuda
    ops_cuda.cu
    conv_1d.cu
    conv_transpose_1d.cu
    mish.cu
    gated_tanh_sigmoid.cu
    layer_norm.cu
    double_swish.cu
)

find_package(CUDAToolkit REQUIRED)

target_include_directories(ggml_ops_ext_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
)

target_link_libraries(ggml_ops_ext_cuda PUBLIC ggml ggml_ops_ext CUDA::cublas)

# Link cuDNN (Optional)
option(GGML_CUDNN "Use cuDNN for CUDA operators if available" ON)

if (GGML_CUDNN)
    find_package(CUDAToolkit REQUIRED)
    find_library(CUDNN_LIBRARY NAMES cudnn libcudnn PATHS ${CUDNN_ROOT_DIR}/lib ${CUDNN_ROOT_DIR}/lib/x64)
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS ${CUDNN_ROOT_DIR}/include)
    if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
        message(STATUS "cuDNN found: enabling cuDNN path for conv operators.")
        target_include_directories(ggml_ops_ext_cuda PRIVATE ${CUDNN_INCLUDE_DIR})
        target_link_libraries(ggml_ops_ext_cuda PRIVATE ${CUDNN_LIBRARY})
        target_compile_definitions(ggml_ops_ext_cuda PRIVATE GGML_USE_CUDNN)
    else()
        message(WARNING "cuDNN not found. Falling back to native cuBLAS/hand-written kernels.")
    endif()
else()
    message(STATUS "cuDNN is manually disabled. Using native cuBLAS/hand-written kernels.")
endif()

target_compile_definitions(ggml_ops_ext_cuda PRIVATE GGML_USE_CUDA)

# Export all symbols on Windows for dynamic DLL linking
set_target_properties(ggml_ops_ext_cuda PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
