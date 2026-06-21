# src/ops/cuda/CMakeLists.txt

add_library(tts_ops_cuda STATIC
    ops_cuda.cu
)

target_include_directories(tts_ops_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
)

target_link_libraries(tts_ops_cuda PUBLIC ggml)

# Link cuDNN
find_package(CUDAToolkit REQUIRED)
find_library(CUDNN_LIBRARY NAMES cudnn libcudnn PATHS ${CUDNN_ROOT_DIR}/lib ${CUDNN_ROOT_DIR}/lib/x64)
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h PATHS ${CUDNN_ROOT_DIR}/include)
if (CUDNN_LIBRARY AND CUDNN_INCLUDE_DIR)
    target_include_directories(tts_ops_cuda PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(tts_ops_cuda PRIVATE ${CUDNN_LIBRARY})
    target_compile_definitions(tts_ops_cuda PRIVATE GGML_USE_CUDA)
else()
    message(FATAL_ERROR "cuDNN not found but GGML_CUDA is enabled. Please verify CUDNN_ROOT_DIR.")
endif()
