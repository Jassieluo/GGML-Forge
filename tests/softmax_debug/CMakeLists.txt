# tests/softmax_debug/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(test_softmax C CXX)

add_executable(test_softmax test_softmax.cpp)

# Include directories
target_include_directories(test_softmax PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link to ggml and ggml_ops_ext
target_link_libraries(test_softmax PRIVATE ggml ggml_ops_ext)

add_dependencies(test_softmax ggml_ops_ext_cpu)
if (GGML_CUDA)
    add_dependencies(test_softmax ggml_ops_ext_cuda)
endif()
if (GGML_SYCL)
    add_dependencies(test_softmax ggml_ops_ext_sycl)
endif()

if (MSVC)
    target_compile_options(test_softmax PRIVATE /EHsc /utf-8)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(test_softmax PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(test_softmax PRIVATE "-fsycl")
        target_link_options(test_softmax PRIVATE "-fsycl")
    endif()
endif()

