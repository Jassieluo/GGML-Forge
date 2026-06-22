# tests/ops_test/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(test_ops C CXX)

add_executable(test_ops test_ops.cpp)

# Include directories
target_include_directories(test_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link to ggml, core ops library
target_link_libraries(test_ops PRIVATE ggml ggml_ops_ext)

# Add dependencies on backend DLL targets so they are compiled before running
add_dependencies(test_ops ggml_ops_ext_cpu)
if (GGML_CUDA)
    add_dependencies(test_ops ggml_ops_ext_cuda)
endif()
if (GGML_SYCL)
    add_dependencies(test_ops ggml_ops_ext_sycl)
endif()

if (MSVC)
    target_compile_options(test_ops PRIVATE /EHsc /utf-8)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(test_ops PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(test_ops PRIVATE "-fsycl")
        target_link_options(test_ops PRIVATE "-fsycl")
    endif()
endif()
