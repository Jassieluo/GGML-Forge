# tests/ops_test/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(test_ops C CXX)

add_executable(test_ops test_ops.cpp)
add_executable(test_new_ops test_new_ops.cpp)

# Include directories
target_include_directories(test_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_new_ops PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link to ggml, core ops library, and backend extensions
target_link_libraries(test_ops PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_new_ops PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
if (GGML_CUDA)
    target_link_libraries(test_ops PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ops PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_new_ops PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_new_ops PRIVATE GGML_USE_CUDA)
endif()
if (GGML_SYCL)
    target_link_libraries(test_ops PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ops PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_new_ops PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_new_ops PRIVATE GGML_USE_SYCL)
endif()

if (MSVC)
    target_compile_options(test_ops PRIVATE /EHsc /utf-8)
    target_compile_options(test_new_ops PRIVATE /EHsc /utf-8)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(test_ops PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_new_ops PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(test_ops PRIVATE "-fsycl")
        target_link_options(test_ops PRIVATE "-fsycl")
        target_compile_options(test_new_ops PRIVATE "-fsycl")
        target_link_options(test_new_ops PRIVATE "-fsycl")
    endif()
endif()
