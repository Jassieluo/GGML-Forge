# tests/ops_test/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(test_ops C CXX)

add_executable(test_ops test_ops.cpp)
add_executable(test_new_ops test_new_ops.cpp)
add_executable(test_snake_op test_snake_op.cpp)
add_executable(test_ada_ln_op test_ada_ln_op.cpp)
add_executable(test_debug_conv test_debug_conv.cpp)
add_executable(test_ops_architecture test_ops_architecture.cpp)
add_executable(test_ops_concurrency test_ops_concurrency.cpp)
add_executable(test_ops_quantized_conv test_ops_quantized_conv.cpp)
add_executable(test_attention_streaming test_attention_streaming.cpp)
add_executable(benchmark_attention benchmark_attention.cpp)

foreach(test_name IN ITEMS
    test_ops_architecture
    test_ops_concurrency
    test_attention_streaming
)
    add_test(NAME ops.${test_name} COMMAND ${test_name})
    set_tests_properties(ops.${test_name} PROPERTIES LABELS "ops;unit")
endforeach()

if (WIN32 AND GGML_SYCL)
    # SYCL/MKL runtime DLLs are supplied by the configured oneAPI toolchain.
    # Preserve that configure-time PATH when CTest is launched from a plain shell.
    string(REPLACE ";" "\\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    foreach(test_name IN ITEMS
        test_ops_architecture
        test_ops_concurrency
        test_attention_streaming
    )
        set_tests_properties(ops.${test_name} PROPERTIES
            ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
        )
    endforeach()
endif()

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
target_include_directories(test_snake_op PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_ada_ln_op PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_debug_conv PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_ops_architecture PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_ops_concurrency PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_ops_quantized_conv PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(test_attention_streaming PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(benchmark_attention PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src
    ${CMAKE_SOURCE_DIR}/include
)

# Link to ggml, core ops library, and backend extensions
target_link_libraries(test_ops PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_new_ops PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_snake_op PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_ada_ln_op PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_debug_conv PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_ops_architecture PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_ops_concurrency PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_ops_quantized_conv PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(test_attention_streaming PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
target_link_libraries(benchmark_attention PRIVATE ggml ggml_ops_ext ggml_ops_ext_cpu)
if (GGML_CUDA)
    target_link_libraries(test_ops PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ops PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_new_ops PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_new_ops PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_snake_op PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_snake_op PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_ada_ln_op PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ada_ln_op PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_debug_conv PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_debug_conv PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_ops_architecture PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ops_architecture PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_ops_concurrency PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ops_concurrency PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_ops_quantized_conv PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_ops_quantized_conv PRIVATE GGML_USE_CUDA)
    target_link_libraries(test_attention_streaming PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(test_attention_streaming PRIVATE GGML_USE_CUDA)
    target_link_libraries(benchmark_attention PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(benchmark_attention PRIVATE GGML_USE_CUDA)
endif()
if (GGML_SYCL)
    target_link_libraries(test_ops PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ops PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_new_ops PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_new_ops PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_snake_op PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_snake_op PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_ada_ln_op PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ada_ln_op PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_debug_conv PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_debug_conv PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_ops_architecture PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ops_architecture PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_ops_concurrency PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ops_concurrency PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_ops_quantized_conv PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_ops_quantized_conv PRIVATE GGML_USE_SYCL)
    target_link_libraries(test_attention_streaming PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(test_attention_streaming PRIVATE GGML_USE_SYCL)
    target_link_libraries(benchmark_attention PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(benchmark_attention PRIVATE GGML_USE_SYCL)
endif()

if (MSVC)
    target_compile_options(test_ops PRIVATE /EHsc /utf-8)
    target_compile_options(test_new_ops PRIVATE /EHsc /utf-8)
    target_compile_options(test_snake_op PRIVATE /EHsc /utf-8)
    target_compile_options(test_ada_ln_op PRIVATE /EHsc /utf-8)
    target_compile_options(test_debug_conv PRIVATE /EHsc /utf-8)
    target_compile_options(test_ops_architecture PRIVATE /EHsc /utf-8)
    target_compile_options(test_ops_concurrency PRIVATE /EHsc /utf-8)
    target_compile_options(test_ops_quantized_conv PRIVATE /EHsc /utf-8)
    target_compile_options(test_attention_streaming PRIVATE /EHsc /utf-8)
    target_compile_options(benchmark_attention PRIVATE /EHsc /utf-8)
endif()

if (GGML_SYCL)
    find_package(IntelSYCL)
    if (IntelSYCL_FOUND)
        target_link_libraries(test_ops PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_new_ops PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_snake_op PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_ada_ln_op PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_debug_conv PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_ops_architecture PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_ops_concurrency PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_ops_quantized_conv PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(test_attention_streaming PRIVATE IntelSYCL::SYCL_CXX)
        target_link_libraries(benchmark_attention PRIVATE IntelSYCL::SYCL_CXX)
    else()
        target_compile_options(test_ops PRIVATE "-fsycl")
        target_link_options(test_ops PRIVATE "-fsycl")
        target_compile_options(test_new_ops PRIVATE "-fsycl")
        target_link_options(test_new_ops PRIVATE "-fsycl")
        target_compile_options(test_snake_op PRIVATE "-fsycl")
        target_link_options(test_snake_op PRIVATE "-fsycl")
        target_compile_options(test_ada_ln_op PRIVATE "-fsycl")
        target_link_options(test_ada_ln_op PRIVATE "-fsycl")
        target_compile_options(test_debug_conv PRIVATE "-fsycl")
        target_link_options(test_debug_conv PRIVATE "-fsycl")
        target_compile_options(test_ops_architecture PRIVATE "-fsycl")
        target_link_options(test_ops_architecture PRIVATE "-fsycl")
        target_compile_options(test_ops_concurrency PRIVATE "-fsycl")
        target_link_options(test_ops_concurrency PRIVATE "-fsycl")
        target_compile_options(test_ops_quantized_conv PRIVATE "-fsycl")
        target_link_options(test_ops_quantized_conv PRIVATE "-fsycl")
        target_compile_options(test_attention_streaming PRIVATE "-fsycl")
        target_link_options(test_attention_streaming PRIVATE "-fsycl")
        target_compile_options(benchmark_attention PRIVATE "-fsycl")
        target_link_options(benchmark_attention PRIVATE "-fsycl")
    endif()
endif()
