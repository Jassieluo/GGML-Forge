cmake_minimum_required(VERSION 3.14)
project(test_ops C CXX)

set(OPS_TEST_TARGETS
    test_ops
    test_new_ops
    test_snake_op
    test_ada_ln_op
    test_debug_conv
    test_ops_architecture
    test_ops_concurrency
    test_ops_quantized_conv
    test_attention_streaming
    test_conv_nd
    test_ops_foundation
    test_pool_nd
    test_pad_nd
    test_adaptive_pool_nd
    test_resize_nd
    test_layer_norm
    test_instance_norm
    test_glu
    test_gated_activation
    test_relative_pe
    test_kv_cache_update
    test_embedding
    test_alias_free_activation
    test_new_virt_ops
    test_spectral
    test_length_regulate
    test_recurrent
)

set(REGISTERED_OPS_TEST_TARGETS
    test_ops
    test_snake_op
    test_ops_quantized_conv
    test_ops_architecture
    test_ops_concurrency
    test_attention_streaming
    test_conv_nd
    test_ops_foundation
    test_pool_nd
    test_pad_nd
    test_adaptive_pool_nd
    test_resize_nd
    test_layer_norm
    test_ada_ln_op
    test_instance_norm
    test_glu
    test_gated_activation
    test_relative_pe
    test_kv_cache_update
    test_embedding
    test_alias_free_activation
    test_new_virt_ops
)

foreach(test_target IN LISTS OPS_TEST_TARGETS)
    add_executable(${test_target} ${test_target}.cpp)
endforeach()
add_executable(benchmark_attention benchmark_attention.cpp)
add_executable(benchmark_snake benchmark_snake.cpp)

set(OPS_EXECUTABLE_TARGETS ${OPS_TEST_TARGETS} benchmark_attention benchmark_snake)

foreach(test_target IN LISTS REGISTERED_OPS_TEST_TARGETS)
    add_test(NAME ops.${test_target} COMMAND ${test_target})
    set_tests_properties(ops.${test_target} PROPERTIES LABELS "ops;unit")
endforeach()

# One ctest entry per spectral op so an edit re-runs only its own suite:
#   ctest -R "ops.stft"  /  ctest -L spectral
set(SPECTRAL_TEST_SUITES fft stft istft cabs)
foreach(spectral_suite IN LISTS SPECTRAL_TEST_SUITES)
    add_test(NAME ops.${spectral_suite} COMMAND test_spectral ${spectral_suite})
    set_tests_properties(ops.${spectral_suite} PROPERTIES LABELS "ops;unit;spectral")
endforeach()
add_test(NAME ops.length_regulate COMMAND test_length_regulate)
set_tests_properties(ops.length_regulate PROPERTIES LABELS "ops;unit;sequence")
set(RECURRENT_TEST_SUITES gru lstm)
foreach(recurrent_suite IN LISTS RECURRENT_TEST_SUITES)
    add_test(NAME ops.${recurrent_suite} COMMAND test_recurrent ${recurrent_suite})
    set_tests_properties(ops.${recurrent_suite} PROPERTIES LABELS "ops;unit;recurrent")
endforeach()

if (WIN32 AND GGML_SYCL)
    # SYCL/MKL runtime DLLs are supplied by the configured oneAPI toolchain.
    # Preserve the configure-time PATH when CTest runs from a plain shell.
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    foreach(test_target IN LISTS REGISTERED_OPS_TEST_TARGETS)
        set_tests_properties(ops.${test_target} PROPERTIES
            ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
        )
    endforeach()
    foreach(spectral_suite IN LISTS SPECTRAL_TEST_SUITES)
        set_tests_properties(ops.${spectral_suite} PROPERTIES
            ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
        )
    endforeach()
    set_tests_properties(ops.length_regulate PROPERTIES
        ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
    )
    foreach(recurrent_suite IN LISTS RECURRENT_TEST_SUITES)
        set_tests_properties(ops.${recurrent_suite} PROPERTIES
            ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
        )
    endforeach()
endif()

foreach(test_target IN LISTS OPS_EXECUTABLE_TARGETS)
    target_include_directories(${test_target} PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/include
        ${CMAKE_SOURCE_DIR}/ggml/src
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(${test_target} PRIVATE
        ggml
        ggml_ops_ext
        ggml_ops_ext_cpu
    )

    if (GGML_CUDA)
        target_link_libraries(${test_target} PRIVATE ggml_ops_ext_cuda)
        target_compile_definitions(${test_target} PRIVATE GGML_USE_CUDA)
    endif()

    if (GGML_SYCL)
        target_link_libraries(${test_target} PRIVATE ggml_ops_ext_sycl)
        target_compile_definitions(${test_target} PRIVATE GGML_USE_SYCL)
    endif()

    if (MSVC)
        target_compile_options(${test_target} PRIVATE /EHsc /utf-8)
    endif()
endforeach()

if (GGML_SYCL)
    find_package(IntelSYCL)
    foreach(test_target IN LISTS OPS_EXECUTABLE_TARGETS)
        if (IntelSYCL_FOUND)
            target_link_libraries(${test_target} PRIVATE IntelSYCL::SYCL_CXX)
        else()
            target_compile_options(${test_target} PRIVATE "-fsycl")
            target_link_options(${test_target} PRIVATE "-fsycl")
        endif()
    endforeach()
endif()

# Focused development target for Forge-owned operators. Backend dependencies
# are pulled in through the test executables; unrelated providers and apps are not.
add_custom_target(forge-ops-dev DEPENDS ${OPS_TEST_TARGETS})
