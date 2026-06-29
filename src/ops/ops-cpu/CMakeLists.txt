add_library(ggml_ops_ext_cpu
    ops_cpu.cpp
    matmul_f32.cpp
    conv_1d.cpp
    conv_transpose_1d.cpp
    mish/mish.cpp
    mish/mish_avx2.cpp
    mish/mish_avx512.cpp
    gated_tanh_sigmoid/gated_tanh_sigmoid.cpp
    gated_tanh_sigmoid/gated_tanh_sigmoid_avx2.cpp
    gated_tanh_sigmoid/gated_tanh_sigmoid_avx512.cpp
    layer_norm.cpp
    double_swish/double_swish.cpp
    double_swish/double_swish_avx2.cpp
    double_swish/double_swish_avx512.cpp
)

target_include_directories(ggml_ops_ext_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/src
)

target_link_libraries(ggml_ops_ext_cpu PUBLIC ggml ggml_ops_ext)

find_package(OpenMP)
if (OpenMP_FOUND)
    target_link_libraries(ggml_ops_ext_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()


if (MSVC)
    set_source_files_properties(mish/mish_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(mish/mish_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
    set_source_files_properties(gated_tanh_sigmoid/gated_tanh_sigmoid_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(gated_tanh_sigmoid/gated_tanh_sigmoid_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
    set_source_files_properties(double_swish/double_swish_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(double_swish/double_swish_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
    set_source_files_properties(mish/mish_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    set_source_files_properties(mish/mish_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
    set_source_files_properties(gated_tanh_sigmoid/gated_tanh_sigmoid_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    set_source_files_properties(gated_tanh_sigmoid/gated_tanh_sigmoid_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
    set_source_files_properties(double_swish/double_swish_avx2.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
    set_source_files_properties(double_swish/double_swish_avx512.cpp PROPERTIES COMPILE_FLAGS "-mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw")
endif()

# Export all symbols on Windows for dynamic DLL linking
set_target_properties(ggml_ops_ext_cpu PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
