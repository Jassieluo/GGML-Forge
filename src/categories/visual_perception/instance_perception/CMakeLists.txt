if (NOT TARGET ggml)
    message(FATAL_ERROR "instance_perception requires the project root ggml target")
endif()

add_library(instance_perception SHARED
    instance_core.cpp
    providers/yolo/provider.cpp
    providers/yolo/yolo_common.cpp
)
target_compile_definitions(instance_perception PRIVATE INSTANCE_PERCEPTION_BUILD_SHARED)
target_compile_definitions(instance_perception PUBLIC INSTANCE_PERCEPTION_USE_SHARED)
set_target_properties(instance_perception PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(instance_perception PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(instance_perception PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(instance_perception PRIVATE
    ggml
    ggml_nn_ext
    yolo_models
    ggml_ops_ext
    ggml_ops_ext_cpu
)

if (GGML_CUDA)
    target_link_libraries(instance_perception PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(instance_perception PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    target_link_libraries(instance_perception PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(instance_perception PRIVATE GGML_USE_SYCL)
endif()

if (MSVC)
    target_compile_options(instance_perception PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(instance_perception PRIVATE -Wall -Wextra)
endif()
