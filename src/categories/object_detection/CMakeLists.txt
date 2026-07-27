if (NOT TARGET ggml)
    message(FATAL_ERROR "object_detection requires the project root ggml target")
endif()

add_library(yolo_models STATIC
    providers/yolo/v8/yolo_v8.cpp
)
set_target_properties(yolo_models PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(yolo_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(yolo_models PUBLIC ggml_nn_ext)

add_library(object_detection SHARED
    detection_core.cpp
    providers/yolo/provider.cpp
    providers/yolo/yolo_common.cpp
)
target_compile_definitions(object_detection PRIVATE DETECTION_BUILD_SHARED)
target_compile_definitions(object_detection PUBLIC DETECTION_USE_SHARED)
set_target_properties(object_detection PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(object_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(object_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(object_detection PRIVATE
    ggml
    ggml_nn_ext
    yolo_models
    ggml_ops_ext
    ggml_ops_ext_cpu
)

if (GGML_CUDA)
    target_link_libraries(object_detection PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(object_detection PRIVATE GGML_USE_CUDA)
endif()

if (GGML_SYCL)
    target_link_libraries(object_detection PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(object_detection PRIVATE GGML_USE_SYCL)
endif()

if (MSVC)
    target_compile_options(object_detection PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(object_detection PRIVATE -Wall -Wextra)
endif()
