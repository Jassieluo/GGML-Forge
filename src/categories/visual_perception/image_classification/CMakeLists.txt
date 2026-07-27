if (NOT TARGET ggml)
    message(FATAL_ERROR "image_classification requires the project root ggml target")
endif()

add_library(image_classification SHARED
    classification_core.cpp
    providers/yolo/provider.cpp
)
target_compile_definitions(image_classification PRIVATE CLASSIFICATION_BUILD_SHARED)
target_compile_definitions(image_classification PUBLIC CLASSIFICATION_USE_SHARED)
set_target_properties(image_classification PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(image_classification PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(image_classification PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/categories/visual_perception
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(image_classification PRIVATE
    ggml ggml_nn_ext yolo_models ggml_ops_ext ggml_ops_ext_cpu
)

if (GGML_CUDA)
    target_link_libraries(image_classification PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(image_classification PRIVATE GGML_USE_CUDA)
endif()
if (GGML_SYCL)
    target_link_libraries(image_classification PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(image_classification PRIVATE GGML_USE_SYCL)
endif()
if (MSVC)
    target_compile_options(image_classification PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(image_classification PRIVATE -Wall -Wextra)
endif()
