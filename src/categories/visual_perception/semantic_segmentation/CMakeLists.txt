if (NOT TARGET ggml)
    message(FATAL_ERROR "semantic_segmentation requires the project root ggml target")
endif()

add_subdirectory(providers/lraspp)

add_library(semantic_segmentation SHARED
    segmentation_core.cpp
    providers/lraspp/provider.cpp
)
target_compile_definitions(semantic_segmentation PRIVATE SEGMENTATION_BUILD_SHARED)
target_compile_definitions(semantic_segmentation PUBLIC SEGMENTATION_USE_SHARED)
set_target_properties(semantic_segmentation PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(semantic_segmentation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(semantic_segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/categories/visual_perception
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(semantic_segmentation PRIVATE
    ggml ggml_nn_ext lraspp_models ggml_ops_ext ggml_ops_ext_cpu forge_image_io
)

if (GGML_CUDA)
    target_link_libraries(semantic_segmentation PRIVATE ggml_ops_ext_cuda)
    target_compile_definitions(semantic_segmentation PRIVATE GGML_USE_CUDA)
endif()
if (GGML_SYCL)
    target_link_libraries(semantic_segmentation PRIVATE ggml_ops_ext_sycl)
    target_compile_definitions(semantic_segmentation PRIVATE GGML_USE_SYCL)
endif()

if (MSVC)
    target_compile_options(semantic_segmentation PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(semantic_segmentation PRIVATE -Wall -Wextra)
endif()
