if (NOT TARGET ggml)
    message(FATAL_ERROR "semantic_segmentation requires the project root ggml target")
endif()

add_library(semantic_segmentation SHARED
    segmentation_core.cpp
)
target_compile_definitions(semantic_segmentation PRIVATE SEGMENTATION_BUILD_SHARED)
target_compile_definitions(semantic_segmentation PUBLIC SEGMENTATION_USE_SHARED)
set_target_properties(semantic_segmentation PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(semantic_segmentation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(semantic_segmentation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(semantic_segmentation PRIVATE ggml)

if (MSVC)
    target_compile_options(semantic_segmentation PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(semantic_segmentation PRIVATE -Wall -Wextra)
endif()
