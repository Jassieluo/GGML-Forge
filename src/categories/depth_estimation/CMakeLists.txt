if (NOT TARGET ggml)
    message(FATAL_ERROR "depth_estimation requires the project root ggml target")
endif()

add_library(depth_estimation SHARED
    depth_core.cpp
)
target_compile_definitions(depth_estimation PRIVATE DEPTH_BUILD_SHARED)
target_compile_definitions(depth_estimation PUBLIC DEPTH_USE_SHARED)
set_target_properties(depth_estimation PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(depth_estimation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(depth_estimation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(depth_estimation PRIVATE ggml)

if (MSVC)
    target_compile_options(depth_estimation PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(depth_estimation PRIVATE -Wall -Wextra)
endif()
