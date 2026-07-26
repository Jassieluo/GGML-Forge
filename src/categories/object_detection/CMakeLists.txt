if (NOT TARGET ggml)
    message(FATAL_ERROR "object_detection requires the project root ggml target")
endif()

add_library(object_detection SHARED
    detection_core.cpp
)
target_compile_definitions(object_detection PRIVATE DETECTION_BUILD_SHARED)
target_compile_definitions(object_detection PUBLIC DETECTION_USE_SHARED)
set_target_properties(object_detection PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(object_detection PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(object_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(object_detection PRIVATE ggml)

if (MSVC)
    target_compile_options(object_detection PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(object_detection PRIVATE -Wall -Wextra)
endif()
