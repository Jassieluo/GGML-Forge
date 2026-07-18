set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
set(SD_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)

if (NOT TARGET ggml)
    message(FATAL_ERROR "visual generation requires the project root ggml target")
endif()

add_subdirectory(
    providers/stable_diffusion_cpp/stable-diffusion.cpp
    stable_diffusion_cpp_upstream
    EXCLUDE_FROM_ALL)

# Upstream includes selected private GGML headers through paths such as
# "ggml/src/ggml-impl.h". Its bundled ggml directory normally supplies that
# include root; when the project root GGML target is reused, provide the
# equivalent root privately without exposing GGML internals to our API.
target_include_directories(stable-diffusion PRIVATE ${CMAKE_SOURCE_DIR})

add_library(visual_generation SHARED
    visual_core.cpp
    providers/stable_diffusion_cpp/provider.cpp
)
target_compile_definitions(visual_generation PRIVATE VISUAL_BUILD_SHARED)
target_compile_definitions(visual_generation PUBLIC VISUAL_USE_SHARED)
set_target_properties(visual_generation PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(visual_generation PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(visual_generation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/stable_diffusion_cpp/stable-diffusion.cpp/include
)
target_link_libraries(visual_generation PRIVATE stable-diffusion ggml)

if (MSVC)
    target_compile_options(visual_generation PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(visual_generation PRIVATE -Wall -Wextra)
endif()
