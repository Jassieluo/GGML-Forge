set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_WEBUI OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)

# Upstream detects the root 'ggml' target and skips its bundled ggml directory.
add_subdirectory(providers/llama_cpp/llama.cpp llama_cpp_upstream EXCLUDE_FROM_ALL)

add_library(llm SHARED
    llm_core.cpp
    providers/llama_cpp/provider.cpp
)
target_compile_definitions(llm PRIVATE LLM_BUILD_SHARED)
target_compile_definitions(llm PUBLIC LLM_USE_SHARED)
set_target_properties(llm PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(llm PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(llm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/llama.cpp/tools/mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/providers/llama_cpp/llama.cpp/common
)
target_link_libraries(llm PRIVATE llama llama-common mtmd ggml)

if (MSVC)
    target_compile_options(llm PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(llm PRIVATE -Wall -Wextra)
endif()
