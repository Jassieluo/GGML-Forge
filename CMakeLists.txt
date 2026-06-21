# CMakeLists.txt for GPT-SoVITS.cpp

cmake_minimum_required(VERSION 3.14)
project(gpt_sovits C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)
add_subdirectory(third_party/phonemizer)

# Define the gpt_sovits shared library
add_library(gpt_sovits SHARED
    src/gpt_sovits.cpp
    src/ggml_hubert.cpp
    src/ggml_bert.cpp
    src/ggml_t2s.cpp
    src/ggml_vits.cpp
)
target_compile_definitions(gpt_sovits PRIVATE GPT_SOVITS_BUILD_SHARED)
target_compile_definitions(gpt_sovits PUBLIC GPT_SOVITS_USE_SHARED)
set_target_properties(gpt_sovits PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Include directories
target_include_directories(gpt_sovits PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ggml/include>
)

# Link libraries
target_link_libraries(gpt_sovits PRIVATE
    ggml
    phonemizer
)

if (MSVC)
    target_compile_options(gpt_sovits PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(gpt_sovits PRIVATE -Wall -Wextra)
endif()

add_subdirectory(examples)

