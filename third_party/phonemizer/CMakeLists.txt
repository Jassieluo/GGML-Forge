# third_party/phonemizer/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(phonemizer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the phonemizer shared library
add_library(phonemizer SHARED
    src/phonemizer.cpp
)
target_compile_definitions(phonemizer PRIVATE PHONEMIZER_BUILD_SHARED)
target_compile_definitions(phonemizer PUBLIC PHONEMIZER_USE_SHARED)
set_target_properties(phonemizer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Include directories
target_include_directories(phonemizer PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp-jieba/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../cpp-jieba/deps/limonp/include>
)

# Enable extra warning flags
if (MSVC)
    target_compile_options(phonemizer PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(phonemizer PRIVATE -Wall -Wextra)
endif()

# Build the unit test executable if tests are enabled
# add_executable(test_phonemizer_bin
#     ../../tests/test_phonemizer.cpp
# )
# 
# target_link_libraries(test_phonemizer_bin PRIVATE
#     phonemizer
# )
# 
# if (MSVC)
#     target_compile_options(test_phonemizer_bin PRIVATE /EHsc /utf-8)
# endif()
