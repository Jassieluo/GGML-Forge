# CMakeLists.txt for GPT-SoVITS.cpp

cmake_minimum_required(VERSION 3.14)
project(gpt_sovits C CXX ASM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)
add_subdirectory(third_party/phonemizer)

add_subdirectory(src)

add_subdirectory(examples)

