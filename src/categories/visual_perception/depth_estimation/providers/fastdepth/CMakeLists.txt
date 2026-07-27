add_library(fastdepth_models STATIC model.cpp)
target_include_directories(fastdepth_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include)
target_link_libraries(fastdepth_models PUBLIC ggml_nn_ext ggml_ops_ext)
if (MSVC)
    target_compile_options(fastdepth_models PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(fastdepth_models PRIVATE -Wall -Wextra)
endif()
