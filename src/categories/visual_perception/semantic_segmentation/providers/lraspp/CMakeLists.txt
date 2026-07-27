add_library(lraspp_models STATIC model.cpp)
target_include_directories(lraspp_models PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
)
target_link_libraries(lraspp_models PUBLIC ggml_nn_ext ggml_ops_ext)
if (MSVC)
    target_compile_options(lraspp_models PRIVATE /EHsc /utf-8 /W3)
else()
    target_compile_options(lraspp_models PRIVATE -Wall -Wextra)
endif()
