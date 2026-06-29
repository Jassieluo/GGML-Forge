# src/nn/CMakeLists.txt

add_library(ggml_nn_ext STATIC
    embedding.cpp
    linear.cpp
    conv.cpp
    norm.cpp
    feedforward.cpp
    attention.cpp
    transformer.cpp
    resblock.cpp
)

target_include_directories(ggml_nn_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(ggml_nn_ext PUBLIC ggml ggml_ops_ext)
