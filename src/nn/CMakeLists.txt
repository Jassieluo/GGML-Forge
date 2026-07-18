# src/nn/CMakeLists.txt

add_library(ggml_nn_ext STATIC
    context.cpp
    executor.cpp
    parameter.cpp
    module.cpp
    model_schema.cpp
    state_dict.cpp
    load.cpp
    gguf_source.cpp
    embedding.cpp
    linear.cpp
    activation.cpp
    conv.cpp
    norm.cpp
    feedforward.cpp
    attention.cpp
    kv_cache.cpp
    transformer.cpp
    resblock.cpp
    prelu.cpp
    positional.cpp
    tensor_ops.cpp
)

target_include_directories(ggml_nn_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(ggml_nn_ext PUBLIC ggml ggml_ops_ext)
