# src/nn/CMakeLists.txt

add_library(ggml_nn_ext STATIC
    core/context.cpp
    core/executor.cpp
    core/parameter.cpp
    core/module.cpp
    core/state_dict.cpp
    schema/model_schema.cpp
    io/load.cpp
    io/gguf_source.cpp
    layers/embedding.cpp
    layers/linear.cpp
    layers/activation.cpp
    functional/linear.cpp
    functional/normalization.cpp
    functional/activation.cpp
    functional/attention.cpp
    functional/convolution.cpp
    layers/convolution.cpp
    layers/normalization.cpp
    layers/feedforward.cpp
    layers/attention.cpp
    runtime/kv_cache.cpp
    layers/transformer.cpp
    functional/positional.cpp
    functional/elementwise.cpp
    functional/interpolation.cpp
)

target_include_directories(ggml_nn_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

target_link_libraries(ggml_nn_ext PUBLIC ggml ggml_ops_ext)
