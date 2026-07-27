add_library(yolo_models STATIC
    v8/instance_model.cpp
    v8/classification_model.cpp
)
set_target_properties(yolo_models PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(yolo_models PUBLIC
    ${CMAKE_SOURCE_DIR}/src/categories/visual_perception
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(yolo_models PUBLIC ggml_nn_ext)
