inference_add_example(detect-image detect.cpp object_detection)
target_link_libraries(detect-image PRIVATE forge_image_io)
