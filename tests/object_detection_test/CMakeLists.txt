add_executable(test_detection_api test_detection_api.cpp)
target_link_libraries(test_detection_api PRIVATE object_detection)
target_include_directories(test_detection_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME detection.api COMMAND test_detection_api)
set_tests_properties(detection.api PROPERTIES LABELS "detection;unit")

add_executable(test_yolo_common
    test_yolo_common.cpp
    ${CMAKE_SOURCE_DIR}/src/categories/object_detection/providers/yolo/yolo_common.cpp
)
target_include_directories(test_yolo_common PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/categories/object_detection
)
add_test(NAME detection.yolo_common COMMAND test_yolo_common)
set_tests_properties(detection.yolo_common PROPERTIES LABELS "detection;yolo;unit")

if (WIN32 AND GGML_SYCL)
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    set_tests_properties(detection.api detection.yolo_common PROPERTIES
        ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}"
    )
endif()
