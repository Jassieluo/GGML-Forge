add_executable(test_detection_api test_detection_api.cpp)
target_link_libraries(test_detection_api PRIVATE object_detection)
target_include_directories(test_detection_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME detection.api COMMAND test_detection_api)
set_tests_properties(detection.api PROPERTIES LABELS "detection;unit")

if (WIN32 AND GGML_SYCL)
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    set_tests_properties(detection.api PROPERTIES ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}")
endif()
