add_executable(test_segmentation_api test_segmentation_api.cpp)
target_link_libraries(test_segmentation_api PRIVATE semantic_segmentation)
target_include_directories(test_segmentation_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME segmentation.api COMMAND test_segmentation_api)
set_tests_properties(segmentation.api PROPERTIES LABELS "segmentation;unit")

if (WIN32 AND GGML_SYCL)
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    set_tests_properties(segmentation.api PROPERTIES ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}")
endif()
