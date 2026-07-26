add_executable(test_depth_api test_depth_api.cpp)
target_link_libraries(test_depth_api PRIVATE depth_estimation)
target_include_directories(test_depth_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME depth.api COMMAND test_depth_api)
set_tests_properties(depth.api PROPERTIES LABELS "depth;unit")

if (WIN32 AND GGML_SYCL)
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    set_tests_properties(depth.api PROPERTIES ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}")
endif()
