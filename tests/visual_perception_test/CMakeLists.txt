if (INFERENCE_BUILD_INSTANCE_PERCEPTION)
    add_executable(test_instance_api test_instance_api.cpp)
    target_link_libraries(test_instance_api PRIVATE instance_perception)
    target_include_directories(test_instance_api PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME instance.api COMMAND test_instance_api)
    set_tests_properties(instance.api PROPERTIES LABELS "visual_perception;instance;unit")

    add_executable(test_yolo_common
        test_yolo_common.cpp
        ${CMAKE_SOURCE_DIR}/src/categories/visual_perception/instance_perception/providers/yolo/yolo_common.cpp
    )
    target_include_directories(test_yolo_common PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/categories/visual_perception/instance_perception
    )
    add_test(NAME instance.yolo_common COMMAND test_yolo_common)
    set_tests_properties(instance.yolo_common PROPERTIES
        LABELS "visual_perception;instance;yolo;unit")
endif()

if (INFERENCE_BUILD_DEPTH_ESTIMATION)
    add_executable(test_depth_api test_depth_api.cpp)
    target_link_libraries(test_depth_api PRIVATE depth_estimation)
    target_include_directories(test_depth_api PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME perception.depth_api COMMAND test_depth_api)
    set_tests_properties(perception.depth_api PROPERTIES
        LABELS "visual_perception;depth;unit")
endif()

if (INFERENCE_BUILD_SEMANTIC_SEGMENTATION)
    add_executable(test_segmentation_api test_segmentation_api.cpp)
    target_link_libraries(test_segmentation_api PRIVATE semantic_segmentation)
    target_include_directories(test_segmentation_api PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME perception.segmentation_api COMMAND test_segmentation_api)
    set_tests_properties(perception.segmentation_api PROPERTIES
        LABELS "visual_perception;segmentation;unit")
endif()

if (WIN32 AND GGML_SYCL)
    string(REPLACE ";" "\;" INFERENCE_TEST_PATH "$ENV{PATH}")
    set(_perception_tests)
    if (INFERENCE_BUILD_INSTANCE_PERCEPTION)
        list(APPEND _perception_tests instance.api instance.yolo_common)
    endif()
    if (INFERENCE_BUILD_DEPTH_ESTIMATION)
        list(APPEND _perception_tests perception.depth_api)
    endif()
    if (INFERENCE_BUILD_SEMANTIC_SEGMENTATION)
        list(APPEND _perception_tests perception.segmentation_api)
    endif()
    set_tests_properties(${_perception_tests} PROPERTIES
        ENVIRONMENT "PATH=${INFERENCE_TEST_PATH}")
endif()
