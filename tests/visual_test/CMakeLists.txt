add_executable(test_visual_api test_visual_api.cpp)
target_link_libraries(test_visual_api PRIVATE visual_generation)
target_include_directories(test_visual_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME visual.api COMMAND test_visual_api)
set_tests_properties(visual.api PROPERTIES LABELS "visual;category;unit")
