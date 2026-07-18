add_executable(test_llm_api test_llm_api.cpp)
target_link_libraries(test_llm_api PRIVATE llm)
target_include_directories(test_llm_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME llm.api COMMAND test_llm_api)
set_tests_properties(llm.api PROPERTIES LABELS "llm;category;unit")
