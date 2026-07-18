add_executable(test_asr_api test_asr_api.cpp)
target_link_libraries(test_asr_api PRIVATE asr)
target_include_directories(test_asr_api PRIVATE ${CMAKE_SOURCE_DIR}/include)

set(WHISPER_TEST_MODEL
    ${CMAKE_SOURCE_DIR}/src/categories/asr/providers/whisper_cpp/whisper.cpp/models/for-tests-ggml-base.en.bin)
add_test(NAME asr.api COMMAND test_asr_api ${WHISPER_TEST_MODEL})
set_tests_properties(asr.api PROPERTIES LABELS "asr;provider;unit")
