add_executable(test_model_config
    test_model_config.cpp
    ${CMAKE_SOURCE_DIR}/src/categories/tts/model_config.cpp
)

add_test(NAME tts.model_config COMMAND test_model_config)
set_tests_properties(tts.model_config PROPERTIES LABELS "tts;category;unit")

add_executable(test_tts_provider_contract test_provider_contract.cpp)
target_include_directories(test_tts_provider_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/src/categories/tts
)
add_test(NAME tts.provider_contract COMMAND test_tts_provider_contract)
set_tests_properties(tts.provider_contract PROPERTIES LABELS "tts;category;provider;unit")

target_include_directories(test_model_config PRIVATE
    ${CMAKE_SOURCE_DIR}/src/categories/tts
    ${CMAKE_SOURCE_DIR}/include
)

if (MSVC)
    target_compile_options(test_model_config PRIVATE /EHsc /utf-8)
    target_compile_options(test_tts_provider_contract PRIVATE /EHsc /utf-8)
else()
    target_compile_options(test_model_config PRIVATE -Wall -Wextra)
    target_compile_options(test_tts_provider_contract PRIVATE -Wall -Wextra)
endif()
