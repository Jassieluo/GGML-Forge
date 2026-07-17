add_executable(test_model_config
    test_model_config.cpp
    ${CMAKE_SOURCE_DIR}/src/model_config.cpp
)

target_include_directories(test_model_config PRIVATE ${CMAKE_SOURCE_DIR}/src)

if (MSVC)
    target_compile_options(test_model_config PRIVATE /EHsc /utf-8)
else()
    target_compile_options(test_model_config PRIVATE -Wall -Wextra)
endif()
