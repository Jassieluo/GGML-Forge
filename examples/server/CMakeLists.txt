set(FORGE_SERVER_EXAMPLE_VENDOR
    ${CMAKE_SOURCE_DIR}/src/categories/llm/providers/llama_cpp/llama.cpp/vendor)

add_library(forge-server-example-client STATIC client_common.cpp)
target_include_directories(forge-server-example-client PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FORGE_SERVER_EXAMPLE_VENDOR}/cpp-httplib
    ${FORGE_SERVER_EXAMPLE_VENDOR}/nlohmann)
target_link_libraries(forge-server-example-client PUBLIC forge-server-httplib)
if (WIN32)
    target_link_libraries(forge-server-example-client PUBLIC shell32)
endif()
target_compile_features(forge-server-example-client PUBLIC cxx_std_17)

function(forge_add_server_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE forge-server-example-client)
    if (MSVC)
        target_compile_options(${name} PRIVATE /EHsc /utf-8)
    endif()
endfunction()

forge_add_server_example(server-openai-chat openai_chat.cpp)
forge_add_server_example(server-anthropic-message anthropic_message.cpp)
forge_add_server_example(server-transcribe transcribe.cpp)
forge_add_server_example(server-speech speech.cpp)
forge_add_server_example(server-image image.cpp)
forge_add_server_example(server-video video.cpp)
