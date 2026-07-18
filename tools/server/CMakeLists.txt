find_package(Threads REQUIRED)

set(FORGE_SERVER_LLAMA_VENDOR
    ${CMAKE_SOURCE_DIR}/src/categories/llm/providers/llama_cpp/llama.cpp/vendor)

add_library(forge-server-httplib STATIC
    ${FORGE_SERVER_LLAMA_VENDOR}/cpp-httplib/httplib.cpp)
target_include_directories(forge-server-httplib PUBLIC
    ${FORGE_SERVER_LLAMA_VENDOR}/cpp-httplib)
target_compile_definitions(forge-server-httplib PRIVATE
    CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH=16777216
    CPPHTTPLIB_LISTEN_BACKLOG=512
    CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=32768
    CPPHTTPLIB_TCP_NODELAY=1)
target_link_libraries(forge-server-httplib PUBLIC Threads::Threads)
if (WIN32)
    target_link_libraries(forge-server-httplib PUBLIC ws2_32)
endif()
if (MSVC)
    target_compile_options(forge-server-httplib PRIVATE /w)
else()
    target_compile_options(forge-server-httplib PRIVATE -w)
endif()

add_library(forge-server-core STATIC
    server_config.cpp
    model_registry.cpp
    codecs.cpp
    protocols/chat.cpp
    routes/common.cpp
    routes/llm.cpp
    routes/media.cpp
    server_context.cpp)
target_include_directories(forge-server-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${FORGE_SERVER_LLAMA_VENDOR}/nlohmann)
target_link_libraries(forge-server-core PUBLIC forge-server-httplib Threads::Threads)

if (INFERENCE_BUILD_LLM)
    target_compile_definitions(forge-server-core PUBLIC FORGE_SERVER_HAS_LLM=1)
    target_link_libraries(forge-server-core PUBLIC llm)
endif()
if (INFERENCE_BUILD_ASR)
    target_compile_definitions(forge-server-core PUBLIC FORGE_SERVER_HAS_ASR=1)
    target_link_libraries(forge-server-core PUBLIC asr)
endif()
if (INFERENCE_BUILD_TTS)
    target_compile_definitions(forge-server-core PUBLIC FORGE_SERVER_HAS_TTS=1)
    target_link_libraries(forge-server-core PUBLIC tts)
endif()
if (INFERENCE_BUILD_VISUAL_GENERATION)
    target_compile_definitions(forge-server-core PUBLIC FORGE_SERVER_HAS_VISUAL=1)
    target_link_libraries(forge-server-core PUBLIC visual_generation)
endif()

add_executable(forge-server main.cpp)
target_link_libraries(forge-server PRIVATE forge-server-core)
target_compile_features(forge-server-core PUBLIC cxx_std_17)
target_compile_features(forge-server PRIVATE cxx_std_17)
if (MSVC)
    target_compile_options(forge-server-core PRIVATE /EHsc /utf-8 /W3)
    target_compile_options(forge-server PRIVATE /EHsc /utf-8 /W3)
endif()
