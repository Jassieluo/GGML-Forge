# examples/CMakeLists.txt

function(inference_add_example target_name source_name category_target)
    add_executable(${target_name} ${source_name})
    target_link_libraries(${target_name} PRIVATE ${category_target})
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/include
    )
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /utf-8)
    endif()
endfunction()

if (INFERENCE_BUILD_ASR)
    add_subdirectory(asr)
endif()
if (INFERENCE_BUILD_LLM)
    add_subdirectory(llm)
endif()
if (INFERENCE_BUILD_TTS)
    add_subdirectory(tts)
endif()
if (INFERENCE_BUILD_VISUAL_GENERATION)
    add_subdirectory(visual_generation)
endif()
if (INFERENCE_BUILD_SERVER)
    add_subdirectory(server)
endif()
