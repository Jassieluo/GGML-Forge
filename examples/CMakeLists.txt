# examples/CMakeLists.txt

function(gpt_sovits_add_example target_name source_name)
    add_executable(${target_name} ${source_name})
    target_link_libraries(${target_name} PRIVATE tts tts_models tts_ops tts_frontends ggml)
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/include 
        ${CMAKE_SOURCE_DIR}/ggml/include
    )
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /utf-8)
    endif()
endfunction()

add_subdirectory(test-backend)
add_subdirectory(hubert-alignment)
add_subdirectory(bert-alignment)
add_subdirectory(t2s-alignment)
add_subdirectory(vits-alignment)
add_subdirectory(test-pipeline)
