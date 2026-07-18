# examples/CMakeLists.txt

function(gpt_sovits_add_example target_name source_name)
    add_executable(${target_name} ${source_name})
    target_link_libraries(${target_name} PRIVATE tts)
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/examples
        ${CMAKE_SOURCE_DIR}/include
    )
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /utf-8)
    endif()
endfunction()

add_subdirectory(tts)
