# examples/CMakeLists.txt

function(gpt_sovits_add_example target_name source_name)
    add_executable(${target_name} ${source_name})
    target_link_libraries(${target_name} PRIVATE tts ggml phonemizer)
    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include 
        ${CMAKE_CURRENT_SOURCE_DIR}/../ggml/include
    )
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /EHsc /utf-8)
    endif()
endfunction()

gpt_sovits_add_example(test_backend     test_backend.cpp)
gpt_sovits_add_example(hubert_alignment hubert_alignment.cpp)
gpt_sovits_add_example(bert_alignment   bert_alignment.cpp)
gpt_sovits_add_example(t2s_alignment    t2s_alignment.cpp)
gpt_sovits_add_example(vits_alignment   vits_alignment.cpp)
