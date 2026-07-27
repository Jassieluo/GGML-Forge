add_executable(test_media_common test_media_common.cpp)
target_link_libraries(test_media_common PRIVATE
    forge_image_io
    forge_audio_io
    forge_audio_dsp)
if (MSVC)
    target_compile_options(test_media_common PRIVATE /EHsc /utf-8)
else()
    target_compile_options(test_media_common PRIVATE -Wall -Wextra)
endif()

add_test(NAME media.common COMMAND test_media_common)
set_tests_properties(media.common PROPERTIES LABELS "media;unit")
