inference_add_example(asr-transcribe transcribe.cpp asr)
target_link_libraries(asr-transcribe PRIVATE forge_audio_io)
