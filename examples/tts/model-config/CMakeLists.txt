inference_add_example(tts-model-config model-config.cpp tts)
target_link_libraries(tts-model-config PRIVATE forge_audio_io)
