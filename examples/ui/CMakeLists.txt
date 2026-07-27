add_executable(ui-demo
    ${CMAKE_SOURCE_DIR}/ui/eui_neo/core/app/glfw_app_main.cpp
    ui_demo.cpp)

eui_neo_configure_app(ui-demo)
eui_neo_copy_app_assets(ui-demo ${CMAKE_SOURCE_DIR}/ui/eui_neo/assets)
target_compile_features(ui-demo PRIVATE cxx_std_17)
target_include_directories(ui-demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include)
# ggml is linked directly for runtime backend/device probing in the sidebar.
target_link_libraries(ui-demo PRIVATE tts llm visual_generation ggml forge_audio_io)
if (WIN32)
    # PlaySoundW in services/engine_service.h.
    target_link_libraries(ui-demo PRIVATE winmm)
endif()
