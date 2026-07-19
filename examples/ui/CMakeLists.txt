add_executable(ui-demo
    ${CMAKE_SOURCE_DIR}/ui/eui_neo/core/app/glfw_app_main.cpp
    ui_demo.cpp)

eui_neo_configure_app(ui-demo)
eui_neo_copy_app_assets(ui-demo ${CMAKE_SOURCE_DIR}/ui/eui_neo/assets)
target_compile_features(ui-demo PRIVATE cxx_std_17)
