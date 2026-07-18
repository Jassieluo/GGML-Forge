# Keep EUI-NEO as an upstream-owned framework while exposing its eui::neo
# target to GGML-Forge. Forge applications remain under the root apps/ tree.
set(EUI_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_USER_APPS OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_TEST_FIXTURES OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_SHARED ON CACHE BOOL "" FORCE)
set(EUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(EUI_ENABLE_MODULES ON CACHE BOOL "" FORCE)

# Deterministic offline defaults. Alternative backends can be exposed through
# Forge presets later when an application needs them.
set(EUI_DEPS_MODE bundled CACHE STRING "" FORCE)
set(EUI_WINDOW_BACKEND glfw CACHE STRING "" FORCE)
set(EUI_RENDER_BACKEND opengl CACHE STRING "" FORCE)

add_subdirectory(eui_neo)
