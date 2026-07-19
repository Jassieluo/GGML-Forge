# Keep EUI-NEO as an upstream-owned framework while exposing its eui::neo
# target to GGML-Forge. Forge applications remain under the root apps/ tree.
set(EUI_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_USER_APPS OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_TEST_FIXTURES OFF CACHE BOOL "" FORCE)
set(EUI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(EUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(EUI_ENABLE_MODULES ON CACHE BOOL "" FORCE)

# Deterministic offline defaults. Alternative backends can be exposed through
# Forge presets later when an application needs them.
set(EUI_DEPS_MODE bundled CACHE STRING "" FORCE)
set(EUI_WINDOW_BACKEND glfw CACHE STRING "" FORCE)
set(EUI_RENDER_BACKEND opengl CACHE STRING "" FORCE)

# EUI's bundled GLAD exposes global function-pointer data without Windows
# dllimport annotations, so it must remain static. Keep this setting scoped to
# the EUI dependency subtree and restore Forge's global shared-library policy.
set(FORGE_SAVED_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(eui_neo)
set(BUILD_SHARED_LIBS "${FORGE_SAVED_BUILD_SHARED_LIBS}")
unset(FORGE_SAVED_BUILD_SHARED_LIBS)
