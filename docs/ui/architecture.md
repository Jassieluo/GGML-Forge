# UI architecture

EUI-NEO is managed as an untouched upstream submodule at `ui/eui_neo`. The
root `ui/CMakeLists.txt` is the Forge-owned integration boundary; application
code will live under `apps/` and must not be added to the upstream tree.

The UI framework is part of the normal build through `INFERENCE_BUILD_UI`,
which defaults to `ON`. The current reproducible configuration is:

- bundled dependency sources, with no configure-time downloads;
- GLFW window backend;
- OpenGL render backend;
- static `eui_neo` framework linked into UI executables;
- optional EUI modules enabled;
- upstream applications, examples, test fixtures, and install rules disabled.

The resulting public CMake target is `eui::neo`. No production Forge
application links it yet. The standalone `ui-demo` target under `examples/ui`
demonstrates EUI-NEO composition and local UI state without loading an
inference model.

Forge keeps its repository-wide `BUILD_SHARED_LIBS=ON` preference. EUI-NEO's
own shared mode is temporarily disabled because the current Windows upstream
build exits during runtime initialization when its app facade crosses the DLL
boundary. The same target runs correctly with the EUI core linked statically.
Other dependencies that honor `BUILD_SHARED_LIBS` may still produce DLLs. This
compatibility setting is isolated to `ui/CMakeLists.txt` and can be removed
when upstream shared-mode initialization is fixed.
