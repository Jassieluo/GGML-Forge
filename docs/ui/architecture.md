# UI architecture

EUI-NEO is managed as an untouched upstream submodule at `ui/eui_neo`. The
root `ui/CMakeLists.txt` is the Forge-owned integration boundary; application
code will live under `apps/` and must not be added to the upstream tree.

The UI framework is part of the normal build through `INFERENCE_BUILD_UI`,
which defaults to `ON`. The current reproducible configuration is:

- bundled dependency sources, with no configure-time downloads;
- GLFW window backend;
- OpenGL render backend;
- shared `eui_neo` library;
- optional EUI modules enabled;
- upstream applications, examples, test fixtures, and install rules disabled.

The resulting public CMake target is `eui::neo`. No Forge application links it
yet; adding an application is a separate concern from maintaining and building
the framework.

EUI-NEO follows the repository-wide shared-library preference. Dependencies
that honor `BUILD_SHARED_LIBS` may therefore produce DLLs, while dependencies
that explicitly declare static targets remain static.
