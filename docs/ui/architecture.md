# UI architecture

EUI-NEO is managed as an untouched upstream submodule at `ui/eui_neo`. The
root `ui/CMakeLists.txt` is the Forge-owned integration boundary; application
code will live under `apps/` and must not be added to the upstream tree.

The UI framework is part of the normal build through `INFERENCE_BUILD_UI`,
which defaults to `ON`. The current reproducible configuration is:

- bundled dependency sources, with no configure-time downloads;
- GLFW window backend;
- render backend selected by the top-level `EUI_RENDER_BACKEND` cache
  variable (`auto`, `opengl`, or `vulkan`; the current default is `vulkan`);
- static `eui_neo` framework linked into UI executables;
- optional EUI modules enabled;
- upstream applications, examples, test fixtures, and install rules disabled.

The resulting public CMake target is `eui::neo`. No production Forge
application links it yet.

Forge keeps its repository-wide `BUILD_SHARED_LIBS=ON` preference. EUI-NEO's
own shared mode is temporarily disabled because the current Windows upstream
build exits during runtime initialization when its app facade crosses the DLL
boundary. The same target runs correctly with the EUI core linked statically.
Other dependencies that honor `BUILD_SHARED_LIBS` may still produce DLLs. This
compatibility setting is isolated to `ui/CMakeLists.txt` and can be removed
when upstream shared-mode initialization is fixed.

## ui-demo

The standalone `ui-demo` target under `examples/ui` demonstrates EUI-NEO
composition against the public category APIs (LLM chat, GPT-SoVITS speech,
stable-diffusion.cpp images). It is intentionally a demo, not a product
application, but it follows the structure a real Forge app under `apps/`
would use:

- `ui_demo.cpp` owns only the app configuration and root compose pass.
- `pages/theme.h` holds colors, theme tokens, and small draw helpers.
- `pages/state.h` holds UI state, model paths, and the worker-to-UI stream
  bridge for token streaming.
- `pages/actions.h` submits work through `core::async` behind a single busy
  guard, so engine access is serialized. It also owns the chat agent loop
  (below).
- `pages/sidebar.h`, `pages/chat_page.h`, `pages/speech_page.h`, and
  `pages/image_page.h` are the compose functions per surface.
- `services/engine_service.h` caches each category's runtime, model, and
  session per backend, so repeated requests do not reload weights from disk.
  Models stay resident between requests; the sidebar shows what is loaded
  and offers an explicit release action. The TTS session is created with the
  bundled default reference voice (`models/tts/gpt_sovits/voices/doubao`),
  which GPT-SoVITS requires before it can synthesize.
- `services/tool_calls.h` parses the `<tool_call>` text protocol emitted by
  the chat model.

The chat page runs a small agent loop: a system prompt declares two tools
(`generate_image`, `text_to_speech`), and a finished reply may contain one
`<tool_call>{...}</tool_call>` block. The completion callback parses it,
executes the matching engine call on the worker, appends the result to the
conversation as a `tool` message (with an inline image or an audio play
button in the chat bubble), and re-queries the LLM with the tool result, up
to three rounds per user message. Half-streamed tool-call JSON is never
rendered; the live bubble swaps it for an activity note.

Qwen-style `<think>` reasoning is split off every reply before tool parsing
and history building (`tool_calls::stripThinking`), so the model never sees
its own reasoning back. In the UI the reasoning renders as a collapsible
"深度思考" section inside the bubble: expanded and dimmed while it streams,
collapsed once the answer starts, toggleable afterwards. Visual language
lives in `pages/theme.h`: a three-level charcoal-blue surface scale with a
mint accent, a fixed type scale, and shared helpers (panel, section label,
labeled progress). The sidebar picks the backend with a segmented control
(ignored while busy) instead of a cycle button.

Progress reporting: visual generation reports sampler steps through
`visual_session_set_callbacks`; TTS reports through the new
`tts_session_set_progress_callback` category API, which GPT-SoVITS feeds
from per-segment windows and an expected-token estimate of the T2S decode
loop. The speech and image pages surface both as a labeled progress bar with
percentage, in addition to the thin task bar above every page.

Notable runtime behavior: LLM answers stream token-by-token into the chat
view; the chat list is a measured scroll view with markdown bubbles; the
llama.cpp provider exposes no per-device selection, so the CUDA/SYCL choice
applies to TTS and visual generation while the LLM only distinguishes CPU
(no offload) from GPU (provider-selected device).

Backend availability: `EngineService::backendAvailable` probes the ggml
device registry once at startup (the demo links `ggml` directly for this);
the sidebar refuses switching to a backend with no device and the app falls
back if the default backend is missing. For SYCL to work without an oneAPI
shell, the top-level CMakeLists copies the oneAPI runtime DLLs
(sycl8, ur_*, mkl, dnnl, tbb, svml) next to the binaries when `GGML_SYCL`
is on. The first SYCL generation JIT-compiles kernels (about two minutes on
an iGPU); the SYCL persistent JIT cache stays off because it crashed the
runtime on the tested Iris Xe driver. stable-diffusion.cpp load failures
are surfaced through a log-callback bridge in the Forge provider
(`FORGE_SD_VERBOSE` adds info-level output).
