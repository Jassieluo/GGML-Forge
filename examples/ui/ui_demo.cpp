// GGML-Forge Studio — EUI-NEO demo application.
//
// This translation unit only owns the app configuration and the root compose
// pass. Everything else lives in focused headers:
//   pages/theme.h            colors, theme tokens, small draw helpers
//   pages/state.h            UI state, model paths, worker->UI stream bridge
//   pages/actions.h          task submission glue + chat agent tool loop
//   pages/sidebar.h          navigation, engine chips, backend card, header
//   pages/chat_page.h        scrollable markdown chat with streaming
//   pages/speech_page.h      GPT-SoVITS text to speech
//   pages/image_page.h       stable-diffusion.cpp text to image
//   services/engine_service.h cached runtimes/models + generation calls
//   services/tool_calls.h    <tool_call> protocol parsing for the agent

#include "eui_neo.h"

#include "pages/actions.h"
#include "pages/chat_page.h"
#include "pages/image_page.h"
#include "pages/sidebar.h"
#include "pages/speech_page.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace app {

// FORGE_UI_AUTOTEST=speech|image drives one agent tool round without typing,
// so the demo can be regression-tested headlessly. Fires once after startup.
static void maybeRunAutotest() {
    static int frames = 0;
    static bool fired = false;
    const char* mode = std::getenv("FORGE_UI_AUTOTEST");
    if (fired || !mode || ++frames < 2 || state.busy) return;
    fired = true;
    if (const char* backend = std::getenv("FORGE_UI_AUTOTEST_BACKEND")) {
        const std::string value(backend);
        state.backend = value == "sycl" ? kBackendSycl
                      : value == "cpu"  ? kBackendCpu
                                        : kBackendCuda;
    }
    std::fprintf(stderr, "[autotest] firing tool: %s backend=%s\n", mode,
                 kBackendNames[state.backend]);
    ToolCall call;
    if (std::string(mode) == "image") {
        call.name = "generate_image";
        call.prompt = "a small cabin in a snowy forest, warm light";
        call.steps = 8;
    } else {
        call.name = "text_to_speech";
        call.text = "你好，这是一条自动测试语音。";
    }
    state.agent_rounds = 1;
    beginTask("autotest");
    runToolCall(call);
}

// Note: SYCL kernels JIT-compile on first use (about two minutes on an
// iGPU), so the first SYCL generation is slow. The persistent JIT cache
// stays at the provider default (off) — enabling it crashed the SYCL
// runtime on the Iris Xe driver tested here.
const DslAppConfig& dslAppConfig() {
    static const DslAppConfig config = [] {
        auto value = DslAppConfig{}
            .title("GGML-Forge Studio")
            .pageId("ggml_forge_studio")
            .clearColor(kBackground)
            .windowSize(1240, 820)
            .fps(60.0);
#ifdef _WIN32
        value.fonts("C:/Windows/Fonts/msyh.ttc");
#endif
        return value;
    }();
    return config;
}

void compose(eui::Ui& ui, const eui::Screen& screen) {
    // One-shot: if the default backend has no device on this machine, fall
    // back instead of failing on the first generation.
    static const bool backend_validated = [] {
        if (!EngineService::backendAvailable(state.backend)) {
            state.backend = EngineService::backendAvailable(kBackendCuda)
                                ? kBackendCuda
                                : kBackendCpu;
            state.status = std::string("默认后端不可用 · 已切换为 ") +
                           kBackendNames[state.backend];
        }
        return true;
    }();
    (void)backend_validated;

    maybeRunAutotest();
    const bool compact = screen.width < 900.0f;
    const float sidebar_width = compact ? 82.0f : 224.0f;
    const float page_x = sidebar_width + 30.0f;
    const float page_width = std::max(320.0f, screen.width - page_x - 30.0f);
    const float content_y = 106.0f;
    const float content_height = std::max(360.0f, screen.height - content_y - 28.0f);

    ui.stack("root").size(screen.width, screen.height).content([&] {
        ui.rect("background").size(screen.width, screen.height).color(kBackground).build();
        composeSidebar(ui, sidebar_width, screen.height, compact);
        composeHeader(ui, page_x, page_width);
        if (state.tool == Tool::Chat) {
            composeChat(ui, page_x, content_y, page_width, content_height);
        } else if (state.tool == Tool::Speech) {
            composeSpeech(ui, page_x, content_y, page_width, content_height);
        } else {
            composeImage(ui, page_x, content_y, page_width, content_height);
        }
        if (state.busy) {
            ui.stack("task.progress.wrap").position(page_x, 96.0f).size(page_width, 3.0f).content([&] {
                components::progress(ui, "task.progress")
                    .size(page_width, 3.0f).value(state.progress.load()).theme(studioTheme()).build();
            }).build();
        }
    }).build();
}

} // namespace app
