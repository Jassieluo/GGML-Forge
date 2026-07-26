#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"

#include <string>
#include <vector>

namespace app {

inline const char* toolTitle() {
    if (state.tool == Tool::Speech) return "语音合成";
    if (state.tool == Tool::Image) return "视觉生成";
    return "本地对话";
}

inline const char* toolSubtitle() {
    if (state.tool == Tool::Speech) return "GPT-SoVITS · 文本转语音";
    if (state.tool == Tool::Image) return "stable-diffusion.cpp · 文本生成图片";
    return "llama.cpp · Qwen3.5 4B · 支持工具调用";
}

inline void navButton(eui::Ui& ui, const std::string& id, const std::string& label,
                      unsigned int icon, Tool tool, float x, float y, float width,
                      bool compact) {
    const bool selected = state.tool == tool;
    components::ButtonStyle style(studioTheme(), selected);
    style.normal = selected ? kAccentSoft : kSidebar;
    style.hover = selected ? kAccentSoft : kSurface;
    style.pressed = kSurfaceRaised;
    style.text = selected ? kAccent : kMuted;
    style.icon = selected ? kAccent : kFaint;
    style.border = {selected ? 1.0f : 0.0f, selected ? kAccentEdge : kSidebar};
    style.radius = 12.0f;
    components::button(ui, id).position(x, y).size(width, 44.0f)
        .text(compact ? "" : label).icon(icon).fontSize(13.5f).iconSize(16.0f)
        .style(style).disabled(state.busy)
        .onClick([tool] {
            state.tool = tool;
            state.status = "就绪";
            state.has_error = false;
        })
        .build();
}

inline void engineChip(eui::Ui& ui, const std::string& id, const std::string& label,
                       bool on, float x, float y, float width) {
    ui.rect(id + ".bg").position(x, y).size(width, 24.0f)
        .color(on ? kAccentSoft : kSurface)
        .radius(7.0f)
        .border(1.0f, on ? kAccentEdge : kBorderSoft)
        .build();
    ui.rect(id + ".dot").position(x + 8.0f, y + 9.0f).size(6.0f, 6.0f)
        .color(on ? kAccent : kFaint).radius(3.0f).build();
    text(ui, id + ".label", label, x + 19.0f, y + 4.0f, width - 24.0f, 16.0f,
         kFontOverline, on ? kAccent : kMuted, 700);
}

inline void composeSidebar(eui::Ui& ui, float width, float height, bool compact) {
    ui.rect("sidebar.background").size(width, height).color(kSidebar)
        .border(1.0f, kBorderSoft).build();

    // Brand: rounded accent mark + wordmark.
    ui.rect("brand.tile").position(20.0f, 22.0f).size(36.0f, 36.0f)
        .color(kAccentSoft).radius(11.0f).border(1.0f, kAccentEdge).build();
    text(ui, "brand.mark", "F", 32.0f, 27.0f, 24.0f, 28.0f, 19.0f, kAccent, 850);
    if (!compact) {
        text(ui, "brand.name", "GGML-Forge", 66.0f, 24.0f, width - 84.0f, 24.0f, 17.0f, kText, 780);
        text(ui, "brand.caption", "LOCAL AI STUDIO", 66.0f, 46.0f, width - 84.0f, 16.0f,
             9.0f, kFaint, 700);
    }

    const float button_x = compact ? 14.0f : 18.0f;
    const float button_width = width - button_x * 2.0f;
    navButton(ui, "nav.chat", "本地对话", 0xF075, Tool::Chat, button_x, 96.0f, button_width, compact);
    navButton(ui, "nav.speech", "语音合成", 0xF130, Tool::Speech, button_x, 148.0f, button_width, compact);
    navButton(ui, "nav.image", "视觉生成", 0xF03E, Tool::Image, button_x, 200.0f, button_width, compact);

    if (compact) return;

    // Loaded engine chips: models stay resident between requests, so show
    // what is currently holding memory.
    const auto& flags = EngineService::loaded();
    const float chips_y = height - 232.0f;
    sectionLabel(ui, "engines.label", "常驻模型", 22.0f, chips_y - 24.0f, width - 44.0f);
    const float chip_width = (width - 36.0f - 12.0f) / 3.0f;
    engineChip(ui, "engines.llm", "LLM", flags.llm.load(), 18.0f, chips_y, chip_width);
    engineChip(ui, "engines.tts", "TTS", flags.tts.load(), 18.0f + chip_width + 6.0f, chips_y, chip_width);
    engineChip(ui, "engines.visual", "图像", flags.visual.load(), 18.0f + (chip_width + 6.0f) * 2.0f, chips_y, chip_width);

    components::button(ui, "engines.release").position(18.0f, chips_y + 34.0f)
        .size(width - 36.0f, 32.0f).text("释放模型").icon(0xF1F8).fontSize(12.0f).iconSize(12.0f)
        .theme(studioTheme(), false).radius(10.0f)
        .disabled(state.busy || !flags.any())
        .onClick(submitReleaseEngines)
        .build();

    // Backend selector: applies to the next generation. Unavailable backends
    // (e.g. SYCL without its runtime) and changes while busy are refused with
    // a status message — the segmented control has no disabled state.
    sectionLabel(ui, "backend.label", "运行后端", 22.0f, height - 118.0f, width - 44.0f);
    ui.stack("backend.select.wrap").position(18.0f, height - 94.0f)
        .size(width - 36.0f, 34.0f)
        .content([&] {
            components::segmented(ui, "backend.select")
                .size(width - 36.0f, 34.0f)
                .items({"CUDA", "SYCL", "CPU"})
                .selected(state.backend)
                .fontSize(12.0f)
                .theme(studioTheme())
                .onChange([](int value) {
                    if (state.busy || value == state.backend) return;
                    if (!EngineService::backendAvailable(value)) {
                        state.has_error = true;
                        state.status = std::string(kBackendNames[value]) +
                                       " 后端不可用：未检测到设备或运行时";
                        return;
                    }
                    state.backend = value;
                    state.has_error = false;
                    state.status = std::string("后端已切换为 ") + kBackendNames[state.backend] +
                                   " · 下次生成时生效";
                })
                .build();
        })
        .build();

    std::string hint;
    if (state.backend == kBackendCpu) {
        hint = "全部在 CPU 运行";
    } else {
        hint = std::string("TTS/图像使用 ") + kBackendNames[state.backend] +
               " · LLM 自动选 GPU";
    }
    std::string missing;
    if (!EngineService::backendAvailable(kBackendCuda)) missing += " CUDA";
    if (!EngineService::backendAvailable(kBackendSycl)) missing += " SYCL";
    if (!missing.empty()) hint = "不可用:" + missing + " · " + hint;
    text(ui, "backend.hint", hint, 22.0f, height - 52.0f, width - 44.0f, 18.0f,
         kFontOverline, kFaint, 600);
}

inline void composeHeader(eui::Ui& ui, float x, float width) {
    text(ui, "page.title", toolTitle(), x, 22.0f, width - 280.0f, 38.0f, kFontTitle, kText, 800);
    text(ui, "page.subtitle", toolSubtitle(), x, 60.0f, width - 280.0f, 22.0f,
         kFontCaption + 1.0f, kMuted, 550);

    // Status pill, right-aligned.
    const eui::Color status_color = state.has_error ? kDanger : (state.busy ? kAccent : kMuted);
    const float pill_width = 236.0f;
    const float pill_x = x + width - pill_width;
    ui.rect("status.pill").position(pill_x, 30.0f).size(pill_width, 30.0f)
        .color(kSurface).radius(15.0f).border(1.0f, kBorderSoft).build();
    ui.rect("status.dot").position(pill_x + 12.0f, 41.0f).size(8.0f, 8.0f)
        .color(status_color).radius(4.0f).build();
    text(ui, "status.text", state.status, pill_x + 27.0f, 36.0f, pill_width - 38.0f, 18.0f,
         kFontCaption, status_color, 640);
}

} // namespace app
