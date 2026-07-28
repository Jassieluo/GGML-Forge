#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"
#include "services/model_catalog.h"

#include <string>
#include <vector>

namespace app {

inline const char* toolTitle() {
    if (state.tool == Tool::Home) return "欢迎使用 GGML-Forge Studio";
    if (state.tool == Tool::Speech) return "语音合成";
    if (state.tool == Tool::Image) return "视觉生成";
    if (state.tool == Tool::Vision) return "视觉分析";
    return "本地对话";
}

inline const char* toolSubtitle() {
    if (state.tool == Tool::Home) return "本地生成与视觉感知工作台";
    if (state.tool == Tool::Speech) return "GPT-SoVITS · 文本转语音";
    if (state.tool == Tool::Image) return "stable-diffusion.cpp · 文本生成图片";
    if (state.tool == Tool::Vision) return "分类 · 检测 · 分割 · 姿态 · OBB · 深度";
    return "llama.cpp · 流式对话 · 支持工具调用";
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
            if (tool == Tool::Chat) preloadChatEngine();
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
    navButton(ui, "nav.home", "能力概览", 0xF015, Tool::Home, button_x, 96.0f, button_width, compact);
    navButton(ui, "nav.chat", "本地对话", 0xF075, Tool::Chat, button_x, 148.0f, button_width, compact);
    navButton(ui, "nav.speech", "语音合成", 0xF130, Tool::Speech, button_x, 200.0f, button_width, compact);
    navButton(ui, "nav.image", "图像生成", 0xF03E, Tool::Image, button_x, 252.0f, button_width, compact);
    navButton(ui, "nav.vision", "视觉分析", 0xF06E, Tool::Vision, button_x, 304.0f, button_width, compact);

    if (compact) return;

    const auto& flags = EngineService::loaded();
    sectionLabel(ui, "appearance.label", "外观", 22.0f, height - 224.0f, width - 44.0f);
    ui.stack("appearance.select.wrap").position(18.0f, height - 200.0f)
        .size(width - 36.0f, 34.0f).content([&] {
            components::segmented(ui, "appearance.select")
                .size(width - 36.0f, 34.0f).items({"深色", "浅色"})
                .selected(state.light_theme ? 1 : 0).fontSize(12.0f).theme(studioTheme())
                .onChange([](int value) {
                    state.light_theme = value == 1;
                    applyStudioPalette(state.light_theme);
                }).build();
        }).build();

    components::button(ui, "engines.release").position(18.0f, height - 154.0f)
        .size(width - 36.0f, 32.0f).text("释放已加载模型").icon(0xF1F8).fontSize(12.0f).iconSize(12.0f)
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
                    if (state.tool == Tool::Chat) preloadChatEngine();
                })
                .build();
        })
        .build();

    std::string hint;
    if (state.backend == kBackendCpu) {
        hint = "全部在 CPU 运行";
    } else {
        hint = std::string("所有新任务固定使用 ") + kBackendNames[state.backend];
    }
    std::string missing;
    if (!EngineService::backendAvailable(kBackendCuda)) missing += " CUDA";
    if (!EngineService::backendAvailable(kBackendSycl)) missing += " SYCL";
    if (!missing.empty()) hint = "不可用:" + missing + " · " + hint;
    text(ui, "backend.hint", hint, 22.0f, height - 52.0f, width - 44.0f, 18.0f,
         kFontOverline, kFaint, 600);
}

inline void composeHeader(eui::Ui& ui, float x, float width) {
    const bool has_model = state.tool != Tool::Home;
    text(ui, "page.title", toolTitle(), x, 20.0f, width - (has_model ? 500.0f : 220.0f), 38.0f, kFontTitle, kText, 800);
    text(ui, "page.subtitle", toolSubtitle(), x, 58.0f, width - (has_model ? 500.0f : 220.0f), 22.0f,
         kFontCaption + 1.0f, kMuted, 550);

    if (has_model) {
        std::vector<const ModelEntry*> models;
        std::string* selected_path = nullptr;
        if (state.tool == Tool::Chat) { models = modelCatalog().models(ModelKind::Llm); selected_path = &state.llm_model; }
        else if (state.tool == Tool::Speech) { models = modelCatalog().models(ModelKind::Tts); selected_path = &state.tts_model; }
        else if (state.tool == Tool::Image) { models = modelCatalog().models(ModelKind::ImageGeneration); selected_path = &state.image_model; }
        else { models = modelCatalog().visionModels(state.vision_task); selected_path = &state.vision_model; }
        if (selected_path && selected_path->empty() && !models.empty()) {
            if (state.tool == Tool::Chat) *selected_path = firstModel(ModelKind::Llm);
            else if (state.tool == Tool::Speech) *selected_path = firstModel(ModelKind::Tts);
            else if (state.tool == Tool::Image) *selected_path = firstModel(ModelKind::ImageGeneration);
            else *selected_path = firstVisionModel(state.vision_task);
        }
        std::vector<std::string> labels;
        int selected = 0;
        for (size_t i = 0; i < models.size(); ++i) {
            labels.push_back(models[i]->name + "  ·  " + models[i]->detail);
            if (selected_path && models[i]->path == *selected_path) selected = static_cast<int>(i);
        }
        if (labels.empty()) labels.push_back("未发现可用模型");
        const float model_width = 286.0f;
        const float model_x = x + width - model_width - 188.0f;
        ui.stack("header.model.wrap").position(model_x, 28.0f).size(model_width, 38.0f).zIndex(1000).content([&] {
            components::dropdown(ui, "header.model")
                .size(model_width, 38.0f).items(labels).selected(selected)
                .zIndex(1000)
                .open(state.model_dropdown_open.get()).theme(studioTheme())
                .onOpenChange([](bool open){ state.model_dropdown_open.set(open); })
                .onChange([models, selected_path](int index){
                    if (!selected_path || index < 0 || index >= static_cast<int>(models.size()) || state.busy) return;
                    *selected_path = models[static_cast<size_t>(index)]->path;
                    state.model_dropdown_open.set(false);
                    state.status = "模型已切换 · 下次运行时加载";
                    state.has_error = false;
                    if (state.tool == Tool::Chat) preloadChatEngine();
                }).build();
        }).build();
    }

    // Status pill, right-aligned.
    const eui::Color status_color = state.has_error ? kDanger : (state.busy ? kAccent : kMuted);
    const float pill_width = 172.0f;
    const float pill_x = x + width - pill_width;
    ui.rect("status.pill").position(pill_x, 30.0f).size(pill_width, 30.0f)
        .color(kSurface).radius(15.0f).border(1.0f, kBorderSoft).build();
    ui.rect("status.dot").position(pill_x + 12.0f, 41.0f).size(8.0f, 8.0f)
        .color(status_color).radius(4.0f).build();
    text(ui, "status.text", state.status, pill_x + 27.0f, 36.0f, pill_width - 38.0f, 18.0f,
         kFontCaption, status_color, 640);
}

} // namespace app
