#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"

#include <algorithm>
#include <string>

namespace app {

inline void composeImage(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float controls = std::max(300.0f, width * 0.36f);
    panel(ui, "image.controls.panel", x, y, controls, height);
    panel(ui, "image.preview.panel", x + controls + gap, y, width - controls - gap, height);

    // ---- Controls ----
    sectionLabel(ui, "image.prompt.section", "图像描述", x + 24.0f, y + 22.0f, controls - 48.0f);
    text(ui, "image.prompt.hint", "建议使用英文提示词以获得稳定效果", x + 24.0f, y + 44.0f,
         controls - 48.0f, 20.0f, kFontCaption, kMuted);
    components::input(ui, "image.prompt").position(x + 24.0f, y + 76.0f)
        .size(controls - 48.0f, 152.0f).multiline().fontSize(kFontSecondary + 1.0f)
        .value(state.image_prompt).placeholder("Describe the image…").theme(studioTheme())
        .onChange([](const std::string& value) { state.image_prompt = value; })
        .build();

    sectionLabel(ui, "image.params.section", "参数", x + 24.0f, y + 252.0f, controls - 48.0f);
    text(ui, "image.size.label", "尺寸", x + 24.0f, y + 278.0f, 100.0f, 22.0f,
         kFontSecondary, kTextSecondary, 600);
    text(ui, "image.size.value", "512 × 512", x + controls - 140.0f, y + 278.0f,
         116.0f, 22.0f, kFontSecondary, kText, 650);
    text(ui, "image.steps.label", "采样步数", x + 24.0f, y + 314.0f, 100.0f, 22.0f,
         kFontSecondary, kTextSecondary, 600);
    components::button(ui, "image.steps.down").position(x + controls - 140.0f, y + 306.0f)
        .size(34.0f, 34.0f).text("−").theme(studioTheme(), false).radius(9.0f).disabled(state.busy)
        .onClick([] { state.image_steps = std::max(4, state.image_steps - 4); })
        .build();
    text(ui, "image.steps.value", std::to_string(state.image_steps), x + controls - 100.0f, y + 312.0f,
         52.0f, 24.0f, 14.0f, kText, 720);
    components::button(ui, "image.steps.up").position(x + controls - 58.0f, y + 306.0f)
        .size(34.0f, 34.0f).text("+").theme(studioTheme(), false).radius(9.0f).disabled(state.busy)
        .onClick([] { state.image_steps = std::min(50, state.image_steps + 4); })
        .build();

    const bool generating = state.busy && state.tool == Tool::Image;
    components::button(ui, "image.generate").position(x + 24.0f, y + height - 68.0f)
        .size(controls - 48.0f, 46.0f)
        .text(generating ? "停止" : "生成图片")
        .icon(generating ? 0xF04D : 0xF1FC).fontSize(14.0f)
        .theme(studioTheme(), true).radius(12.0f)
        .disabled(state.busy && !generating)
        .onClick([generating] { generating ? requestCancel() : submitImage(); })
        .build();

    // ---- Preview ----
    const float preview_x = x + controls + gap;
    const float preview_width = width - controls - gap;
    if (generating) {
        progressWithLabel(ui, "image.progress", preview_x + 24.0f, y + height * 0.46f,
                          preview_width - 48.0f, state.status, state.progress.load());
    } else if (state.image_path.empty()) {
        text(ui, "image.empty.icon", "□", preview_x + 24.0f, y + height * 0.34f,
             preview_width - 48.0f, 64.0f, 44.0f, kFaint, 400);
        text(ui, "image.empty", "生成结果会显示在这里", preview_x + 24.0f, y + height * 0.34f + 72.0f,
             preview_width - 48.0f, 26.0f, kFontSecondary, kMuted, 550);
    } else {
        const float image_size = std::min(preview_width - 48.0f, height - 84.0f);
        components::image(ui, "image.result")
            .position(preview_x + (preview_width - image_size) * 0.5f, y + 20.0f)
            .size(image_size, image_size).source(state.image_path).contain().radius(12.0f)
            .build();
        text(ui, "image.filename", filenameOnly(state.image_path), preview_x + 24.0f, y + height - 40.0f,
             preview_width - 48.0f, 20.0f, kFontCaption, kMuted);
    }
}

} // namespace app
