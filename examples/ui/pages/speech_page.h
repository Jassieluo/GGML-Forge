#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace app {

inline void composeSpeech(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float left = width * 0.58f;
    panel(ui, "speech.input.panel", x, y, left, height);
    panel(ui, "speech.output.panel", x + left + gap, y, width - left - gap, height);

    // ---- Input panel ----
    sectionLabel(ui, "speech.input.section", "合成文本", x + 24.0f, y + 22.0f, left - 48.0f);
    text(ui, "speech.input.hint", "对话的最新回答会自动同步到这里", x + 24.0f, y + 44.0f,
         left - 48.0f, 20.0f, kFontCaption, kMuted);
    components::input(ui, "speech.input").position(x + 24.0f, y + 76.0f)
        .size(left - 48.0f, std::max(150.0f, height - 216.0f)).multiline().fontSize(kFontBody)
        .value(state.speech_text).placeholder("输入要合成的中文文本…").theme(studioTheme())
        .onChange([](const std::string& value) { state.speech_text = value; })
        .build();

    char speed[32] = {};
    std::snprintf(speed, sizeof(speed), "语速  %.1fx", state.speech_speed);
    text(ui, "speech.speed.label", speed, x + 24.0f, y + height - 114.0f, 120.0f, 24.0f,
         kFontSecondary, kTextSecondary, 620);
    ui.stack("speech.speed.wrap").position(x + 142.0f, y + height - 114.0f)
        .size(left - 166.0f, 26.0f).content([&] {
            components::slider(ui, "speech.speed")
                .size(left - 166.0f, 26.0f).value((state.speech_speed - 0.7f) / 0.8f).theme(studioTheme())
                .onChange([](float value) { state.speech_speed = 0.7f + value * 0.8f; })
                .build();
        })
        .build();
    components::button(ui, "speech.generate").position(x + 24.0f, y + height - 68.0f)
        .size(left - 48.0f, 46.0f).text(state.busy ? "正在合成…" : "生成并播放")
        .icon(0xF028).fontSize(14.0f).theme(studioTheme(), true).radius(12.0f).disabled(state.busy)
        .onClick(submitSpeech)
        .build();

    // ---- Output panel ----
    const float right_x = x + left + gap;
    const float right_width = width - left - gap;
    sectionLabel(ui, "speech.output.section", "本次输出", right_x + 24.0f, y + 22.0f,
                 right_width - 48.0f);
    if (state.busy && state.tool == Tool::Speech) {
        progressWithLabel(ui, "speech.progress", right_x + 24.0f, y + 96.0f,
                          right_width - 48.0f, state.status, state.progress.load());
    } else if (state.audio_path.empty()) {
        text(ui, "speech.empty.icon", "♪", right_x + 24.0f, y + height * 0.32f,
             right_width - 48.0f, 64.0f, 44.0f, kFaint, 500);
        text(ui, "speech.empty", "还没有生成语音", right_x + 24.0f, y + height * 0.32f + 72.0f,
             right_width - 48.0f, 24.0f, kFontSecondary, kMuted, 550);
    } else {
        char duration[64] = {};
        std::snprintf(duration, sizeof(duration), "%.1f 秒 · WAV · 豆包音色", state.audio_seconds);
        ui.rect("speech.result.card").position(right_x + 24.0f, y + 60.0f)
            .size(right_width - 48.0f, 118.0f).color(kSurfaceRaised).radius(12.0f)
            .border(1.0f, kBorderSoft).build();
        text(ui, "speech.file", filenameOnly(state.audio_path), right_x + 40.0f, y + 76.0f,
             right_width - 80.0f, 44.0f, kFontBody, kText, 680, true);
        text(ui, "speech.duration", duration, right_x + 40.0f, y + 124.0f,
             right_width - 80.0f, 22.0f, kFontCaption, kMuted);
        components::button(ui, "speech.play").position(right_x + 24.0f, y + 194.0f)
            .size(std::min(160.0f, right_width - 48.0f), 42.0f).text("重新播放").icon(0xF04B)
            .fontSize(13.0f).theme(studioTheme(), false).radius(11.0f)
            .onClick([] { EngineService::playAudio(state.audio_path); })
            .build();
    }
}

} // namespace app
