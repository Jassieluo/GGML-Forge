#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"
#include "services/model_catalog.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace app {

inline void composeTts(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float left = width * 0.58f;
    const float right_x = x + left + gap;
    const float right_width = width - left - gap;
    panel(ui, "speech.input.panel", x, y, left, height);
    panel(ui, "speech.output.panel", right_x, y, right_width, height);

    sectionLabel(ui, "speech.input.section", tr("合成文本", "SYNTHESIS TEXT"), x + 24.0f, y + 22.0f, left - 48.0f);
    const float text_height = std::clamp(height * 0.27f, 116.0f, 184.0f);
    components::input(ui, "speech.input").position(x + 24.0f, y + 56.0f)
        .size(left - 48.0f, text_height).multiline().fontSize(kFontBody)
        .value(state.speech_text).placeholder(tr("输入要合成的中文文本…", "Enter text to synthesize…")).theme(studioTheme())
        .onChange([](const std::string& value) { state.speech_text = value; })
        .build();

    const float voice_label_y = y + 72.0f + text_height;
    sectionLabel(ui, "speech.voice.section", tr("参考声音", "REFERENCE VOICE"), x + 24.0f, voice_label_y, left - 48.0f);
    const auto& voices = voiceCatalog();
    std::vector<std::string> voice_labels;
    int selected_voice = -1;
    for (size_t i = 0; i < voices.size(); ++i) {
        voice_labels.push_back(voices[i].name);
        if (voices[i].path == resolveProjectPath(state.tts_voice_path)) {
            selected_voice = static_cast<int>(i);
        }
    }
    if (selected_voice < 0 && !state.tts_voice_path.empty()) {
        voice_labels.push_back(tr("自定义 · ", "Custom · ") + filenameOnly(state.tts_voice_path));
        selected_voice = static_cast<int>(voice_labels.size()) - 1;
    }
    if (voice_labels.empty()) voice_labels.push_back(tr("未发现预置声音", "No preset voices found"));
    const float import_width = 132.0f;
    const float voice_y = voice_label_y + 26.0f;
    ui.stack("speech.voice.dropdown.wrap").position(x + 24.0f, voice_y)
        .size(left - 48.0f - import_width - 10.0f, 40.0f).zIndex(600).content([&] {
            components::dropdown(ui, "speech.voice.dropdown")
                .size(left - 48.0f - import_width - 10.0f, 40.0f)
                .items(voice_labels).selected(std::max(0, selected_voice)).itemHeight(32.0f)
                .open(state.voice_dropdown_open.get()).zIndex(600).theme(studioTheme())
                .onOpenChange([](bool open) { state.voice_dropdown_open.set(open); })
                .onChange([](int index) { selectTtsVoice(index); }).build();
        }).build();
    components::button(ui, "speech.voice.import").position(x + left - 24.0f - import_width, voice_y)
        .size(import_width, 40.0f).text(tr("导入 WAV", "Import WAV")).icon(0xF07C).fontSize(12.5f)
        .theme(studioTheme(), false).radius(10.0f).disabled(state.busy)
        .onClick(chooseTtsReference).build();

    const float transcript_y = voice_y + 56.0f;
    text(ui, "speech.reference.hint", tr("参考文字（必须与 WAV 中实际说出的内容一致）", "Transcript (must exactly match the reference WAV)"),
         x + 24.0f, transcript_y, left - 48.0f, 20.0f, kFontCaption, kMuted, 560);
    const float transcript_height = std::max(54.0f, y + height - 142.0f - (transcript_y + 24.0f));
    components::input(ui, "speech.reference.text").position(x + 24.0f, transcript_y + 24.0f)
        .size(left - 48.0f, transcript_height).multiline().fontSize(kFontSecondary)
        .value(state.tts_reference_text).placeholder(tr("导入新 WAV 后，请填写它的准确台词…", "Enter the exact words spoken in the imported WAV…"))
        .theme(studioTheme())
        .onChange([](const std::string& value) { state.tts_reference_text = value; })
        .build();

    char speed[32] = {};
    std::snprintf(speed, sizeof(speed), tr("语速  %.1fx", "Speed  %.1fx"), state.speech_speed);
    text(ui, "speech.speed.label", speed, x + 24.0f, y + height - 114.0f, 120.0f, 24.0f,
         kFontSecondary, kTextSecondary, 620);
    ui.stack("speech.speed.wrap").position(x + 142.0f, y + height - 114.0f)
        .size(left - 166.0f, 26.0f).content([&] {
            components::slider(ui, "speech.speed")
                .size(left - 166.0f, 26.0f).value((state.speech_speed - 0.7f) / 0.8f)
                .theme(studioTheme())
                .onChange([](float value) { state.speech_speed = 0.7f + value * 0.8f; })
                .build();
        }).build();
    components::button(ui, "speech.generate").position(x + 24.0f, y + height - 68.0f)
        .size(left - 48.0f, 46.0f).text(state.busy ? tr("正在合成…", "Synthesizing…") : tr("生成并播放", "Generate and play"))
        .icon(0xF028).fontSize(14.0f).theme(studioTheme(), true).radius(12.0f)
        .disabled(state.busy).onClick(submitSpeech).build();

    sectionLabel(ui, "speech.output.section", tr("本次输出", "OUTPUT"), right_x + 24.0f, y + 22.0f,
                 right_width - 48.0f);
    if (state.busy && state.speech_mode == SpeechMode::Synthesis) {
        progressWithLabel(ui, "speech.progress", right_x + 24.0f, y + 96.0f,
                          right_width - 48.0f, state.status, state.progress.load());
    } else if (state.audio_path.empty()) {
        text(ui, "speech.empty.icon", "♪", right_x + 24.0f, y + height * 0.32f,
             right_width - 48.0f, 64.0f, 44.0f, kFaint, 500);
        text(ui, "speech.empty", tr("还没有生成语音", "No speech generated yet"), right_x + 24.0f, y + height * 0.32f + 72.0f,
             right_width - 48.0f, 24.0f, kFontSecondary, kMuted, 550);
    } else {
        char duration[64] = {};
        std::snprintf(duration, sizeof(duration), tr("%.1f 秒 · WAV", "%.1f sec · WAV"), state.audio_seconds);
        ui.rect("speech.result.card").position(right_x + 24.0f, y + 60.0f)
            .size(right_width - 48.0f, 118.0f).color(kSurfaceRaised).radius(12.0f)
            .border(1.0f, kBorderSoft).build();
        text(ui, "speech.file", filenameOnly(state.audio_path), right_x + 40.0f, y + 76.0f,
             right_width - 80.0f, 44.0f, kFontBody, kText, 680, true);
        text(ui, "speech.duration", duration, right_x + 40.0f, y + 124.0f,
             right_width - 80.0f, 22.0f, kFontCaption, kMuted);
        components::button(ui, "speech.play").position(right_x + 24.0f, y + 194.0f)
            .size(std::min(160.0f, right_width - 48.0f), 42.0f).text(tr("重新播放", "Play again")).icon(0xF04B)
            .fontSize(13.0f).theme(studioTheme(), false).radius(11.0f)
            .onClick([] { EngineService::playAudio(state.audio_path); }).build();
    }
}

inline void composeAsr(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float left = width * 0.42f;
    const float right_x = x + left + gap;
    const float right_width = width - left - gap;
    panel(ui, "asr.input.panel", x, y, left, height);
    panel(ui, "asr.output.panel", right_x, y, right_width, height);

    sectionLabel(ui, "asr.input.section", tr("输入音频", "INPUT AUDIO"), x + 24.0f, y + 22.0f, left - 48.0f);
    ui.rect("asr.audio.card").position(x + 24.0f, y + 58.0f).size(left - 48.0f, 96.0f)
        .color(kSurfaceRaised).radius(12.0f).border(1.0f, kBorderSoft).build();
    text(ui, "asr.audio.name",
         state.asr_audio_path.empty() ? tr("请选择一段 WAV 音频", "Choose a WAV audio file") : filenameOnly(state.asr_audio_path),
         x + 42.0f, y + 72.0f, left - 84.0f, 28.0f, kFontBody, kText, 650, true);
    text(ui, "asr.audio.hint", tr("支持 PCM/Float WAV · 当前演示固定使用 CPU 稳定路径", "PCM/Float WAV · This demo uses the stable CPU path"),
         x + 42.0f, y + 112.0f, left - 84.0f, 20.0f, kFontCaption, kMuted);

    components::button(ui, "asr.audio.choose").position(x + 24.0f, y + 172.0f)
        .size(left - 48.0f, 42.0f).text(state.asr_audio_path.empty() ? tr("选择 WAV", "Choose WAV") : tr("更换 WAV", "Change WAV"))
        .icon(0xF07C).fontSize(13.0f).theme(studioTheme(), false).radius(11.0f)
        .disabled(state.busy).onClick(chooseAsrAudio).build();
    if (!state.asr_audio_path.empty()) {
        components::button(ui, "asr.audio.play").position(x + 24.0f, y + 224.0f)
            .size(left - 48.0f, 38.0f).text(tr("播放原音", "Play source")).icon(0xF04B).fontSize(12.5f)
            .theme(studioTheme(), false).radius(10.0f)
            .onClick([] { EngineService::playAudio(state.asr_audio_path); }).build();
    }

    sectionLabel(ui, "asr.language.section", tr("音频语言", "AUDIO LANGUAGE"), x + 24.0f, y + 286.0f, left - 48.0f);
    int language = state.asr_language == "zh" ? 1 : state.asr_language == "en" ? 2 : 0;
    ui.stack("asr.language.wrap").position(x + 24.0f, y + 314.0f).size(left - 48.0f, 36.0f)
        .content([&] {
            components::segmented(ui, "asr.language").size(left - 48.0f, 36.0f)
                .items({tr("自动", "Auto"), tr("中文", "Chinese"), tr("英文", "English")}).selected(language).fontSize(12.0f)
                .theme(studioTheme()).onChange([](int value) {
                    state.asr_language = value == 1 ? "zh" : value == 2 ? "en" : "auto";
                }).build();
        }).build();
    sectionLabel(ui, "asr.task.section", tr("任务", "TASK"), x + 24.0f, y + 370.0f, left - 48.0f);
    ui.stack("asr.task.wrap").position(x + 24.0f, y + 398.0f).size(left - 48.0f, 36.0f)
        .content([&] {
            components::segmented(ui, "asr.task").size(left - 48.0f, 36.0f)
                .items({tr("原文转写", "Transcribe"), tr("翻译为英文", "Translate to English")}).selected(state.asr_translate ? 1 : 0)
                .fontSize(12.0f).theme(studioTheme())
                .onChange([](int value) { state.asr_translate = value == 1; }).build();
        }).build();
    components::button(ui, "asr.run").position(x + 24.0f, y + height - 68.0f)
        .size(left - 48.0f, 46.0f).text(state.busy ? tr("正在识别…", "Transcribing…") : tr("开始识别", "Start transcription"))
        .icon(0xF130).fontSize(14.0f).theme(studioTheme(), true).radius(12.0f)
        .disabled(state.busy || state.asr_audio_path.empty()).onClick(submitAsr).build();

    sectionLabel(ui, "asr.output.section", tr("识别结果", "TRANSCRIPT"), right_x + 24.0f, y + 22.0f,
                 right_width - 48.0f);
    const std::string language_text = state.asr_detected_language.empty()
        ? tr("Whisper · 自动语言检测", "Whisper · Automatic language detection")
        : tr("检测语言 · ", "Detected language · ") + state.asr_detected_language;
    text(ui, "asr.output.language", language_text, right_x + 24.0f, y + 48.0f,
         right_width - 48.0f, 20.0f, kFontCaption, kMuted);
    if (state.busy && state.speech_mode == SpeechMode::Recognition) {
        progressWithLabel(ui, "asr.progress", right_x + 24.0f, y + 96.0f,
                          right_width - 48.0f, state.status, state.progress.load());
    } else {
        components::input(ui, "asr.output.text").position(right_x + 24.0f, y + 82.0f)
            .size(right_width - 48.0f, height - 106.0f).multiline().fontSize(kFontBody)
            .value(state.asr_transcript).placeholder(tr("识别完成后，文字会显示在这里…", "The transcript will appear here…"))
            .theme(studioTheme())
            .onChange([](const std::string& value) { state.asr_transcript = value; }).build();
    }
}

inline void composeSpeech(eui::Ui& ui, float x, float y, float width, float height) {
    const float switch_width = std::min(360.0f, width);
    ui.stack("speech.mode.wrap").position(x, y).size(switch_width, 38.0f).content([&] {
        components::segmented(ui, "speech.mode").size(switch_width, 38.0f)
            .items({tr("语音合成", "Synthesis"), tr("Whisper 识别", "Whisper ASR")})
            .selected(state.speech_mode == SpeechMode::Recognition ? 1 : 0)
            .fontSize(13.0f).theme(studioTheme()).onChange([](int value) {
                if (state.busy) return;
                state.speech_mode = value == 1 ? SpeechMode::Recognition : SpeechMode::Synthesis;
                state.model_dropdown_open.set(false);
                state.has_error = false;
                state.status = tr("就绪", "Ready");
            }).build();
    }).build();
    const float content_y = y + 52.0f;
    const float content_height = height - 52.0f;
    if (state.speech_mode == SpeechMode::Recognition) {
        composeAsr(ui, x, content_y, width, content_height);
    } else {
        composeTts(ui, x, content_y, width, content_height);
    }
}

} // namespace app
