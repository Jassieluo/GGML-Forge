#include "eui_neo.h"
#include "core/platform/async.h"

#include "pages/state.h"
#include "services/engine_service.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace app {

const DslAppConfig& dslAppConfig() {
    static const DslAppConfig config = [] {
        auto value = DslAppConfig{}
            .title("GGML-Forge Studio")
            .pageId("ggml_forge_studio")
            .clearColor({0.035f, 0.039f, 0.047f, 1.0f})
            .windowSize(1240, 820)
            .fps(60.0);
#ifdef _WIN32
        value.fonts("C:/Windows/Fonts/msyh.ttc");
#endif
        return value;
    }();
    return config;
}

namespace {

StudioState state;

constexpr eui::Color kBackground{0.035f, 0.039f, 0.047f, 1.0f};
constexpr eui::Color kSidebar{0.052f, 0.057f, 0.067f, 1.0f};
constexpr eui::Color kSurface{0.068f, 0.074f, 0.086f, 1.0f};
constexpr eui::Color kSurfaceRaised{0.085f, 0.092f, 0.108f, 1.0f};
constexpr eui::Color kBorder{0.145f, 0.157f, 0.180f, 1.0f};
constexpr eui::Color kText{0.93f, 0.94f, 0.96f, 1.0f};
constexpr eui::Color kMuted{0.57f, 0.60f, 0.66f, 1.0f};
constexpr eui::Color kAccent{0.31f, 0.78f, 0.67f, 1.0f};
constexpr eui::Color kDanger{0.93f, 0.44f, 0.42f, 1.0f};

components::theme::ThemeColorTokens studioTheme() {
    auto theme = components::theme::dark();
    theme.background = kBackground;
    theme.surface = kSurface;
    theme.border = kBorder;
    theme.text = kText;
    theme.primary = kAccent;
    return theme;
}

const char* toolTitle() {
    if (state.tool == Tool::Speech) return "语音合成";
    if (state.tool == Tool::Image) return "视觉生成";
    return "本地对话";
}

const char* toolSubtitle() {
    if (state.tool == Tool::Speech) return "GPT-SoVITS · 文本转语音";
    if (state.tool == Tool::Image) return "stable-diffusion.cpp · 文本生成图片";
    return "llama.cpp · Qwen3.5 4B";
}

std::string filenameOnly(const std::string& path) {
    return path.empty() ? std::string{} : std::filesystem::u8path(path).filename().string();
}

void text(eui::Ui& ui, const std::string& id, const std::string& value,
          float x, float y, float width, float height, float size,
          eui::Color color = kText, int weight = 500, bool wrap = false) {
    ui.text(id).position(x, y).size(width, height).text(value)
        .fontSize(size).lineHeight(size * 1.45f).fontWeight(weight)
        .wrap(wrap).color(color).build();
}

void panel(eui::Ui& ui, const std::string& id, float x, float y, float width, float height,
           eui::Color color = kSurface) {
    ui.rect(id).position(x, y).size(width, height).color(color)
        .radius(14.0f).border(1.0f, kBorder).build();
}

void navButton(eui::Ui& ui, const std::string& id, const std::string& label,
               unsigned int icon, Tool tool, float x, float y, float width, bool compact) {
    const bool selected = state.tool == tool;
    components::ButtonStyle style(studioTheme(), selected);
    style.normal = selected ? eui::Color{0.12f, 0.24f, 0.22f, 1.0f} : kSidebar;
    style.hover = selected ? eui::Color{0.14f, 0.28f, 0.25f, 1.0f} : kSurface;
    style.pressed = kSurfaceRaised;
    style.text = selected ? eui::Color{0.72f, 1.0f, 0.92f, 1.0f} : kMuted;
    style.icon = selected ? kAccent : kMuted;
    style.border = {selected ? 1.0f : 0.0f, selected ? eui::Color{0.22f, 0.52f, 0.45f, 1.0f} : kSidebar};
    style.radius = 11.0f;
    components::button(ui, id).position(x, y).size(width, 46.0f)
        .text(compact ? "" : label).icon(icon).fontSize(14.0f).iconSize(17.0f)
        .style(style).disabled(state.busy)
        .onClick([tool] { state.tool = tool; state.status = "Ready"; }).build();
}

void beginTask(const std::string& status) {
    state.busy = true;
    state.has_error = false;
    state.progress.store(0.02f);
    state.status = status;
}

void finishTaskError(const std::string& error) {
    state.busy = false;
    state.has_error = true;
    state.progress.store(0.0f);
    state.status = error;
}

void submitChat() {
    if (state.busy || state.chat_input.empty()) return;
    state.messages.push_back({"user", state.chat_input});
    state.chat_input.clear();
    const auto history = state.messages;
    const int backend = state.backend;
    beginTask("正在生成回答…");
    core::async::restart("forge.ui.chat",
        [history, backend] { return EngineService::chat(history, backend, &state.progress); },
        [](core::async::Result<LlmResult> result) {
            if (!result.ok) return finishTaskError(result.error);
            state.messages.push_back({"assistant", std::move(result.value.text)});
            state.speech_text = state.messages.back().text;
            state.busy = false;
            state.has_error = false;
            state.status = "回答完成 · 已同步到语音合成";
        });
}

void submitSpeech() {
    if (state.busy || state.speech_text.empty()) return;
    const std::string input = state.speech_text;
    const float speed = state.speech_speed;
    const int backend = state.backend;
    beginTask("正在合成语音…");
    core::async::restart("forge.ui.speech",
        [input, speed, backend] { return EngineService::synthesize(input, speed, backend, &state.progress); },
        [](core::async::Result<SpeechResult> result) {
            if (!result.ok) return finishTaskError(result.error);
            state.audio_path = std::move(result.value.path);
            state.audio_seconds = result.value.seconds;
            state.busy = false;
            state.has_error = false;
            state.status = "语音已生成";
            EngineService::playAudio(state.audio_path);
        });
}

void submitImage() {
    if (state.busy || state.image_prompt.empty()) return;
    const std::string prompt = state.image_prompt;
    const int steps = state.image_steps;
    const int backend = state.backend;
    beginTask("正在生成图片…");
    core::async::restart("forge.ui.image",
        [prompt, steps, backend] { return EngineService::generateImage(prompt, steps, backend, &state.progress); },
        [](core::async::Result<ImageResult> result) {
            if (!result.ok) return finishTaskError(result.error);
            state.image_path = std::move(result.value.path);
            state.image_width = result.value.width;
            state.image_height = result.value.height;
            state.busy = false;
            state.has_error = false;
            state.status = "图片已生成";
        });
}

void composeSidebar(eui::Ui& ui, float width, float height, bool compact) {
    ui.rect("sidebar.background").size(width, height).color(kSidebar)
        .border(1.0f, kBorder).build();
    text(ui, "brand.mark", "F", 24.0f, 24.0f, 34.0f, 34.0f, 23.0f, kAccent, 850);
    if (!compact) {
        text(ui, "brand.name", "GGML-Forge", 58.0f, 23.0f, width - 76.0f, 28.0f, 19.0f, kText, 760);
        text(ui, "brand.caption", "LOCAL AI STUDIO", 58.0f, 49.0f, width - 76.0f, 18.0f, 10.0f, kMuted, 650);
    }
    const float button_x = compact ? 14.0f : 18.0f;
    const float button_width = width - button_x * 2.0f;
    navButton(ui, "nav.chat", "本地对话", 0xF075, Tool::Chat, button_x, 108.0f, button_width, compact);
    navButton(ui, "nav.speech", "语音合成", 0xF130, Tool::Speech, button_x, 164.0f, button_width, compact);
    navButton(ui, "nav.image", "视觉生成", 0xF03E, Tool::Image, button_x, 220.0f, button_width, compact);

    if (!compact) {
        text(ui, "backend.label", "运行后端", 22.0f, height - 150.0f, width - 44.0f, 22.0f, 12.0f, kMuted, 650);
        panel(ui, "backend.card", 18.0f, height - 118.0f, width - 36.0f, 76.0f, kSurface);
        text(ui, "backend.value", kBackendNames[state.backend], 34.0f, height - 103.0f,
             width - 116.0f, 26.0f, 16.0f, kText, 720);
        text(ui, "backend.hint", "点击切换设备", 34.0f, height - 77.0f,
             width - 116.0f, 20.0f, 11.0f, kMuted);
        components::button(ui, "backend.switch").position(width - 76.0f, height - 100.0f)
            .size(42.0f, 42.0f).icon(0xF021).text("").theme(studioTheme(), false)
            .radius(10.0f).disabled(state.busy)
            .onClick([] { state.backend = (state.backend + 1) % 3; state.status = "后端已切换"; state.has_error = false; }).build();
    }
}

void composeHeader(eui::Ui& ui, float x, float width) {
    text(ui, "page.title", toolTitle(), x, 24.0f, width - 260.0f, 38.0f, 28.0f, kText, 800);
    text(ui, "page.subtitle", toolSubtitle(), x, 62.0f, width - 260.0f, 24.0f, 13.0f, kMuted);
    const eui::Color status_color = state.has_error ? kDanger : (state.busy ? kAccent : kMuted);
    ui.rect("status.dot").position(x + width - 222.0f, 39.0f).size(8.0f, 8.0f)
        .color(status_color).radius(4.0f).build();
    text(ui, "status.text", state.status, x + width - 204.0f, 30.0f, 204.0f, 28.0f,
         12.0f, status_color, 600);
}

void composeChat(eui::Ui& ui, float x, float y, float width, float height) {
    panel(ui, "chat.panel", x, y, width, height);
    const float composer_height = 86.0f;
    const float messages_height = height - composer_height - 24.0f;
    const int visible = std::max(1, static_cast<int>(messages_height / 92.0f));
    const size_t start = state.messages.size() > static_cast<size_t>(visible)
        ? state.messages.size() - static_cast<size_t>(visible) : 0;
    float message_y = y + 18.0f;
    for (size_t index = start; index < state.messages.size(); ++index) {
        const auto& message = state.messages[index];
        const bool user = message.role == "user";
        const float bubble_width = width * 0.72f;
        const float bubble_x = user ? x + width - bubble_width - 20.0f : x + 20.0f;
        ui.rect("message.bg." + std::to_string(index)).position(bubble_x, message_y)
            .size(bubble_width, 76.0f)
            .color(user ? eui::Color{0.105f, 0.19f, 0.18f, 1.0f} : kSurfaceRaised)
            .radius(12.0f).border(1.0f, user ? eui::Color{0.18f, 0.38f, 0.34f, 1.0f} : kBorder).build();
        text(ui, "message.role." + std::to_string(index), user ? "YOU" : "FORGE",
             bubble_x + 16.0f, message_y + 9.0f, bubble_width - 32.0f, 16.0f, 10.0f,
             user ? kAccent : kMuted, 750);
        text(ui, "message.text." + std::to_string(index), message.text,
             bubble_x + 16.0f, message_y + 28.0f, bubble_width - 32.0f, 42.0f, 13.0f,
             kText, 450, true);
        message_y += 88.0f;
    }
    ui.rect("chat.composer.line").position(x + 20.0f, y + height - composer_height)
        .size(width - 40.0f, 1.0f).color(kBorder).build();
    const float input_width = width - 148.0f;
    components::input(ui, "chat.input").position(x + 20.0f, y + height - 64.0f)
        .size(input_width, 46.0f).value(state.chat_input).placeholder("输入消息，按 Enter 发送")
        .theme(studioTheme()).fontSize(14.0f).onChange([](const std::string& value) { state.chat_input = value; })
        .onEnter(submitChat).build();
    components::button(ui, "chat.send").position(x + width - 116.0f, y + height - 64.0f)
        .size(96.0f, 46.0f).text(state.busy ? "生成中" : "发送").icon(0xF1D8)
        .theme(studioTheme(), true).radius(10.0f).disabled(state.busy).onClick(submitChat).build();
}

void composeSpeech(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float left = width * 0.58f;
    panel(ui, "speech.input.panel", x, y, left, height);
    panel(ui, "speech.output.panel", x + left + gap, y, width - left - gap, height);
    text(ui, "speech.input.title", "合成文本", x + 24.0f, y + 22.0f, left - 48.0f, 28.0f, 18.0f, kText, 730);
    text(ui, "speech.input.hint", "LLM 的最近回答会自动同步到这里", x + 24.0f, y + 53.0f,
         left - 48.0f, 22.0f, 12.0f, kMuted);
    components::input(ui, "speech.input").position(x + 24.0f, y + 88.0f)
        .size(left - 48.0f, std::max(150.0f, height - 224.0f)).multiline().fontSize(15.0f)
        .value(state.speech_text).placeholder("输入要合成的中文文本…").theme(studioTheme())
        .onChange([](const std::string& value) { state.speech_text = value; }).build();
    char speed[32] = {};
    std::snprintf(speed, sizeof(speed), "语速  %.1fx", state.speech_speed);
    text(ui, "speech.speed.label", speed, x + 24.0f, y + height - 112.0f, 120.0f, 24.0f, 13.0f, kMuted, 600);
    ui.stack("speech.speed.wrap").position(x + 142.0f, y + height - 112.0f)
        .size(left - 166.0f, 26.0f).content([&] {
            components::slider(ui, "speech.speed")
                .size(left - 166.0f, 26.0f).value((state.speech_speed - 0.7f) / 0.8f).theme(studioTheme())
                .onChange([](float value) { state.speech_speed = 0.7f + value * 0.8f; }).build();
        }).build();
    components::button(ui, "speech.generate").position(x + 24.0f, y + height - 66.0f)
        .size(left - 48.0f, 44.0f).text(state.busy ? "正在合成…" : "生成并播放")
        .icon(0xF028).theme(studioTheme(), true).radius(10.0f).disabled(state.busy).onClick(submitSpeech).build();

    const float right_x = x + left + gap;
    const float right_width = width - left - gap;
    text(ui, "speech.output.title", "本次输出", right_x + 24.0f, y + 22.0f,
         right_width - 48.0f, 28.0f, 18.0f, kText, 730);
    if (state.audio_path.empty()) {
        text(ui, "speech.empty.icon", "♪", right_x + 24.0f, y + 118.0f,
             right_width - 48.0f, 72.0f, 48.0f, kMuted, 500);
        text(ui, "speech.empty", "还没有生成语音", right_x + 24.0f, y + 196.0f,
             right_width - 48.0f, 26.0f, 14.0f, kMuted, 550);
    } else {
        char duration[64] = {};
        std::snprintf(duration, sizeof(duration), "%.1f 秒 · WAV", state.audio_seconds);
        text(ui, "speech.file", filenameOnly(state.audio_path), right_x + 24.0f, y + 92.0f,
             right_width - 48.0f, 52.0f, 15.0f, kText, 650, true);
        text(ui, "speech.duration", duration, right_x + 24.0f, y + 150.0f,
             right_width - 48.0f, 24.0f, 12.0f, kMuted);
        components::button(ui, "speech.play").position(right_x + 24.0f, y + 194.0f)
            .size(std::min(160.0f, right_width - 48.0f), 42.0f).text("重新播放").icon(0xF04B)
            .theme(studioTheme(), false).radius(10.0f)
            .onClick([] { EngineService::playAudio(state.audio_path); }).build();
    }
}

void composeImage(eui::Ui& ui, float x, float y, float width, float height) {
    const float gap = 18.0f;
    const float controls = std::max(300.0f, width * 0.36f);
    panel(ui, "image.controls.panel", x, y, controls, height);
    panel(ui, "image.preview.panel", x + controls + gap, y, width - controls - gap, height);
    text(ui, "image.prompt.title", "图像描述", x + 24.0f, y + 22.0f,
         controls - 48.0f, 28.0f, 18.0f, kText, 730);
    text(ui, "image.prompt.hint", "建议使用英文提示词以获得稳定效果", x + 24.0f, y + 53.0f,
         controls - 48.0f, 22.0f, 12.0f, kMuted);
    components::input(ui, "image.prompt").position(x + 24.0f, y + 88.0f)
        .size(controls - 48.0f, 152.0f).multiline().fontSize(14.0f)
        .value(state.image_prompt).placeholder("Describe the image…").theme(studioTheme())
        .onChange([](const std::string& value) { state.image_prompt = value; }).build();
    text(ui, "image.size.label", "尺寸", x + 24.0f, y + 270.0f, 100.0f, 22.0f, 12.0f, kMuted, 600);
    text(ui, "image.size.value", "512 × 512", x + controls - 140.0f, y + 270.0f,
         116.0f, 22.0f, 12.0f, kText, 650);
    text(ui, "image.steps.label", "采样步数", x + 24.0f, y + 306.0f, 100.0f, 22.0f, 12.0f, kMuted, 600);
    components::button(ui, "image.steps.down").position(x + controls - 140.0f, y + 298.0f)
        .size(34.0f, 34.0f).text("−").theme(studioTheme(), false).radius(8.0f).disabled(state.busy)
        .onClick([] { state.image_steps = std::max(4, state.image_steps - 4); }).build();
    text(ui, "image.steps.value", std::to_string(state.image_steps), x + controls - 100.0f, y + 305.0f,
         52.0f, 24.0f, 14.0f, kText, 700);
    components::button(ui, "image.steps.up").position(x + controls - 58.0f, y + 298.0f)
        .size(34.0f, 34.0f).text("+").theme(studioTheme(), false).radius(8.0f).disabled(state.busy)
        .onClick([] { state.image_steps = std::min(50, state.image_steps + 4); }).build();
    components::button(ui, "image.generate").position(x + 24.0f, y + height - 66.0f)
        .size(controls - 48.0f, 44.0f).text(state.busy ? "正在采样…" : "生成图片")
        .icon(0xF1FC).theme(studioTheme(), true).radius(10.0f).disabled(state.busy).onClick(submitImage).build();

    const float preview_x = x + controls + gap;
    const float preview_width = width - controls - gap;
    if (state.image_path.empty()) {
        text(ui, "image.empty.icon", "□", preview_x + 24.0f, y + height * 0.34f,
             preview_width - 48.0f, 68.0f, 48.0f, kMuted, 400);
        text(ui, "image.empty", "生成结果会显示在这里", preview_x + 24.0f, y + height * 0.50f,
             preview_width - 48.0f, 28.0f, 14.0f, kMuted, 550);
    } else {
        const float image_size = std::min(preview_width - 48.0f, height - 82.0f);
        components::image(ui, "image.result").position(preview_x + (preview_width - image_size) * 0.5f, y + 20.0f)
            .size(image_size, image_size).source(state.image_path).contain().radius(10.0f).build();
        text(ui, "image.filename", filenameOnly(state.image_path), preview_x + 24.0f, y + height - 42.0f,
             preview_width - 48.0f, 22.0f, 11.0f, kMuted);
    }
}

} // namespace

void compose(eui::Ui& ui, const eui::Screen& screen) {
    const bool compact = screen.width < 900.0f;
    const float sidebar_width = compact ? 82.0f : 218.0f;
    const float page_x = sidebar_width + 30.0f;
    const float page_width = std::max(320.0f, screen.width - page_x - 30.0f);
    const float content_y = 106.0f;
    const float content_height = std::max(360.0f, screen.height - content_y - 28.0f);

    ui.stack("root").size(screen.width, screen.height).content([&] {
        ui.rect("background").size(screen.width, screen.height).color(kBackground).build();
        composeSidebar(ui, sidebar_width, screen.height, compact);
        composeHeader(ui, page_x, page_width);
        if (state.tool == Tool::Chat) composeChat(ui, page_x, content_y, page_width, content_height);
        else if (state.tool == Tool::Speech) composeSpeech(ui, page_x, content_y, page_width, content_height);
        else composeImage(ui, page_x, content_y, page_width, content_height);
        if (state.busy) {
            ui.stack("task.progress.wrap").position(page_x, 96.0f).size(page_width, 3.0f).content([&] {
                components::progress(ui, "task.progress")
                    .size(page_width, 3.0f).value(state.progress.load()).theme(studioTheme()).build();
            }).build();
        }
    }).build();
}

} // namespace app
