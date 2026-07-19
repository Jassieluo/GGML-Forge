#include "eui_neo.h"

#include <algorithm>
#include <array>
#include <string>

namespace app {

const DslAppConfig& dslAppConfig() {
    static const DslAppConfig config = DslAppConfig{}
        .title("GGML-Forge UI Demo")
        .pageId("ggml_forge_ui_demo")
        .clearColor({0.025f, 0.035f, 0.055f, 1.0f})
        .windowSize(1100, 760)
        .fps(60.0);
    return config;
}

namespace {

struct DemoState {
    int selected = 0;
    int completedTasks = 0;
};

DemoState state;

constexpr std::array<const char*, 4> kCategories = {
    "LLM", "ASR", "TTS", "Visual Generation",
};

constexpr std::array<const char*, 4> kProviders = {
    "llama.cpp", "whisper.cpp", "GPT-SoVITS", "stable-diffusion.cpp",
};

constexpr std::array<const char*, 4> kDescriptions = {
    "Chat, completion and multimodal inference",
    "Local speech recognition and transcription",
    "Reference-guided speech synthesis",
    "Image and video generation providers",
};

constexpr std::array<eui::Color, 4> kAccents = {{
    {0.38f, 0.62f, 1.00f, 1.0f},
    {0.24f, 0.82f, 0.66f, 1.0f},
    {0.78f, 0.52f, 1.00f, 1.0f},
    {1.00f, 0.58f, 0.32f, 1.0f},
}};

} // namespace

void compose(eui::Ui& ui, const eui::Screen& screen) {
    const float pageWidth = std::min(1080.0f, std::max(320.0f, screen.width - 48.0f));
    const float pageX = (screen.width - pageWidth) * 0.5f;
    const float gap = 16.0f;
    const float cardWidth = (pageWidth - gap) * 0.5f;
    const float cardHeight = 154.0f;
    const eui::Color background{0.025f, 0.035f, 0.055f, 1.0f};
    const eui::Color surface{0.055f, 0.075f, 0.115f, 1.0f};
    const eui::Color border{0.14f, 0.19f, 0.28f, 1.0f};
    const eui::Color primaryText{0.93f, 0.96f, 1.0f, 1.0f};
    const eui::Color secondaryText{0.55f, 0.64f, 0.76f, 1.0f};

    ui.stack("root")
        .size(screen.width, screen.height)
        .content([&] {
            ui.rect("background")
                .size(screen.width, screen.height)
                .color(background)
                .build();

            ui.text("brand")
                .position(pageX, 34.0f)
                .size(pageWidth, 42.0f)
                .text("GGML-Forge")
                .fontSize(32.0f)
                .lineHeight(40.0f)
                .fontWeight(850)
                .color(primaryText)
                .build();

            ui.text("subtitle")
                .position(pageX, 78.0f)
                .size(pageWidth, 28.0f)
                .text("Provider-neutral inference, now with an EUI-NEO application layer")
                .fontSize(16.0f)
                .lineHeight(24.0f)
                .color(secondaryText)
                .build();

            ui.rect("runtime.status")
                .position(pageX, 122.0f)
                .size(pageWidth, 48.0f)
                .color({0.045f, 0.105f, 0.095f, 1.0f})
                .radius(12.0f)
                .border(1.0f, {0.12f, 0.32f, 0.27f, 1.0f})
                .build();
            ui.text("runtime.status.text")
                .position(pageX + 18.0f, 134.0f)
                .size(pageWidth - 36.0f, 24.0f)
                .text("Framework ready  |  GLFW + OpenGL  |  eui_neo.dll")
                .fontSize(14.0f)
                .lineHeight(22.0f)
                .color({0.48f, 0.92f, 0.76f, 1.0f})
                .build();

            for (int index = 0; index < 4; ++index) {
                const int row = index / 2;
                const int column = index % 2;
                const float x = pageX + column * (cardWidth + gap);
                const float y = 190.0f + row * (cardHeight + gap);
                const bool selected = state.selected == index;
                const eui::Color accent = kAccents[static_cast<size_t>(index)];

                ui.rect("category.card." + std::to_string(index))
                    .position(x, y)
                    .size(cardWidth, cardHeight)
                    .color(selected ? eui::Color{0.075f, 0.105f, 0.16f, 1.0f} : surface)
                    .radius(16.0f)
                    .border(selected ? 2.0f : 1.0f, selected ? accent : border)
                    .shadow(selected ? 18.0f : 8.0f, 0.0f, 5.0f,
                        selected ? eui::Color{accent.r, accent.g, accent.b, 0.18f}
                                 : eui::Color{0.0f, 0.0f, 0.0f, 0.16f})
                    .build();
                ui.rect("category.accent." + std::to_string(index))
                    .position(x + 20.0f, y + 22.0f)
                    .size(10.0f, 10.0f)
                    .color(accent)
                    .radius(5.0f)
                    .build();
                ui.text("category.title." + std::to_string(index))
                    .position(x + 42.0f, y + 14.0f)
                    .size(cardWidth - 62.0f, 30.0f)
                    .text(kCategories[static_cast<size_t>(index)])
                    .fontSize(21.0f)
                    .lineHeight(28.0f)
                    .fontWeight(760)
                    .color(primaryText)
                    .build();
                ui.text("category.provider." + std::to_string(index))
                    .position(x + 20.0f, y + 58.0f)
                    .size(cardWidth - 40.0f, 24.0f)
                    .text(kProviders[static_cast<size_t>(index)])
                    .fontSize(14.0f)
                    .lineHeight(22.0f)
                    .color(accent)
                    .build();
                ui.text("category.description." + std::to_string(index))
                    .position(x + 20.0f, y + 91.0f)
                    .size(cardWidth - 40.0f, 42.0f)
                    .text(kDescriptions[static_cast<size_t>(index)])
                    .fontSize(13.0f)
                    .lineHeight(20.0f)
                    .color(secondaryText)
                    .build();
            }

            const float controlsY = 548.0f;
            components::button(ui, "select.next")
                .position(pageX, controlsY)
                .size(190.0f, 48.0f)
                .text("Next category")
                .onClick([] { state.selected = (state.selected + 1) % 4; })
                .build();

            components::button(ui, "task.run")
                .position(pageX + 206.0f, controlsY)
                .size(190.0f, 48.0f)
                .text("Run demo task")
                .onClick([] { ++state.completedTasks; })
                .build();

            ui.text("selection")
                .position(pageX, controlsY + 70.0f)
                .size(pageWidth, 28.0f)
                .text("Selected: " + std::string(kCategories[static_cast<size_t>(state.selected)]) +
                    "  |  Demo tasks completed: " + std::to_string(state.completedTasks))
                .fontSize(15.0f)
                .lineHeight(24.0f)
                .color(primaryText)
                .build();

            ui.text("note")
                .position(pageX, controlsY + 108.0f)
                .size(pageWidth, 24.0f)
                .text("This example exercises UI composition only; no model is loaded.")
                .fontSize(13.0f)
                .lineHeight(22.0f)
                .color(secondaryText)
                .build();
        });
}

} // namespace app
