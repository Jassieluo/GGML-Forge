#pragma once

#include "eui_neo.h"

#include <cstdio>
#include <string>

namespace app {

// ---- Palette -------------------------------------------------------------
// Deep charcoal-blue neutrals with a mint accent. Surfaces step up in three
// levels (background -> surface -> raised); every border comes from the same
// hue so panels read as one family.

constexpr eui::Color kBackground{0.043f, 0.051f, 0.070f, 1.0f};     // window
constexpr eui::Color kSidebar{0.055f, 0.065f, 0.088f, 1.0f};        // nav rail
constexpr eui::Color kSurface{0.071f, 0.082f, 0.108f, 1.0f};        // panels
constexpr eui::Color kSurfaceRaised{0.094f, 0.108f, 0.141f, 1.0f};  // cards
constexpr eui::Color kSurfaceInset{0.058f, 0.067f, 0.090f, 1.0f};   // wells
constexpr eui::Color kBorder{0.153f, 0.173f, 0.216f, 1.0f};
constexpr eui::Color kBorderSoft{0.114f, 0.129f, 0.161f, 1.0f};

constexpr eui::Color kText{0.929f, 0.941f, 0.961f, 1.0f};
constexpr eui::Color kTextSecondary{0.741f, 0.769f, 0.816f, 1.0f};
constexpr eui::Color kMuted{0.545f, 0.580f, 0.647f, 1.0f};
constexpr eui::Color kFaint{0.396f, 0.427f, 0.490f, 1.0f};

constexpr eui::Color kAccent{0.380f, 0.816f, 0.702f, 1.0f};         // mint
constexpr eui::Color kAccentDim{0.278f, 0.596f, 0.518f, 1.0f};
constexpr eui::Color kAccentSoft{0.098f, 0.184f, 0.176f, 1.0f};     // fills
constexpr eui::Color kAccentEdge{0.196f, 0.408f, 0.365f, 1.0f};     // borders
constexpr eui::Color kDanger{0.937f, 0.463f, 0.443f, 1.0f};
constexpr eui::Color kToolLabel{0.867f, 0.706f, 0.396f, 1.0f};      // amber

constexpr eui::Color kUserBubble{0.102f, 0.180f, 0.173f, 1.0f};
constexpr eui::Color kUserBubbleBorder{0.180f, 0.365f, 0.329f, 1.0f};
constexpr eui::Color kThinkText{0.502f, 0.545f, 0.627f, 1.0f};      // dimmed

// ---- Type scale ----------------------------------------------------------
// 26 page title / 15 body / 13 secondary / 11 caption / 10 overline.

constexpr float kFontTitle = 26.0f;
constexpr float kFontBody = 15.0f;
constexpr float kFontSecondary = 13.0f;
constexpr float kFontCaption = 11.0f;
constexpr float kFontOverline = 10.0f;

inline const components::theme::ThemeColorTokens& studioTheme() {
    static const components::theme::ThemeColorTokens theme = [] {
        auto tokens = components::theme::dark();
        tokens.background = kBackground;
        tokens.surface = kSurfaceRaised;
        tokens.border = kBorder;
        tokens.text = kText;
        tokens.primary = kAccent;
        return tokens;
    }();
    return theme;
}

// Markdown style tuned for chat bubbles (denser than document defaults).
inline const components::MarkdownStyle& chatMarkdownStyle() {
    static const components::MarkdownStyle style = [] {
        components::MarkdownStyle value(studioTheme());
        value.text = kText;
        value.heading = kText;
        value.muted = kMuted;
        value.accent = kAccent;
        value.codeBackground = kSurfaceInset;
        value.quoteBackground = kSurfaceInset;
        value.divider = kBorderSoft;
        value.bodySize = 14.0f;
        value.bodyLineHeight = 23.0f;
        value.h1Size = 19.0f;
        value.h2Size = 17.0f;
        value.h3Size = 15.0f;
        value.codeSize = 12.5f;
        value.blockGap = 9.0f;
        value.radius = 8.0f;
        return value;
    }();
    return style;
}

// Dimmed, compact markdown for collapsed reasoning ("think") sections.
inline const components::MarkdownStyle& thinkMarkdownStyle() {
    static const components::MarkdownStyle style = [] {
        components::MarkdownStyle value = chatMarkdownStyle();
        value.text = kThinkText;
        value.heading = kThinkText;
        value.accent = kAccentDim;
        value.bodySize = 12.5f;
        value.bodyLineHeight = 20.0f;
        value.codeSize = 11.5f;
        value.blockGap = 7.0f;
        return value;
    }();
    return style;
}

// ---- Draw helpers --------------------------------------------------------

// Positioned single-line text helper shared by the pages.
inline void text(eui::Ui& ui, const std::string& id, const std::string& value,
                 float x, float y, float width, float height, float size,
                 eui::Color color = kText, int weight = 500, bool wrap = false) {
    ui.text(id).position(x, y).size(width, height).text(value)
        .fontSize(size).lineHeight(size * 1.5f).fontWeight(weight)
        .wrap(wrap).color(color).build();
}

inline void panel(eui::Ui& ui, const std::string& id, float x, float y,
                  float width, float height, eui::Color color = kSurface) {
    ui.rect(id).position(x, y).size(width, height).color(color)
        .radius(16.0f).border(1.0f, kBorderSoft).build();
}

// Uppercase-style section label ("合成文本", "本次输出", ...).
inline void sectionLabel(eui::Ui& ui, const std::string& id, const std::string& value,
                         float x, float y, float width) {
    text(ui, id, value, x, y, width, 18.0f, kFontCaption, kFaint, 700);
}

// Progress bar with a status line and percent readout, shared by the speech
// and image pages while a generation is running.
inline void progressWithLabel(eui::Ui& ui, const std::string& id,
                              float x, float y, float width,
                              const std::string& label, float value) {
    char percent[16] = {};
    std::snprintf(percent, sizeof(percent), "%d%%",
                  static_cast<int>(value * 100.0f + 0.5f));
    text(ui, id + ".label", label, x, y, width - 56.0f, 20.0f, kFontSecondary, kTextSecondary, 600);
    text(ui, id + ".percent", percent, x + width - 48.0f, y, 48.0f, 20.0f, kFontSecondary, kAccent, 750);
    ui.stack(id + ".wrap").position(x, y + 28.0f).size(width, 6.0f).content([&] {
        components::progress(ui, id)
            .size(width, 6.0f).value(value).theme(studioTheme()).build();
    }).build();
}

} // namespace app
