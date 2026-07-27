#pragma once

#include "eui_neo.h"

#include <cstdio>
#include <string>

namespace app {

// ---- Palette -------------------------------------------------------------
// Deep charcoal-blue neutrals with a mint accent. Surfaces step up in three
// levels (background -> surface -> raised); every border comes from the same
// hue so panels read as one family.

inline eui::Color kBackground{0.031f, 0.039f, 0.059f, 1.0f};
inline eui::Color kSidebar{0.039f, 0.047f, 0.070f, 1.0f};
inline eui::Color kSurface{0.055f, 0.066f, 0.094f, 1.0f};
inline eui::Color kSurfaceRaised{0.075f, 0.090f, 0.125f, 1.0f};
inline eui::Color kSurfaceInset{0.039f, 0.049f, 0.074f, 1.0f};
inline eui::Color kBorder{0.145f, 0.169f, 0.224f, 1.0f};
inline eui::Color kBorderSoft{0.098f, 0.118f, 0.165f, 1.0f};

inline eui::Color kText{0.929f, 0.941f, 0.961f, 1.0f};
inline eui::Color kTextSecondary{0.741f, 0.769f, 0.816f, 1.0f};
inline eui::Color kMuted{0.545f, 0.580f, 0.647f, 1.0f};
// Faint still has to clear ~4.5:1 on kBackground — small labels use it.
inline eui::Color kFaint{0.475f, 0.506f, 0.569f, 1.0f};

inline eui::Color kAccent{0.376f, 0.647f, 0.980f, 1.0f};
inline eui::Color kAccentDim{0.255f, 0.455f, 0.780f, 1.0f};
inline eui::Color kAccentSoft{0.078f, 0.133f, 0.235f, 1.0f};
inline eui::Color kAccentEdge{0.180f, 0.333f, 0.596f, 1.0f};
inline eui::Color kBlue{0.220f, 0.749f, 0.969f, 1.0f};
inline eui::Color kPurple{0.655f, 0.545f, 0.980f, 1.0f};
inline eui::Color kOrange{0.984f, 0.573f, 0.235f, 1.0f};
inline eui::Color kDanger{0.937f, 0.463f, 0.443f, 1.0f};
inline eui::Color kToolLabel{0.867f, 0.706f, 0.396f, 1.0f};

inline eui::Color kUserBubble{0.078f, 0.133f, 0.235f, 1.0f};
inline eui::Color kUserBubbleBorder{0.180f, 0.333f, 0.596f, 1.0f};
inline eui::Color kThinkText{0.502f, 0.545f, 0.627f, 1.0f};

inline bool studioLightTheme = false;

inline void applyStudioPalette(bool light) {
    studioLightTheme = light;
    if (light) {
        kBackground={0.955f,0.965f,0.982f,1}; kSidebar={0.982f,0.986f,0.996f,1};
        kSurface={1,1,1,1}; kSurfaceRaised={0.965f,0.973f,0.988f,1};
        kSurfaceInset={0.935f,0.949f,0.973f,1}; kBorder={0.765f,0.804f,0.871f,1};
        kBorderSoft={0.855f,0.882f,0.925f,1}; kText={0.075f,0.098f,0.145f,1};
        kTextSecondary={0.225f,0.267f,0.345f,1}; kMuted={0.390f,0.435f,0.515f,1};
        kFaint={0.500f,0.545f,0.620f,1}; kAccent={0.165f,0.455f,0.890f,1};
        kAccentDim={0.125f,0.350f,0.710f,1}; kAccentSoft={0.875f,0.925f,1.0f,1};
        kAccentEdge={0.560f,0.710f,0.950f,1}; kUserBubble={0.875f,0.925f,1.0f,1};
        kUserBubbleBorder={0.560f,0.710f,0.950f,1}; kThinkText={0.390f,0.435f,0.515f,1};
    } else {
        kBackground={0.031f,0.039f,0.059f,1}; kSidebar={0.039f,0.047f,0.070f,1};
        kSurface={0.055f,0.066f,0.094f,1}; kSurfaceRaised={0.075f,0.090f,0.125f,1};
        kSurfaceInset={0.039f,0.049f,0.074f,1}; kBorder={0.145f,0.169f,0.224f,1};
        kBorderSoft={0.098f,0.118f,0.165f,1}; kText={0.929f,0.941f,0.961f,1};
        kTextSecondary={0.741f,0.769f,0.816f,1}; kMuted={0.545f,0.580f,0.647f,1};
        kFaint={0.475f,0.506f,0.569f,1}; kAccent={0.376f,0.647f,0.980f,1};
        kAccentDim={0.255f,0.455f,0.780f,1}; kAccentSoft={0.078f,0.133f,0.235f,1};
        kAccentEdge={0.180f,0.333f,0.596f,1}; kUserBubble={0.078f,0.133f,0.235f,1};
        kUserBubbleBorder={0.180f,0.333f,0.596f,1}; kThinkText={0.502f,0.545f,0.627f,1};
    }
}

// ---- Type scale ----------------------------------------------------------
// 20 page title / 14 body / 13 secondary / 12 caption / 11 overline.
// Sizes are true em sizes (the renderer no longer shrinks tall-line-box
// fonts). CJK glyphs need >= 12px to stay legible; overlines are Latin-only.

constexpr float kFontTitle = 20.0f;
constexpr float kFontBody = 14.0f;
constexpr float kFontSecondary = 13.0f;
constexpr float kFontCaption = 12.0f;
constexpr float kFontOverline = 11.0f;

inline const components::theme::ThemeColorTokens& studioTheme() {
    static components::theme::ThemeColorTokens theme;
    theme = studioLightTheme ? components::theme::light() : components::theme::dark();
    theme.background = kBackground;
    theme.surface = kSurfaceRaised;
    theme.border = kBorder;
    theme.text = kText;
    theme.primary = kAccent;
    return theme;
}

// Markdown style tuned for chat bubbles (denser than document defaults).
inline const components::MarkdownStyle& chatMarkdownStyle() {
    static components::MarkdownStyle style(studioTheme());
    style = [&] {
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
    static components::MarkdownStyle style(studioTheme());
    style = [] {
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
