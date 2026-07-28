#pragma once

#include "eui_neo.h"
#include "pages/actions.h"
#include "pages/state.h"
#include "pages/theme.h"
#include "services/engine_service.h"
#include "services/tool_calls.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>

namespace app {

inline void toggleThink(std::size_t index) {
    if (index >= state.messages.size()) return;
    state.messages[index].think_expanded = !state.messages[index].think_expanded;
    ++state.chat_layout_epoch;  // Invalidate the scroll view's measure cache.
}

// Collapsible reasoning section: an inset header row that toggles the dimmed
// think markdown below it. The live bubble passes SIZE_MAX (no toggling) and
// controls expansion itself while the model is still reasoning.
inline void thinkSection(eui::Ui& ui, const std::string& id, const ChatMessage& message,
                         float inner_width, bool reasoning_live, std::size_t message_index) {
    const bool expanded = message.think_expanded;
    const std::string label = std::string(expanded ? "▾  " : "▸  ") +
        (reasoning_live ? "正在思考…" : "深度思考");

    ui.stack(id + ".think.header")
        .width(inner_width)
        .height(28.0f)
        .content([&] {
            components::mouseArea(ui, id + ".think.hit")
                .size(inner_width, 28.0f)
                .radius(8.0f)
                .color(kSurfaceInset)
                .onTap([message_index] { toggleThink(message_index); })
                .build();
            text(ui, id + ".think.label", label, 10.0f, 6.0f, inner_width - 20.0f, 16.0f,
                 kFontCaption, reasoning_live ? kAccentDim : kFaint, 650);
        })
        .build();

    if (expanded && !message.thinking.empty()) {
        components::card(ui, id + ".think.body")
            .width(inner_width)
            .wrapContentHeight()
            .padding(10.0f)
            .color(kSurfaceInset)
            .radius(8.0f)
            .border(1.0f, kBorderSoft)
            .content([&] {
                components::markdown(ui, id + ".think.md")
                    .markdown(message.thinking)
                    .width(inner_width - 20.0f)
                    .wrapContentHeight()
                    .style(thinkMarkdownStyle())
                    .build();
            })
            .build();
    }
}

// One message bubble: a card that grows with its content. Markdown is used
// for both roles because it measures its own wrapped height (the raw layout
// engine does not measure plain wrapped text). Assistant bubbles may carry a
// collapsible reasoning section; tool bubbles may carry image or audio
// attachments.
inline void chatBubble(eui::Ui& ui, const std::string& id, const ChatMessage& message,
                       float contentWidth, bool live,
                       std::size_t message_index = static_cast<std::size_t>(-1)) {
    const bool user = message.role == "user";
    const bool tool = message.role == "tool";
    const bool reasoning_live = live && !message.thinking.empty() && message.text.empty();
    const float bubble_width = std::min(680.0f, contentWidth * 0.84f);
    const float inner_width = bubble_width - 30.0f;

    const char* label = user ? "你"
                      : tool ? "工具"
                      : live ? (reasoning_live ? "FORGE · 思考中" : "FORGE · 生成中")
                             : "FORGE";
    const eui::Color label_color = user ? kAccent : (tool ? kToolLabel : kMuted);

    ui.row(id + ".row")
        .width(contentWidth)
        .height(core::SizeValue::wrapContent())
        .justifyContent(user ? core::Align::END : core::Align::START)
        .content([&] {
            components::card(ui, id)
                .width(bubble_width)
                .wrapContentHeight()
                .padding(15.0f)
                .color(user ? kUserBubble : kSurfaceRaised)
                .radius(14.0f)
                .border(1.0f, user ? kUserBubbleBorder : kBorderSoft)
                .content([&] {
                    ui.column(id + ".body")
                        .width(inner_width)
                        .height(core::SizeValue::wrapContent())
                        .gap(8.0f)
                        .content([&] {
                            // Role line: a small tinted dot plus the label.
                            ui.stack(id + ".head").width(inner_width).height(14.0f).content([&] {
                                ui.rect(id + ".dot").position(0.0f, 4.0f).size(7.0f, 7.0f)
                                    .color(label_color).radius(3.5f).build();
                                text(ui, id + ".role", label, 14.0f, 0.0f, inner_width - 14.0f,
                                     14.0f, kFontOverline, label_color, 750);
                            }).build();

                            if (!message.thinking.empty()) {
                                thinkSection(ui, id, message, inner_width, reasoning_live,
                                             message_index);
                            }
                            if (!message.image_path.empty()) {
                                const float image_width = std::min(320.0f, inner_width);
                                const float aspect = message.image_width > 0 && message.image_height > 0
                                    ? static_cast<float>(message.image_height) / message.image_width
                                    : 1.0f;
                                const float image_height = std::min(320.0f, image_width * aspect);
                                ui.stack(id + ".img.wrap")
                                    .size(image_width, image_height)
                                    .clip()
                                    .content([&] {
                                        components::image(ui, id + ".img")
                                            .position(0.0f, 0.0f)
                                            .size(image_width, image_height)
                                            .source(message.image_path).contain().radius(10.0f)
                                            .build();
                                    })
                                    .build();
                            }
                            if (!message.text.empty()) {
                                components::markdown(ui, id + ".md")
                                    .markdown(message.text)
                                    .width(inner_width)
                                    .wrapContentHeight()
                                    .style(chatMarkdownStyle())
                                    .build();
                            }
                            if (!message.audio_path.empty()) {
                                const std::string path = message.audio_path;
                                components::button(ui, id + ".audio")
                                    .size(std::min(150.0f, inner_width), 34.0f)
                                    .text("播放语音").icon(0xF04B).fontSize(12.0f).iconSize(12.0f)
                                    .theme(studioTheme(), false).radius(9.0f)
                                    .onClick([path] { EngineService::playAudio(path); })
                                    .build();
                            }
                        })
                        .build();
                })
                .build();
        })
        .build();
}

inline void composeChat(eui::Ui& ui, float x, float y, float width, float height) {
    // Pull freshly streamed tokens onto the UI thread once per frame.
    if (state.streaming) {
        state.stream.snapshot(state.stream_seen, state.streaming_text);
    }

    panel(ui, "chat.panel", x, y, width, height);
    const float pad = 20.0f;
    const bool has_attachment = !state.chat_attachment_path.empty();
    const float composer_height = has_attachment ? 154.0f : 68.0f;
    const float list_height = std::max(120.0f, height - composer_height - pad * 2.0f);
    const float list_width = width - pad * 2.0f;
    // Match the scroll view's content width (scrollbar + gap reservation) so
    // bubbles do not re-wrap when switching between pinned and scrolling mode.
    const float bubble_width = std::max(120.0f, list_width - 24.0f);

    const auto compose_messages = [&](eui::Ui& list_ui, float contentWidth) {
        for (std::size_t index = 0; index < state.messages.size(); ++index) {
            chatBubble(list_ui, "chat.msg." + std::to_string(index),
                       state.messages[index], contentWidth, false, index);
        }
        if (state.streaming) {
            // The live bubble splits reasoning from the visible answer as it
            // streams: reasoning shows expanded (dimmed) while the model is
            // thinking, collapses once the answer starts, and a half-streamed
            // <tool_call> blob is swapped for a short activity note.
            ChatMessage live;
            live.role = "assistant";
            live.text = state.streaming_text;
            live.thinking = tool_calls::stripThinking(live.text);
            live.think_expanded = !live.thinking.empty() && live.text.empty();
            const std::size_t call_at = live.text.find("<tool_call>");
            if (call_at != std::string::npos) {
                live.text.resize(call_at);
                while (!live.text.empty() &&
                       std::isspace(static_cast<unsigned char>(live.text.back()))) {
                    live.text.pop_back();
                }
                live.text += live.text.empty() ? "正在准备工具调用…"
                                               : "\n\n正在准备工具调用…";
            }
            chatBubble(list_ui, "chat.msg.live", live, contentWidth, true);
        }
    };

    if (state.chat_stick_bottom) {
        // Pinned: a bottom-justified clipped column always shows the newest
        // content (END-justify goes negative when the content overflows, and
        // clip() hides the top). Scrolling up hands control to the scroll
        // view, remounted via the epoch so its bottom seed is honored.
        ui.column("chat.pin." + std::to_string(state.chat_scroll_epoch))
            .position(x + pad, y + pad)
            .size(list_width, list_height)
            .gap(14.0f)
            .justifyContent(core::Align::END)
            .clip()
            .onScroll([](const core::ScrollEvent& event) {
                if (event.y > 0.0) {
                    state.chat_stick_bottom = false;
                    state.chat_scroll.set(1.0e9f);  // Seed at the bottom; clamped on mount.
                    ++state.chat_scroll_epoch;
                }
            })
            .content([&] { compose_messages(ui, bubble_width); })
            .build();
    } else {
        // The content key ties the scroll view's measurement cache to the chat
        // content: messages, streamed tokens, and think-toggle layout changes.
        const std::string content_key =
            "m" + std::to_string(state.messages.size()) +
            "s" + std::to_string(state.stream_seen) +
            "t" + std::to_string(state.chat_layout_epoch) +
            (state.streaming ? "L" : "");

        components::scrollView(ui, "chat.scroll." + std::to_string(state.chat_scroll_epoch))
            .position(x + pad, y + pad)
            .size(list_width, list_height)
            .gap(14.0f)
            .bind(state.chat_scroll)
            .theme(studioTheme())
            .contentKey(content_key)
            .content([&](eui::Ui& list_ui, float contentWidth, float) {
                compose_messages(list_ui, contentWidth);
            })
            .build();
    }

    ui.rect("chat.composer.line").position(x + pad, y + height - composer_height - 8.0f)
        .size(width - pad * 2.0f, 1.0f).color(kBorderSoft).build();

    if (has_attachment) {
        const float preview_y = y + height - 144.0f;
        ui.rect("chat.attachment.card").position(x + pad, preview_y)
            .size(76.0f, 76.0f).color(kSurfaceInset)
            .radius(12.0f).border(1.0f, kBorderSoft).build();
        components::image(ui, "chat.attachment.preview")
            .position(x + pad + 4.0f, preview_y + 4.0f).size(68.0f, 68.0f)
            .source(state.chat_attachment_path).cover().radius(9.0f).build();
    components::button(ui, "chat.attachment.remove")
            .position(x + pad + 56.0f, preview_y - 5.0f).size(24.0f, 24.0f)
            .text("").icon(0xF00D).iconSize(11.0f)
            .theme(studioTheme(), false).radius(12.0f)
            .disabled(state.busy && !state.chat_preloading).onClick(clearChatImage).build();
    }

    components::button(ui, "chat.attach").position(x + pad, y + height - 58.0f)
        .size(46.0f, 46.0f).text("").icon(0xF03E).iconSize(16.0f)
        .theme(studioTheme(), false).radius(12.0f)
        .disabled(state.busy && !state.chat_preloading).onClick(chooseChatImage).build();

    const float input_width = width - 208.0f;
    components::input(ui, "chat.input").position(x + pad + 58.0f, y + height - 58.0f)
        .size(input_width, 46.0f).value(state.chat_input)
        .placeholder("输入消息，或附加图片 · Enter 发送")
        .theme(studioTheme()).fontSize(kFontBody - 1.0f)
        .onChange([](const std::string& value) { state.chat_input = value; })
        .onEnter(submitChat)
        .build();

    const bool can_stop = state.busy && !state.chat_preloading && state.tool == Tool::Chat;
    const bool queued = state.chat_preloading && state.chat_submit_queued;
    components::button(ui, "chat.send").position(x + width - 118.0f, y + height - 58.0f)
        .size(98.0f, 46.0f)
        .text(can_stop ? "停止" : queued ? "已排队" : "发送")
        .icon(can_stop ? 0xF04D : queued ? 0xF017 : 0xF1D8).fontSize(14.0f)
        .theme(studioTheme(), true).radius(12.0f)
        .disabled((state.busy && !can_stop && !state.chat_preloading) || queued)
        .onClick([can_stop] { can_stop ? requestCancel() : submitChat(); })
        .build();
}

} // namespace app
