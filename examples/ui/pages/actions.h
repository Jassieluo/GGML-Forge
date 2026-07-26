#pragma once

// UI-thread task glue: every submit* runs its work on the core::async worker
// and mutates StudioState only from the completion callback (UI thread). The
// single state.busy guard serializes engine access.
//
// The chat page runs a small agent loop: the LLM may answer directly or emit
// one <tool_call> per round (services/tool_calls.h). Tool output is appended
// as a "tool" message and the LLM is asked again, up to kMaxAgentRounds.

#include "core/platform/async.h"
#include "pages/state.h"
#include "services/engine_service.h"
#include "services/tool_calls.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace app {

constexpr int kMaxAgentRounds = 3;

// Runs a worker job and converts any escaped exception into a failed Result
// instead of letting it reach std::terminate (which exits the app silently).
template <typename Fn>
auto guardedRun(Fn&& fn) {
    using R = decltype(fn());
    try {
        return fn();
    } catch (const std::exception& e) {
        R result;
        result.ok = false;
        result.error = std::string("工作线程异常: ") + e.what();
        return result;
    } catch (...) {
        R result;
        result.ok = false;
        result.error = "工作线程发生未知异常";
        return result;
    }
}

constexpr const char* kAgentSystemPrompt =
    "你是 GGML-Forge Studio 中的本地 AI 助手，运行在用户自己的电脑上。\n"
    "你可以调用以下工具：\n"
    "1. generate_image：文本生成图片。参数 prompt（英文提示词，必填）、steps（采样步数，可选，4-50）。\n"
    "2. text_to_speech：中文语音合成并自动播放。参数 text（要朗读的中文文本，必填）、speed（语速，可选，0.7-1.5）。\n"
    "当用户需要图片或语音时，单独输出一行工具调用，格式如下：\n"
    "<tool_call>{\"name\":\"generate_image\",\"arguments\":{\"prompt\":\"a cat sitting on a windowsill\"}}</tool_call>\n"
    "系统会以「[工具结果]」开头的消息返回执行结果，之后你再根据结果用中文简洁回复用户。\n"
    "一次只调用一个工具。不需要工具时直接用中文回答，绝不要输出 <tool_call>。";

inline void beginTask(const std::string& status) {
    state.busy = true;
    state.has_error = false;
    state.cancel_requested.store(false);
    state.progress.store(0.02f);
    state.status = status;
}

inline void finishTaskError(const std::string& error) {
    state.busy = false;
    state.has_error = true;
    state.progress.store(0.0f);
    state.status = error;
}

// ---- Chat agent loop ----

inline void startChatRound();

inline void finishChatTurn(const std::string& status) {
    state.streaming = false;
    state.busy = false;
    state.has_error = false;
    state.status = status;
    state.chat_stick_bottom = true;
}

// Appends the assistant reply. `thinking` (already split from the visible
// text) is kept for display only — history building and the speech页 sync use
// the answer alone.
inline void appendAssistantMessage(std::string text, std::string thinking = {}) {
    if (text.empty() && thinking.empty()) return;
    if (!text.empty()) state.speech_text = text;
    ChatMessage message;
    message.role = "assistant";
    message.text = std::move(text);
    message.thinking = std::move(thinking);
    state.messages.push_back(std::move(message));
}

inline void runToolCall(const ToolCall& call) {
    state.progress.store(0.02f);
    if (call.name == "generate_image") {
        state.status = "Agent 正在生成图片…";
        const std::string prompt = call.prompt;
        const int steps = call.steps > 0 ? std::clamp(call.steps, 4, 50) : state.image_steps;
        const int backend = state.backend;
        core::async::restart("forge.ui.chat",
            [prompt, steps, backend] {
                return guardedRun([&] {
                    return EngineService::generateImage(prompt, steps, backend,
                                                        &state.progress, &state.cancel_requested);
                });
            },
            [prompt](core::async::Result<ImageResult> result) {
                if (result.ok && result.value.canceled) return finishChatTurn("已停止");
                ChatMessage message;
                message.role = "tool";
                if (!result.ok) {
                    message.text = "[工具结果] 图片生成失败: " + result.error;
                } else {
                    message.text = "[工具结果] 图片已生成 (" +
                        std::to_string(result.value.width) + "×" +
                        std::to_string(result.value.height) + ")，提示词: " + prompt;
                    message.image_path = result.value.path;
                    message.image_width = result.value.width;
                    message.image_height = result.value.height;
                    // Mirror onto the image page so the result stays reachable.
                    state.image_path = result.value.path;
                    state.image_width = result.value.width;
                    state.image_height = result.value.height;
                }
                state.messages.push_back(std::move(message));
                state.chat_stick_bottom = true;
                startChatRound();
            });
    } else {  // text_to_speech
        state.status = "Agent 正在合成语音…";
        const std::string text = call.text;
        const float speed = call.speed > 0.0f ? std::clamp(call.speed, 0.7f, 1.5f) : state.speech_speed;
        const int backend = state.backend;
        core::async::restart("forge.ui.chat",
            [text, speed, backend] {
                return guardedRun([&] {
                    return EngineService::synthesize(text, speed, backend, &state.progress);
                });
            },
            [](core::async::Result<SpeechResult> result) {
                if (state.cancel_requested.load()) return finishChatTurn("已停止");
                ChatMessage message;
                message.role = "tool";
                if (!result.ok) {
                    message.text = "[工具结果] 语音合成失败: " + result.error;
                } else {
                    char duration[32] = {};
                    std::snprintf(duration, sizeof(duration), "%.1f", result.value.seconds);
                    message.text = std::string("[工具结果] 语音已合成并播放，时长 ") + duration + " 秒";
                    message.audio_path = result.value.path;
                    // Mirror onto the speech page and play immediately.
                    state.audio_path = result.value.path;
                    state.audio_seconds = result.value.seconds;
                    EngineService::playAudio(result.value.path);
                }
                state.messages.push_back(std::move(message));
                state.chat_stick_bottom = true;
                startChatRound();
            });
    }
}

inline void handleChatResult(core::async::Result<LlmResult> result) {
    state.streaming = false;
    if (!result.ok) return finishTaskError(result.error);

    // Reasoning is split off before anything else so tool-call parsing and
    // history building only ever see the visible answer.
    std::string reply = std::move(result.value.text);
    std::string thinking = tool_calls::stripThinking(reply);

    if (result.value.canceled) {
        appendAssistantMessage(std::move(reply), std::move(thinking));  // Keep the partial answer.
        return finishChatTurn("已停止");
    }

    ToolCall call;
    std::string cleaned;
    if (tool_calls::parse(reply, call, cleaned)) {
        if (state.agent_rounds < kMaxAgentRounds) {
            ++state.agent_rounds;
            if (!cleaned.empty() || !thinking.empty()) {
                appendAssistantMessage(cleaned, std::move(thinking));
            }
            state.chat_stick_bottom = true;
            runToolCall(call);
            return;
        }
        appendAssistantMessage(std::move(cleaned), std::move(thinking));
        return finishChatTurn("已达到本轮工具调用上限");
    }

    appendAssistantMessage(std::move(reply), std::move(thinking));
    finishChatTurn("回答完成 · 已同步到语音合成");
}

inline void startChatRound() {
    state.streaming = true;
    state.streaming_text.clear();
    state.stream.clear();
    state.stream.snapshot(state.stream_seen, state.streaming_text);  // Resync version.
    state.streaming_text.clear();
    state.chat_stick_bottom = true;
    state.status = "正在生成回答…";
    state.progress.store(0.02f);

    // The provider only understands user/assistant/system, so tool results
    // travel as user-role text (already prefixed with [工具结果]). Reasoning
    // stays out of the transcript, and think-only messages are skipped.
    std::vector<ChatMessage> history;
    history.reserve(state.messages.size() + 1);
    history.push_back({"system", kAgentSystemPrompt});
    for (const auto& message : state.messages) {
        if (message.text.empty()) continue;
        history.push_back({message.role == "tool" ? "user" : message.role, message.text});
    }

    const int backend = state.backend;
    core::async::restart("forge.ui.chat",
        [history = std::move(history), backend] {
            return guardedRun([&] {
                return EngineService::chat(history, backend, &state.progress,
                                           &state.stream, &state.cancel_requested);
            });
        },
        handleChatResult);
}

inline void submitChat() {
    if (state.busy || state.chat_input.empty()) return;
    state.messages.push_back({"user", state.chat_input});
    state.chat_input.clear();
    state.agent_rounds = 0;
    beginTask("正在生成回答…");
    startChatRound();
}

// ---- Speech / image pages ----

inline void submitSpeech() {
    if (state.busy || state.speech_text.empty()) return;
    const std::string input = state.speech_text;
    const float speed = state.speech_speed;
    const int backend = state.backend;
    beginTask("正在合成语音…");
    core::async::restart("forge.ui.speech",
        [input, speed, backend] {
            return guardedRun([&] {
                return EngineService::synthesize(input, speed, backend, &state.progress);
            });
        },
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

inline void submitImage() {
    if (state.busy || state.image_prompt.empty()) return;
    const std::string prompt = state.image_prompt;
    const int steps = state.image_steps;
    const int backend = state.backend;
    beginTask("正在生成图片…");
    core::async::restart("forge.ui.image",
        [prompt, steps, backend] {
            return guardedRun([&] {
                return EngineService::generateImage(prompt, steps, backend,
                                                    &state.progress, &state.cancel_requested);
            });
        },
        [](core::async::Result<ImageResult> result) {
            if (!result.ok) return finishTaskError(result.error);
            state.busy = false;
            state.has_error = false;
            if (result.value.canceled) {
                state.status = "已停止";
                return;
            }
            state.image_path = std::move(result.value.path);
            state.image_width = result.value.width;
            state.image_height = result.value.height;
            state.status = "图片已生成";
        });
}

inline void submitReleaseEngines() {
    if (state.busy || !EngineService::loaded().any()) return;
    beginTask("正在释放模型…");
    core::async::restart("forge.ui.release",
        [] { return guardedRun([] { return EngineService::releaseAll(); }); },
        [](core::async::Result<void> result) {
            state.busy = false;
            state.has_error = !result.ok;
            state.status = result.ok ? "模型已释放" : "释放失败: " + result.error;
        });
}

// Stop button: chat stops at the next streamed token; an in-flight image
// generation (from the image page or an agent tool round) is asked to cancel
// via the provider. Speech has no cancellation path.
inline void requestCancel() {
    if (!state.busy) return;
    state.cancel_requested.store(true);
    EngineService::requestImageCancel();  // No-op unless an image is in flight.
    state.status = "正在停止…";
}

inline void switchBackend() {
    state.backend = (state.backend + 1) % kBackendCount;
    state.has_error = false;
    state.status = std::string("后端已切换为 ") + kBackendNames[state.backend] +
                   " · 下次生成时生效";
}

} // namespace app
