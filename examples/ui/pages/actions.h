#pragma once

// UI-thread task glue: every submit* runs its work on the core::async worker
// and mutates StudioState only from the completion callback (UI thread). The
// single state.busy guard serializes engine access.
//
// The chat page runs a small agent loop: the LLM may answer directly or emit
// one <tool_call> per round (services/tool_calls.h). Tool output is appended
// as a "tool" message and the LLM is asked again, up to kMaxAgentRounds.

#include "core/platform/async.h"
#include "image/image_io.h"
#include "pages/state.h"
#include "services/engine_service.h"
#include "services/tool_calls.h"
#include "services/model_catalog.h"
#include "services/vision_service.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
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

inline void chooseChatImage() {
    if (state.busy && !state.chat_preloading) return;
    const std::string path = eui::platform::chooseFile(
        {tr("选择要发送的图片", "Choose an image to send"), {".png", ".jpg", ".jpeg", ".webp"}});
    if (path.empty()) return;
    forge::media::Image image;
    std::string error;
    if (!forge::media::load_image(std::filesystem::u8path(path), image, error)) {
        return finishTaskError(error.empty() ? tr("无法读取图片", "Unable to read image") : error);
    }
    state.chat_attachment_path = path;
    state.chat_attachment_width = static_cast<int>(image.width);
    state.chat_attachment_height = static_cast<int>(image.height);
    state.has_error = false;
}

inline void clearChatImage() {
    if (state.busy && !state.chat_preloading) return;
    state.chat_attachment_path.clear();
    state.chat_attachment_width = 0;
    state.chat_attachment_height = 0;
}

inline void selectTtsVoice(int index) {
    const auto& voices = voiceCatalog();
    if (state.busy || index < 0 || index >= static_cast<int>(voices.size())) return;
    const auto& voice = voices[static_cast<size_t>(index)];
    state.tts_voice_path = voice.path;
    state.tts_reference_text = voice.text;
    state.tts_reference_language = voice.language;
    state.voice_dropdown_open.set(false);
    state.has_error = false;
    state.status = tr("参考声音已切换 · 下次合成时生效", "Reference voice changed · Applies to the next synthesis");
}

inline void chooseTtsReference() {
    if (state.busy) return;
    const std::string path = eui::platform::chooseFile(
        {tr("选择参考声音 WAV", "Choose a reference voice WAV"), {".wav"}, {}, tr("WAV 音频", "WAV audio")});
    if (path.empty()) return;
    std::vector<float> audio;
    uint32_t sample_rate = 0;
    std::string error;
    if (!forge::media::load_wav_mono(
            std::filesystem::u8path(path), audio, sample_rate, error)) {
        return finishTaskError(error.empty() ? tr("无法读取参考 WAV", "Unable to read reference WAV") : error);
    }
    state.tts_voice_path = path;
    state.tts_reference_text.clear();
    state.tts_reference_language = "zh";
    state.has_error = false;
    state.status = tr("已导入参考 WAV · 请填写音频中实际说出的文字", "Reference WAV imported · Enter its exact transcript");
}

inline void chooseAsrAudio() {
    if (state.busy) return;
    const std::string path = eui::platform::chooseFile(
        {tr("选择要识别的 WAV", "Choose a WAV to transcribe"), {".wav"}, {}, tr("WAV 音频", "WAV audio")});
    if (path.empty()) return;
    std::vector<float> audio;
    uint32_t sample_rate = 0;
    std::string error;
    if (!forge::media::load_wav_mono(
            std::filesystem::u8path(path), audio, sample_rate, error)) {
        return finishTaskError(error.empty() ? tr("无法读取 WAV", "Unable to read WAV") : error);
    }
    state.asr_audio_path = path;
    state.asr_transcript.clear();
    state.asr_detected_language.clear();
    state.has_error = false;
    state.status = tr("音频已就绪", "Audio ready");
}

inline void submitChat();

inline void preloadChatEngine() {
    if (state.busy) return;
    const int backend = state.backend;
    state.chat_preloading = true;
    beginTask(tr("正在准备对话模型…", "Preparing chat model…"));
    core::async::restart("forge.ui.chat.preload",
        [backend] { return EngineService::preloadChat(backend); },
        [](core::async::Result<bool> result) {
            state.chat_preloading = false;
            state.busy = false;
            state.progress.store(0.0f);
            if (!result.ok) {
                state.chat_submit_queued = false;
                return finishTaskError(result.error);
            }
            state.has_error = false;
            state.status = tr("对话模型已就绪", "Chat model ready");
            if (state.chat_submit_queued) {
                state.chat_submit_queued = false;
                submitChat();
            }
        });
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
        state.status = tr("Agent 正在生成图片…", "Agent is generating an image…");
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
        state.status = tr("Agent 正在合成语音…", "Agent is synthesizing speech…");
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
    char timing[128] = {};
    std::snprintf(timing, sizeof(timing),
                  tr("回答完成 · 准备 %.2fs · 首字 %.2fs · 总计 %.2fs",
                     "Complete · Prepare %.2fs · First token %.2fs · Total %.2fs"),
                  result.value.prepare_seconds, result.value.ttft_seconds,
                  result.value.total_seconds);
    finishChatTurn(timing);
}

inline void startChatRound() {
    state.streaming = true;
    state.streaming_text.clear();
    state.stream.clear();
    state.stream.snapshot(state.stream_seen, state.streaming_text);  // Resync version.
    state.streaming_text.clear();
    state.chat_stick_bottom = true;
    state.status = tr("正在生成回答…", "Generating response…");
    state.progress.store(0.02f);

    // The provider only understands user/assistant/system, so tool results
    // travel as user-role text (already prefixed with [工具结果]). Reasoning
    // stays out of the transcript, and think-only messages are skipped.
    std::vector<ChatMessage> history;
    history.reserve(state.messages.size() + 1);
    history.push_back({"system", kAgentSystemPrompt});
    for (std::size_t index = 0; index < state.messages.size(); ++index) {
        const auto& message = state.messages[index];
        if (message.text.empty() && message.image_path.empty()) continue;
        ChatMessage copy;
        copy.role = message.role == "tool" ? "user" : message.role;
        copy.text = message.text;
        // Media embeddings are expensive and the SYCL compatibility path
        // encodes vision on CPU. Only the newest message can introduce media;
        // older turns are represented by their text and assistant response,
        // so every follow-up does not re-encode the same image again.
        if (index + 1 == state.messages.size()) {
            copy.image_path = message.image_path;
            copy.image_width = message.image_width;
            copy.image_height = message.image_height;
        }
        history.push_back(std::move(copy));
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
    if (state.chat_input.empty() && state.chat_attachment_path.empty()) return;
    if (state.chat_preloading) {
        state.chat_submit_queued = true;
        state.status = tr("模型准备中 · 请求已排队", "Model is loading · Request queued");
        return;
    }
    if (state.busy) return;
    ChatMessage message;
    message.role = "user";
    message.text = state.chat_input;
    message.image_path = state.chat_attachment_path;
    message.image_width = state.chat_attachment_width;
    message.image_height = state.chat_attachment_height;
    state.messages.push_back(std::move(message));
    state.chat_input.clear();
    clearChatImage();
    state.agent_rounds = 0;
    beginTask(tr("正在生成回答…", "Generating response…"));
    startChatRound();
}

// ---- Speech / image pages ----

inline void submitSpeech() {
    if (state.busy || state.speech_text.empty()) return;
    if (state.tts_voice_path.empty()) return finishTaskError(tr("请选择参考声音", "Choose a reference voice"));
    if (state.tts_reference_text.empty()) return finishTaskError(tr("请填写参考音频中实际说出的文字", "Enter the exact reference transcript"));
    const std::string input = state.speech_text;
    const float speed = state.speech_speed;
    const int backend = state.backend;
    beginTask(tr("正在合成语音…", "Synthesizing speech…"));
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
            state.status = tr("语音已生成", "Speech generated");
            EngineService::playAudio(state.audio_path);
        });
}

inline void submitAsr() {
    if (state.busy || state.asr_audio_path.empty()) return;
    if (state.asr_model.empty()) state.asr_model = firstModel(ModelKind::Asr);
    if (state.asr_model.empty()) return finishTaskError(tr("没有找到 Whisper Q4 模型", "Whisper Q4 model not found"));
    const std::string input = state.asr_audio_path;
    const std::string language = state.asr_language;
    const bool translate = state.asr_translate;
    // The current shared-GGML Whisper CUDA state path crashes during reset on
    // Windows. Keep the release demo on the verified CPU path until that
    // provider-level issue is resolved; other pages still honor state.backend.
    beginTask(tr("Whisper 正在使用 CPU 识别语音…", "Whisper is transcribing on CPU…"));
    core::async::restart("forge.ui.asr",
        [input, language, translate] {
            return guardedRun([&] {
                return EngineService::transcribe(input, language, translate, kBackendCpu,
                                                 &state.progress);
            });
        },
        [](core::async::Result<AsrResult> result) {
            if (!result.ok) return finishTaskError(result.error);
            state.asr_transcript = std::move(result.value.text);
            state.asr_detected_language = std::move(result.value.language);
            state.busy = false;
            state.has_error = false;
            state.status = tr("语音识别完成", "Transcription complete");
        });
}

inline void submitImage() {
    if (state.busy || state.image_prompt.empty()) return;
    const std::string prompt = state.image_prompt;
    const int steps = state.image_steps;
    const int backend = state.backend;
    beginTask(tr("正在生成图片…", "Generating image…"));
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
                state.status = tr("已停止", "Stopped");
                return;
            }
            state.image_path = std::move(result.value.path);
            state.image_width = result.value.width;
            state.image_height = result.value.height;
            state.status = tr("图片已生成", "Image generated");
        });
}

inline void submitVision() {
    if (state.busy || state.vision_input_path.empty()) return;
    if (state.vision_task == VisionTask::StereoDepth && state.vision_right_path.empty()) {
        return finishTaskError(tr("双目深度需要左、右两张校正后的图片", "Stereo depth requires rectified left and right images"));
    }
    if (state.vision_model.empty()) state.vision_model = firstVisionModel(state.vision_task);
    if (state.vision_model.empty()) return finishTaskError(tr("没有找到适用于当前任务的 Q4 模型", "No Q4 model found for this task"));
    beginTask(tr("正在分析图片…", "Analyzing image…"));
    const auto task = state.vision_task;
    const auto model = state.vision_model;
    const auto input = state.vision_input_path;
    const auto right = state.vision_right_path;
    const int backend = state.backend;
    const float score = state.vision_score_threshold;
    const float iou = state.vision_iou_threshold;
    core::async::restart("forge.ui.vision",
        [task, model, input, right, backend, score, iou] {
            return guardedRun([&] {
                return VisionService::analyze(task, model, input, right, backend,
                                              score, iou, &state.progress);
            });
        },
        [](core::async::Result<VisionResult> result) {
            state.busy = false;
            state.progress.store(0.0f);
            if (!result.ok) return finishTaskError(result.error);
            state.has_error = false;
            state.status = tr("分析完成", "Analysis complete");
            state.vision_output_path = result.value.path;
            state.vision_summary = result.value.summary;
        });
}

inline void selectVisionTask(VisionTask task) {
    if (state.busy || state.vision_task == task) return;
    state.vision_task = task;
    state.vision_model = firstVisionModel(task);
    state.vision_output_path.clear();
    state.vision_summary.clear();
    state.has_error = false;
    state.status = tr("请选择图片", "Choose an image");
}

inline void submitReleaseEngines() {
    if (state.busy || !EngineService::loaded().any()) return;
    beginTask(tr("正在释放模型…", "Releasing models…"));
    core::async::restart("forge.ui.release",
        [] { return guardedRun([] { return EngineService::releaseAll(); }); },
        [](core::async::Result<void> result) {
            state.busy = false;
            state.has_error = !result.ok;
            state.status = result.ok ? tr("模型已释放", "Models released")
                                     : tr("释放失败: ", "Release failed: ") + result.error;
        });
}

// Stop button: chat stops at the next streamed token; an in-flight image
// generation (from the image page or an agent tool round) is asked to cancel
// via the provider. Speech has no cancellation path.
inline void requestCancel() {
    if (!state.busy) return;
    state.cancel_requested.store(true);
    EngineService::requestImageCancel();  // No-op unless an image is in flight.
    state.status = tr("正在停止…", "Stopping…");
}

inline void switchBackend() {
    state.backend = (state.backend + 1) % kBackendCount;
    state.has_error = false;
    state.status = tr("后端已切换为 ", "Backend switched to ") +
                   std::string(kBackendNames[state.backend]) +
                   tr(" · 下次生成时生效", " · Applies to the next task");
}

} // namespace app
