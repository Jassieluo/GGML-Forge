#include "providers/llm_provider.h"

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llm {
namespace {

class ScopedMtmdDevice {
public:
    explicit ScopedMtmdDevice(const std::string& device) {
        const char* existing = std::getenv("MTMD_BACKEND_DEVICE");
        if (existing) {
            had_value_ = true;
            previous_ = existing;
        }
        if (!device.empty() && device != "auto" && device != "cpu") {
#ifdef _WIN32
            _putenv_s("MTMD_BACKEND_DEVICE", device.c_str());
#else
            setenv("MTMD_BACKEND_DEVICE", device.c_str(), 1);
#endif
            changed_ = true;
        }
    }

    ~ScopedMtmdDevice() {
        if (!changed_) return;
#ifdef _WIN32
        _putenv_s("MTMD_BACKEND_DEVICE", had_value_ ? previous_.c_str() : "");
#else
        if (had_value_) setenv("MTMD_BACKEND_DEVICE", previous_.c_str(), 1);
        else unsetenv("MTMD_BACKEND_DEVICE");
#endif
    }

private:
    bool changed_ = false;
    bool had_value_ = false;
    std::string previous_;
};

class BackendLifetime {
public:
    BackendLifetime() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~BackendLifetime() { llama_backend_free(); }
};

std::shared_ptr<BackendLifetime> acquire_backend_lifetime() {
    static std::mutex mutex;
    static std::weak_ptr<BackendLifetime> weak;
    std::lock_guard<std::mutex> lock(mutex);
    auto lifetime = weak.lock();
    if (!lifetime) {
        lifetime = std::make_shared<BackendLifetime>();
        weak = lifetime;
    }
    return lifetime;
}

mtmd_context* load_multimodal_projector(
    const std::string& path, llama_model* model, const RuntimeConfig& runtime) {
    if (path.empty() || !model) return nullptr;
    // mtmd backend selection uses a process environment variable. Serialize
    // initialization so concurrent model loads cannot observe each other's
    // temporary device selection.
    static std::mutex mtmd_load_mutex;
    std::lock_guard<std::mutex> mtmd_load_lock(mtmd_load_mutex);
    std::string requested = runtime.device;
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool sycl = requested.rfind("sycl", 0) == 0;
    const bool cpu = requested == "cpu";
    ScopedMtmdDevice mtmd_device(sycl ? "cpu" : runtime.device);
    auto params = mtmd_context_params_default();
    // The combined CUDA+SYCL release currently hits an access violation after
    // Qwen3-VL image encoding when mtmd is forced onto SYCL. Keep generation
    // on SYCL and use CPU for the projector on that backend.
    params.use_gpu = runtime.n_gpu_layers != 0 && !sycl && !cpu;
    params.n_threads = static_cast<int>(runtime.n_threads);
    params.batch_max_tokens = static_cast<int32_t>(runtime.n_batch);
    return mtmd_init_from_file(path.c_str(), model, params);
}

class LlamaModel;

class LlamaSession final : public ILLMSession {
public:
    explicit LlamaSession(std::shared_ptr<LlamaModel> model);
    ~LlamaSession() override;

    bool ready() const { return context_ != nullptr; }
    bool reset() override;
    bool generate(const GenerationRequest& request, TextSink sink) override;
    bool generate_chat(
        const std::vector<ChatMessage>& messages,
        const GenerationRequest& parameters,
        TextSink sink) override;

private:
    bool generate_text(const GenerationRequest& request, TextSink sink);
    bool generate_content(const GenerationRequest& request, TextSink sink);
    bool sample(const GenerationRequest& request, TextSink sink);

    std::shared_ptr<LlamaModel> model_;
    llama_context* context_ = nullptr;
};

class LlamaModel final : public ILLMModel, public std::enable_shared_from_this<LlamaModel> {
public:
    LlamaModel(
        std::shared_ptr<BackendLifetime> lifetime,
        llama_model* model,
        std::string mmproj_path,
        RuntimeConfig config)
        : lifetime_(std::move(lifetime)), model_(model),
          mmproj_path_(std::move(mmproj_path)), config_(config) {}
    ~LlamaModel() override {
        if (multimodal_) mtmd_free(multimodal_);
        if (model_) llama_model_free(model_);
    }

    std::unique_ptr<ILLMSession> create_session() override {
        auto session = std::make_unique<LlamaSession>(shared_from_this());
        return session->ready() ? std::move(session) : nullptr;
    }

    llama_model* native_model() const { return model_; }
    mtmd_context* multimodal() const {
        if (multimodal_ || mmproj_path_.empty()) return multimodal_;
        std::lock_guard<std::mutex> lock(multimodal_load_mutex_);
        if (!multimodal_) {
            multimodal_ = load_multimodal_projector(mmproj_path_, model_, config_);
        }
        return multimodal_;
    }
    std::mutex& multimodal_mutex() { return multimodal_mutex_; }
    const RuntimeConfig& config() const { return config_; }
    Capabilities capabilities() const override {
        mtmd_context* context = multimodal();
        return {
            context && mtmd_support_vision(context),
            context && mtmd_support_audio(context),
        };
    }
    bool format_chat(const std::vector<ChatMessage>& messages, std::string& prompt) const override {
        if (messages.empty()) return false;
        std::vector<llama_chat_message> native;
        native.reserve(messages.size());
        for (const ChatMessage& message : messages) {
            native.push_back({message.role.c_str(), message.content.c_str()});
        }
        const char* model_template = llama_model_chat_template(model_, nullptr);
        int32_t size = llama_chat_apply_template(
            model_template, native.data(), native.size(), true, nullptr, 0);
        if (size <= 0) return false;
        prompt.resize(static_cast<size_t>(size));
        size = llama_chat_apply_template(
            model_template, native.data(), native.size(), true, prompt.data(), size);
        if (size < 0) { prompt.clear(); return false; }
        prompt.resize(static_cast<size_t>(size));
        return !prompt.empty();
    }

private:
    std::shared_ptr<BackendLifetime> lifetime_;
    llama_model* model_ = nullptr;
    std::string mmproj_path_;
    mutable mtmd_context* multimodal_ = nullptr;
    mutable std::mutex multimodal_load_mutex_;
    std::mutex multimodal_mutex_;
    RuntimeConfig config_;
};

LlamaSession::LlamaSession(std::shared_ptr<LlamaModel> model) : model_(std::move(model)) {
    auto params = llama_context_default_params();
    params.n_ctx = model_->config().n_ctx;
    params.n_batch = model_->config().n_batch;
    params.n_threads = static_cast<int32_t>(model_->config().n_threads);
    params.n_threads_batch = static_cast<int32_t>(model_->config().n_threads);
    context_ = llama_init_from_model(model_->native_model(), params);
    if (!context_) return;

    // Match common_init_from_params(): reserving the scheduler only builds the
    // graphs; the first decode still triggers SYCL device compilation. Execute
    // one tiny batch while the UI reports "preparing model" so the first user
    // request does not unexpectedly pay the JIT cost.
    llama_model* native = model_->native_model();
    const llama_vocab* vocab = llama_model_get_vocab(native);
    std::vector<llama_token> warmup_tokens;
    const llama_token bos = llama_vocab_bos(vocab);
    const llama_token eos = llama_vocab_eos(vocab);
    if (bos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(bos);
    if (eos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(eos);
    if (warmup_tokens.empty()) warmup_tokens.push_back(0);

    bool warmup_ok = true;
    if (llama_model_has_encoder(native)) {
        warmup_ok = llama_encode(
            context_, llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size())) == 0;
        llama_token decoder_start = llama_model_decoder_start_token(native);
        if (decoder_start == LLAMA_TOKEN_NULL) decoder_start = bos;
        warmup_tokens.assign(1, decoder_start);
    }
    if (warmup_ok && llama_model_has_decoder(native)) {
        warmup_ok = llama_decode(
            context_, llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size())) == 0;
    }
    llama_memory_clear(llama_get_memory(context_), true);
    llama_synchronize(context_);
    llama_perf_context_reset(context_);
    if (!warmup_ok) {
        llama_free(context_);
        context_ = nullptr;
    }
}

LlamaSession::~LlamaSession() {
    if (context_) llama_free(context_);
}

bool LlamaSession::reset() {
    if (!context_) return false;
    llama_memory_clear(llama_get_memory(context_), true);
    return true;
}

bool LlamaSession::generate(const GenerationRequest& request, TextSink sink) {
    return request.content.empty()
        ? generate_text(request, std::move(sink))
        : generate_content(request, std::move(sink));
}

bool LlamaSession::generate_chat(
    const std::vector<ChatMessage>& messages,
    const GenerationRequest& parameters,
    TextSink sink
) {
    if (messages.empty()) return false;
    std::vector<ChatMessage> templated = messages;
    std::vector<ContentPart> media;
    std::vector<std::string> placeholders;
    for (ChatMessage& message : templated) {
        if (message.parts.empty()) continue;
        message.content.clear();
        for (const ContentPart& part : message.parts) {
            if (!part.data || part.size == 0) return false;
            if (part.type == ContentPart::Type::Text) {
                message.content.append(static_cast<const char*>(part.data), part.size);
            } else {
                const std::string marker = "<|forge_media_" + std::to_string(media.size()) + "|>";
                message.content += marker;
                placeholders.push_back(marker);
                media.push_back(part);
            }
        }
    }
    GenerationRequest prepared = parameters;
    if (!model_->format_chat(templated, prepared.prompt)) return false;
    if (media.empty()) return generate_text(prepared, std::move(sink));

    std::vector<std::string> text_parts;
    text_parts.reserve(media.size() + 1);
    prepared.content.clear();
    prepared.content.reserve(media.size() * 2 + 1);
    size_t cursor = 0;
    for (size_t i = 0; i < media.size(); ++i) {
        const size_t marker = prepared.prompt.find(placeholders[i], cursor);
        if (marker == std::string::npos) return false;
        if (marker > cursor) {
            text_parts.push_back(prepared.prompt.substr(cursor, marker - cursor));
            const std::string& text = text_parts.back();
            prepared.content.push_back({ContentPart::Type::Text, text.data(), text.size()});
        }
        prepared.content.push_back(media[i]);
        cursor = marker + placeholders[i].size();
    }
    if (cursor < prepared.prompt.size()) {
        text_parts.push_back(prepared.prompt.substr(cursor));
        const std::string& text = text_parts.back();
        prepared.content.push_back({ContentPart::Type::Text, text.data(), text.size()});
    }
    prepared.prompt.clear();
    return generate_content(prepared, std::move(sink));
}

bool LlamaSession::generate_text(const GenerationRequest& request, TextSink sink) {
    if (!context_ || request.prompt.empty() || !sink || !reset()) return false;
    llama_model* model = model_->native_model();
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int32_t token_count = llama_tokenize(
        vocab, request.prompt.data(), request.prompt.size(), nullptr, 0, true, true);
    if (token_count >= 0) return false;

    std::vector<llama_token> tokens(static_cast<size_t>(-token_count));
    if (llama_tokenize(
            vocab, request.prompt.data(), request.prompt.size(), tokens.data(),
            static_cast<int32_t>(tokens.size()), true, true) < 0) {
        return false;
    }
    if (tokens.size() + static_cast<size_t>(request.max_tokens) > model_->config().n_ctx) return false;

    size_t offset = 0;
    while (offset < tokens.size()) {
        const int32_t count = static_cast<int32_t>(std::min<size_t>(
            model_->config().n_batch, tokens.size() - offset));
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, count);
        const int result = llama_model_has_encoder(model)
            ? llama_encode(context_, batch)
            : llama_decode(context_, batch);
        if (result != 0) return false;
        offset += static_cast<size_t>(count);
    }

    return sample(request, std::move(sink));
}

bool LlamaSession::generate_content(const GenerationRequest& request, TextSink sink) {
    mtmd_context* multimodal = model_->multimodal();
    if (!context_ || !multimodal || request.content.empty() || !sink || !reset()) return false;

    std::lock_guard<std::mutex> lock(model_->multimodal_mutex());
    std::string prompt;
    std::vector<mtmd::bitmap> bitmaps;
    bitmaps.reserve(request.content.size());
    const auto capabilities = model_->capabilities();

    for (const ContentPart& part : request.content) {
        if (!part.data || part.size == 0) return false;
        if (part.type == ContentPart::Type::Text) {
            prompt.append(static_cast<const char*>(part.data), part.size);
            continue;
        }
        if ((part.type == ContentPart::Type::Image && !capabilities.vision) ||
            (part.type == ContentPart::Type::Audio && !capabilities.audio)) {
            return false;
        }
        const auto decoded = mtmd_helper_bitmap_init_from_buf(
            multimodal,
            static_cast<const unsigned char*>(part.data),
            part.size,
            false);
        if (decoded.video_ctx) mtmd_helper_video_free(decoded.video_ctx);
        if (!decoded.bitmap) return false;
        const bool audio = mtmd_bitmap_is_audio(decoded.bitmap);
        if ((part.type == ContentPart::Type::Audio) != audio) {
            mtmd_bitmap_free(decoded.bitmap);
            return false;
        }
        prompt += mtmd_default_marker();
        bitmaps.emplace_back(decoded.bitmap);
    }
    if (prompt.empty()) return false;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    if (!chunks.ptr) return false;
    std::vector<const mtmd_bitmap*> bitmap_ptrs;
    bitmap_ptrs.reserve(bitmaps.size());
    for (const auto& bitmap : bitmaps) bitmap_ptrs.push_back(bitmap.ptr.get());
    const mtmd_input_text text{prompt.data(), prompt.size(), true, true};
    if (mtmd_tokenize(
            multimodal,
            chunks.ptr.get(),
            &text,
            bitmap_ptrs.data(),
            bitmap_ptrs.size()) != 0) {
        return false;
    }

    const llama_pos positions = mtmd_helper_get_n_pos(chunks.ptr.get());
    if (positions < 0 ||
        static_cast<uint64_t>(positions) + static_cast<uint64_t>(request.max_tokens) >
            model_->config().n_ctx) {
        return false;
    }
    llama_pos n_past = 0;
    if (mtmd_helper_eval_chunks(
            multimodal,
            context_,
            chunks.ptr.get(),
            0,
            0,
            static_cast<int32_t>(model_->config().n_batch),
            true,
            &n_past) != 0) {
        return false;
    }

    return sample(request, std::move(sink));
}

bool LlamaSession::sample(const GenerationRequest& request, TextSink sink) {
    llama_model* model = model_->native_model();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) return false;
    if (request.temperature <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(request.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(request.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed));
    }

    bool success = true;
    llama_token next = LLAMA_TOKEN_NULL;
    if (llama_model_has_encoder(model)) {
        next = llama_model_decoder_start_token(model);
        if (next == LLAMA_TOKEN_NULL) next = llama_vocab_bos(vocab);
    }

    for (int32_t generated = 0; generated < request.max_tokens; ++generated) {
        if (next != LLAMA_TOKEN_NULL) {
            llama_batch batch = llama_batch_get_one(&next, 1);
            if (llama_decode(context_, batch) != 0) { success = false; break; }
        }
        next = llama_sampler_sample(sampler, context_, -1);
        if (llama_vocab_is_eog(vocab, next)) break;

        std::vector<char> piece(256);
        int32_t size = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
        if (size < 0) {
            piece.resize(static_cast<size_t>(-size));
            size = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
        }
        if (size < 0 || !sink(piece.data(), static_cast<size_t>(size))) {
            success = size >= 0;
            break;
        }

        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(context_, batch) != 0) { success = false; break; }
        next = LLAMA_TOKEN_NULL;
    }

    llama_sampler_free(sampler);
    return success;
}

class LlamaCppProvider final : public ILLMProvider {
public:
    const char* name() const override { return "llama.cpp"; }

    std::shared_ptr<ILLMModel> load(const ModelConfig& config, const RuntimeConfig& runtime) const override {
        auto lifetime = acquire_backend_lifetime();
        auto params = llama_model_default_params();
        params.n_gpu_layers = runtime.n_gpu_layers;

        std::array<ggml_backend_dev_t, 2> selected_devices{};
        std::string requested = runtime.device;
        std::transform(requested.begin(), requested.end(), requested.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (requested == "cpu") {
            params.n_gpu_layers = 0;
        } else if (!requested.empty() && requested != "auto") {
            ggml_backend_dev_t device = ggml_backend_dev_by_name(runtime.device.c_str());
            if (!device) return nullptr;
            selected_devices[0] = device;
            selected_devices[1] = nullptr;
            params.devices = selected_devices.data();
            params.split_mode = LLAMA_SPLIT_MODE_NONE;
            params.main_gpu = 0;
        }
        llama_model* model = llama_model_load_from_file(config.model.c_str(), params);
        if (!model) return nullptr;

        // Keep the projector path but initialize mtmd only when capabilities
        // or media generation are first requested. Text-only startup should
        // not pay the large vision model load and warmup cost.
        return std::make_shared<LlamaModel>(
            std::move(lifetime), model, config.mmproj, runtime);
    }
};

[[maybe_unused]] const bool registered = [] {
    ProviderRegistry::get().register_provider("llama.cpp", [] {
        return std::make_unique<LlamaCppProvider>();
    });
    return true;
}();

} // namespace
} // namespace llm
