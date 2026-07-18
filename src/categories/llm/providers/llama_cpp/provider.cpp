#include "providers/llm_provider.h"

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llm {
namespace {

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

class LlamaModel;

class LlamaSession final : public ILLMSession {
public:
    explicit LlamaSession(std::shared_ptr<LlamaModel> model);
    ~LlamaSession() override;

    bool ready() const { return context_ != nullptr; }
    bool reset() override;
    bool generate(const GenerationRequest& request, TextSink sink) override;

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
        mtmd_context* multimodal,
        RuntimeConfig config)
        : lifetime_(std::move(lifetime)), model_(model), multimodal_(multimodal), config_(config) {}
    ~LlamaModel() override {
        if (multimodal_) mtmd_free(multimodal_);
        if (model_) llama_model_free(model_);
    }

    std::unique_ptr<ILLMSession> create_session() override {
        auto session = std::make_unique<LlamaSession>(shared_from_this());
        return session->ready() ? std::move(session) : nullptr;
    }

    llama_model* native_model() const { return model_; }
    mtmd_context* multimodal() const { return multimodal_; }
    std::mutex& multimodal_mutex() { return multimodal_mutex_; }
    const RuntimeConfig& config() const { return config_; }
    Capabilities capabilities() const override {
        return {
            multimodal_ && mtmd_support_vision(multimodal_),
            multimodal_ && mtmd_support_audio(multimodal_),
        };
    }

private:
    std::shared_ptr<BackendLifetime> lifetime_;
    llama_model* model_ = nullptr;
    mtmd_context* multimodal_ = nullptr;
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
        llama_model* model = llama_model_load_from_file(config.model.c_str(), params);
        if (!model) return nullptr;

        mtmd_context* multimodal = nullptr;
        if (!config.mmproj.empty()) {
            auto mtmd_params = mtmd_context_params_default();
            mtmd_params.use_gpu = runtime.n_gpu_layers != 0;
            mtmd_params.n_threads = static_cast<int>(runtime.n_threads);
            mtmd_params.batch_max_tokens = static_cast<int32_t>(runtime.n_batch);
            multimodal = mtmd_init_from_file(config.mmproj.c_str(), model, mtmd_params);
            if (!multimodal) {
                llama_model_free(model);
                return nullptr;
            }
        }
        return std::make_shared<LlamaModel>(std::move(lifetime), model, multimodal, runtime);
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
