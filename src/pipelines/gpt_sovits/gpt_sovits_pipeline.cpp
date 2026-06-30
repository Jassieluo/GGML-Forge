#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include "gpt_sovits_internal.h"
#include "dsp.h"
#include "frontends/gpt_sovits/text_utils.h"
#include "phonemizer.h"
#include "symbols.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdlib>
#include <filesystem>

static void set_env_var(const std::string& name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

static std::string get_env_var(const std::string& name) {
#ifdef _WIN32
    char* buf = nullptr;
    size_t sz = 0;
    if (_dupenv_s(&buf, &sz, name.c_str()) == 0 && buf != nullptr) {
        std::string res(buf);
        free(buf);
        return res;
    }
    return "";
#else
    const char* val = std::getenv(name.c_str());
    return val ? std::string(val) : "";
#endif
}

namespace gpt_sovits {

bool g_log_enabled = true;

static void safe_ggml_backend_tensor_get(const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    if (!tensor) return;
    if (tensor->buffer == nullptr) {
        std::memcpy(data, (const char*)tensor->data + offset, size);
    } else {
        ggml_backend_tensor_get(tensor, data, offset, size);
    }
}

static std::vector<float> get_tensor_as_float(struct ggml_tensor* tensor) {
    if (!tensor) return {};
    int64_t nelements = ggml_nelements(tensor);
    std::vector<float> data(nelements);
    if (tensor->buffer == nullptr) {
        std::memcpy(data.data(), tensor->data, nelements * sizeof(float));
    } else {
        ggml_backend_tensor_get(tensor, data.data(), 0, nelements * sizeof(float));
    }
    return data;
}

// CoutSilencer implementation
CoutSilencer::CoutSilencer(bool silencer_active) : old_buf(nullptr), active(silencer_active) {
    if (active) {
        old_buf = std::cout.rdbuf();
        std::cout.rdbuf(null_stream.rdbuf());
    }
}

CoutSilencer::~CoutSilencer() {
    if (active && old_buf) {
        std::cout.rdbuf(old_buf);
    }
}

#ifndef GPT_SOVITS_DEBUG_ENABLED
#define GPT_SOVITS_DEBUG_ENABLED() (std::getenv("GPT_SOVITS_DEBUG") != nullptr)
#endif

#define GPT_SOVITS_DEBUG_PRINT(x) do {  } while (0)

ggml_backend_t Impl::get_backend_for_device(const std::string& device_name) {
    std::string name = device_name;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);

    if (name == "cpu" || name.empty()) {
        if (device_backends.count("cpu") == 0) {
            ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
            if (b) {
                device_backends["cpu"] = b;
                auto * d = ggml_backend_get_device(b);
                if (d) {
                    auto * reg = ggml_backend_dev_backend_reg(d);
                    auto * fn = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_n_threads");
                    if (fn) fn(b, params.n_threads);
                }
                create_and_bind_shared_threadpool(b);
            }
        }
        return device_backends["cpu"];
    }

    if (device_backends.count(name) > 0) {
        return device_backends[name];
    }

    size_t n_devs = ggml_backend_dev_count();
    ggml_backend_dev_t found_dev = nullptr;

    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (d) {
            std::string dev_name = ggml_backend_dev_name(d);
            std::string dev_name_lower = dev_name;
            std::transform(dev_name_lower.begin(), dev_name_lower.end(), dev_name_lower.begin(), ::tolower);
            if (dev_name_lower.find(name) != std::string::npos || name.find(dev_name_lower) != std::string::npos) {
                found_dev = d;
                break;
            }
        }
    }

    if (found_dev) {
        ggml_backend_t b = ggml_backend_dev_init(found_dev, nullptr);
        if (b) {
            device_backends[name] = b;
            return b;
        }
    }

    if (name.find("gpu") != std::string::npos || name.find("cuda") != std::string::npos || name.find("sycl") != std::string::npos) {
        for (size_t i = 0; i < n_devs; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (d) {
                std::string dev_name = ggml_backend_dev_name(d);
                if (dev_name.rfind("CUDA", 0) == 0 || dev_name.rfind("SYCL", 0) == 0) {
                    ggml_backend_t b = ggml_backend_dev_init(d, nullptr);
                    if (b) {
                        device_backends[name] = b;
                        return b;
                    }
                }
            }
        }
    }

    return get_backend_for_device("cpu");
}

void Impl::create_and_bind_shared_threadpool(ggml_backend_t backend) {
    if (!shared_cpu_threadpool) {
        auto * d = ggml_backend_get_device(backend);
        if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto * reg = ggml_backend_dev_backend_reg(d);
            auto * ggml_threadpool_new_fn = (struct ggml_threadpool * (*)(const struct ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
            if (ggml_threadpool_new_fn) {
                struct ggml_threadpool_params tpp = ggml_threadpool_params_default(params.n_threads);
                shared_cpu_threadpool = ggml_threadpool_new_fn(&tpp);
                if (shared_cpu_threadpool && GPT_SOVITS_DEBUG_ENABLED()) {
                    if (g_log_enabled) std::cout << "[GPT-SoVITS] Created shared CPU threadpool with " << params.n_threads << " threads" << std::endl;
                }
            }
        }
    }

    if (shared_cpu_threadpool) {
        auto * d = ggml_backend_get_device(backend);
        if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            auto * reg = ggml_backend_dev_backend_reg(d);
            auto * set_threadpool_fn = (void (*)(ggml_backend_t, struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (set_threadpool_fn) {
                set_threadpool_fn(backend, shared_cpu_threadpool);
            }
        }
    }
}

bool Impl::load_model(int model_type) {
    if (model_type < 0 || model_type >= 4) return false;
    ModelSlot& slot = slots[model_type];
    if (slot.is_loaded) return true;

    if (slot.path.empty()) {
        std::cerr << "[GPT-SoVITS load_model] Error: Model path for slot " << model_type << " is empty!" << std::endl;
        return false;
    }

    ggml_backend_t backend = get_backend_for_device(slot.device);
    if (!backend) {
        std::cerr << "[GPT-SoVITS load_model] Error: Could not resolve backend for device: '" << slot.device << "'" << std::endl;
        return false;
    }
    bool ok = false;

    if (model_type == 0) {
        if (!hubert) hubert = std::make_unique<HubertModel>();
        ok = hubert->load(slot.path, backend);
        vits_backend = backend;
    } else if (model_type == 1) {
        if (!bert) bert = std::make_unique<BertModel>();
        ok = bert->load(slot.path, backend);
        bert_backend = backend;
    } else if (model_type == 2) {
        if (!t2s) t2s = std::make_unique<T2SModel>();
        ok = t2s->load(slot.path, backend);
        t2s_backend = backend;
    } else if (model_type == 3) {
        if (!vits) vits = std::make_unique<VITSModel>();
        ok = vits->load(slot.path, backend);
        vits_target_backend = backend;
    }

    if (ok) {
        slot.is_loaded = true;
        ggml_ops_ext::install_ops_hook(backend);
    } else {
        std::cerr << "[GPT-SoVITS load_model] Failed to load model slot " << model_type << "." << std::endl;
    }
    return ok;
}

void Impl::offload_model(int model_type) {
    if (model_type < 0 || model_type >= 4) return;
    if (bypass_offload) return;
    ModelSlot& slot = slots[model_type];
    if (!slot.is_loaded) return;
    if (slot.is_resident) return;

    if (model_type == 0) {
        hubert.reset();
    } else if (model_type == 1) {
        bert.reset();
    } else if (model_type == 2) {
        t2s.reset();
    } else if (model_type == 3) {
        vits.reset();
    }
    slot.is_loaded = false;
}

void Impl::configure_sycl_cache_impl() {
    std::string s_persistent = get_env_var("SYCL_CACHE_PERSISTENT");
    if (!s_persistent.empty()) {
        if (g_log_enabled) std::cout << "[GPT-SoVITS SYCL Cache] Using programmatically configured settings: SYCL_CACHE_PERSISTENT="
                  << s_persistent << ", SYCL_CACHE_DIR=" << get_env_var("SYCL_CACHE_DIR") << std::endl;
        return;
    }

    std::string persistent_cfg = get_env_var("GPT_SOVITS_SYCL_CACHE_PERSISTENT");
    bool enable_cache = false;
    if (persistent_cfg == "1" || persistent_cfg == "true") {
        enable_cache = true;
    }

    if (!enable_cache) {
        set_env_var("SYCL_CACHE_PERSISTENT", "0");
        if (g_log_enabled) std::cout << "[GPT-SoVITS SYCL Cache] JIT persistent cache disabled via GPT_SOVITS_SYCL_CACHE_PERSISTENT." << std::endl;
        return;
    }

    set_env_var("SYCL_CACHE_PERSISTENT", "1");
    std::string cache_dir = get_env_var("GPT_SOVITS_SYCL_CACHE_DIR");
    if (cache_dir.empty()) {
        cache_dir = get_env_var("SYCL_CACHE_DIR");
    }

    if (cache_dir.empty()) {
        try {
            std::filesystem::path default_path = std::filesystem::current_path() / "sycl_cache";
            cache_dir = default_path.string();
        } catch (const std::exception& e) {
            cache_dir = "sycl_cache";
        }
    }

    set_env_var("SYCL_CACHE_DIR", cache_dir);
    try {
        std::filesystem::path p(cache_dir);
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p);
            if (g_log_enabled) std::cout << "[GPT-SoVITS SYCL Cache] Created JIT cache directory: " << std::filesystem::absolute(p).string() << std::endl;
        } else {
            if (g_log_enabled) std::cout << "[GPT-SoVITS SYCL Cache] Using existing JIT cache directory: " << std::filesystem::absolute(p).string() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[GPT-SoVITS SYCL Cache] Warning: Failed to check/create cache directory " << cache_dir << ": " << e.what() << std::endl;
    }
}

Impl::Impl(
    const char* dict_dir,
    const char* hubert_model_path,
    const char* bert_model_path,
    const char* t2s_model_path,
    const char* vits_model_path,
    int n_threads,
    int backend_mode,
    const char* device_name
) {
    configure_sycl_cache_impl();
    ggml_backend_load_all(); // Load backends unconditionally first

    if (dict_dir) params.dict_dir = dict_dir;
    if (hubert_model_path) params.hubert_model_path = hubert_model_path;
    if (bert_model_path) params.bert_model_path = bert_model_path;
    if (t2s_model_path) params.t2s_model_path = t2s_model_path;
    if (vits_model_path) params.vits_model_path = vits_model_path;
    params.n_threads = n_threads;
    params.use_gpu = (backend_mode > 0);

    frontend = std::make_unique<GPTSoVITSFrontend>(params.dict_dir);
    frontend->initialize();

    slots[0].path = hubert_model_path ? hubert_model_path : "";
    slots[1].path = bert_model_path ? bert_model_path : "";
    slots[2].path = t2s_model_path ? t2s_model_path : "";
    slots[3].path = vits_model_path ? vits_model_path : "";
    std::string static_dev = "cpu";
    std::string vits_target_dev = "cpu";
    std::string default_gpu_name = "";

    if (backend_mode > 0) {
        size_t n_devs = ggml_backend_dev_count();
        ggml_backend_dev_t gpu_dev = nullptr;
        if (device_name && strlen(device_name) > 0) {
            gpu_dev = ggml_backend_dev_by_name(device_name);
            if (!gpu_dev) {
                std::string target_dev(device_name);
                for (size_t i = 0; i < n_devs; ++i) {
                    ggml_backend_dev_t d = ggml_backend_dev_get(i);
                    if (d) {
                        std::string name = ggml_backend_dev_name(d);
                        std::string name_lower = name;
                        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                        std::string target_lower = target_dev;
                        std::transform(target_lower.begin(), target_lower.end(), target_lower.begin(), ::tolower);
                        if (name_lower.find(target_lower) != std::string::npos || target_lower.find(name_lower) != std::string::npos) {
                            gpu_dev = d;
                            break;
                        }
                    }
                }
            }
        }

        if (!gpu_dev) {
            for (size_t i = 0; i < n_devs; ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (d) {
                    std::string name = ggml_backend_dev_name(d);
                    if (name.rfind("CUDA", 0) == 0 || name.rfind("SYCL", 0) == 0) {
                        gpu_dev = d;
                        break;
                    }
                }
            }
        }

        if (gpu_dev) {
            default_gpu_name = ggml_backend_dev_name(gpu_dev);
        }
    }

    if (backend_mode == 3 && !default_gpu_name.empty()) {
        static_dev = "cpu";
        vits_target_dev = default_gpu_name;
    } else if (!default_gpu_name.empty()) {
        static_dev = default_gpu_name;
        if (backend_mode == 1) {
            vits_target_dev = default_gpu_name;
        } else {
            vits_target_dev = "cpu";
        }
    }
    slots[0].device = vits_target_dev;
    slots[1].device = static_dev;
    slots[2].device = static_dev;
    slots[3].device = vits_target_dev;

    bool is_sycl = (default_gpu_name.rfind("SYCL", 0) == 0);
    if (is_sycl) {
        slots[0].device = default_gpu_name;
        slots[1].device = default_gpu_name;
        slots[2].device = default_gpu_name;
        slots[3].device = default_gpu_name;
    }
    slots[0].is_resident = true;
    slots[1].is_resident = true;
    slots[2].is_resident = true;
    slots[3].is_resident = true;

    if (!slots[0].path.empty()) load_model(0);
    if (!slots[1].path.empty()) load_model(1);
    if (!slots[2].path.empty()) load_model(2);
    if (!slots[3].path.empty()) load_model(3);
}

Impl::~Impl() {
    if (vits_galloc) {
        ggml_gallocr_free(vits_galloc);
    }
    hubert.reset();
    bert.reset();
    t2s.reset();
    vits.reset();
    t2s_standby.reset();
    vits_standby.reset();

    if (shared_cpu_threadpool) {
        ggml_backend_t cpu_b = nullptr;
        if (device_backends.count("cpu") > 0) cpu_b = device_backends["cpu"];
        if (cpu_b) {
            auto * d = ggml_backend_get_device(cpu_b);
            if (d) {
                auto * reg = ggml_backend_dev_backend_reg(d);
                auto * free_fn = (void (*)(struct ggml_threadpool *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
                if (free_fn) {
                    free_fn(shared_cpu_threadpool);
                    shared_cpu_threadpool = nullptr;
                }
            }
        }
    }

    for (auto& pair : device_backends) {
        if (pair.second) {
            ggml_backend_free(pair.second);
        }
    }
    device_backends.clear();
}

} // namespace gpt_sovits

using namespace gpt_sovits;

void gpt_sovits_get_or_create_prompt_cache(
    gpt_sovits_engine_t engine,
    const char* cache_id,
    const float* ref_audio_data,
    size_t ref_audio_len,
    const char* ref_text,
    const char* ref_language

) {
    gpt_sovits::CoutSilencer silencer(!gpt_sovits::g_log_enabled);

    if (!engine || !cache_id || !ref_text || !ref_language) return;

    Impl* impl = (Impl*)engine;

    std::string cid(cache_id);

    auto it = impl->prompt_caches.find(cid);

    if (it != impl->prompt_caches.end()) {
        return;
    }

    impl->load_model(0); // Hubert
    impl->load_model(1); // BERT
    impl->load_model(3); // VITS

    {
        PromptCache cache;
        cache.prompt_text = ref_text;
        cache.prompt_lang = ref_language;
        // Initialize graph context early so it can be shared by mixed-mode processing
        struct ggml_init_params init_params = {
            /* .mem_size   = */ 512 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false
        };
        struct ggml_context* ctx_graph = ggml_init(init_params);
        // 1. Phonemes
        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 1: Processing phonemes..." << std::endl;
        FrontendResult front_res;
        impl->frontend->process(cache.prompt_text, cache.prompt_lang, impl->bert.get(), ctx_graph, impl->bert_backend, front_res);
        cache.prompt_phones = front_res.phones;
        cache.prompt_word2ph = front_res.word2ph;
        cache.bert_features = front_res.bert_features;

        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 1 finished. Phonemes count: " << front_res.phones.size() << std::endl;

        if (g_log_enabled) std::cout << "[GPT-SoVITS] Prompt Phones: ";
        for (const auto& ph : front_res.phones) std::cout << "'" << ph << "' ";
        if (g_log_enabled) std::cout << "\n";

        // 2. CNHuBERT codes
        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 2: Running CNHuBERT..." << std::endl;

        // Convert ref_audio to ggml_tensor with zero padding at the end matching PyTorch's padding (9600 samples)
        size_t pad_samples = 9600;
        size_t total_samples = ref_audio_len + pad_samples;

        std::vector<float> padded_audio(total_samples, 0.0f);
        std::memcpy(padded_audio.data(), ref_audio_data, ref_audio_len * sizeof(float));
        impl->hubert->input_audio.set(padded_audio.data(), total_samples * sizeof(float));

        struct ggml_tensor* input_audio_view = impl->hubert->input_audio.view_1d(ctx_graph, total_samples);
        struct ggml_tensor* ssl_content = impl->hubert->forward(ctx_graph, input_audio_view, impl->vits_backend);
        // Project and quantize using SoVITS VITS quantizer to get hubert_codes
        int n_frames = ssl_content->ne[1]; // seq_len
        cache.hubert_codes.resize(n_frames);

        struct ggml_tensor* cb = impl->vits->get_tensor("quantizer.vq.layers.0._codebook.embed");
        struct ggml_tensor* ssl_proj_w = impl->vits->get_tensor("ssl_proj.weight");
        struct ggml_tensor* ssl_proj_b = impl->vits->get_tensor("ssl_proj.bias");

        if (!cb || !ssl_proj_w || !ssl_proj_b) {
            std::cerr << "[GPT-SoVITS] Warning: Missing VITS quantizer GGUF tensors! Falling back to mock codes." << std::endl;
            for (int i = 0; i < n_frames; ++i) {
                cache.hubert_codes[i] = i % 1024; // Mock fallback
            }
        } else {
            const int code_dim = (int)cb->ne[0];
            const int codebook_size = (int)cb->ne[1];
            const int out_frames = (n_frames - 2) / 2 + 1;
            std::vector<float> ssl_content_cpu(code_dim * n_frames);
            safe_ggml_backend_tensor_get(ssl_content, ssl_content_cpu.data(), 0, code_dim * n_frames * sizeof(float));

            std::vector<float> cb_data = get_tensor_as_float(cb);
            std::vector<float> cb_norms(codebook_size, 0.0f);
            for (int k = 0; k < codebook_size; ++k) {
                const float* code = cb_data.data() + k * code_dim;
                float sum_sq = 0.0f;
                for (int i = 0; i < code_dim; ++i) {
                    sum_sq += code[i] * code[i];
                }
                cb_norms[k] = sum_sq;
            }

            std::vector<float> projected_data(code_dim * out_frames);
            std::vector<float> w_data = get_tensor_as_float(ssl_proj_w);
            std::vector<float> bias = get_tensor_as_float(ssl_proj_b);

            for (int t = 0; t < out_frames; ++t) {
                const float* x0 = ssl_content_cpu.data() + (2 * t + 0) * code_dim;
                const float* x1 = ssl_content_cpu.data() + (2 * t + 1) * code_dim;
                float* out = projected_data.data() + t * code_dim;

                for (int oc = 0; oc < code_dim; ++oc) {
                    float acc = bias[oc];
                    const float* w0 = w_data.data() + 0 + 2 * oc * code_dim;
                    const float* w1 = w_data.data() + 1 + 2 * oc * code_dim;
                    for (int ic = 0; ic < code_dim; ++ic) {
                        acc += x0[ic] * w0[2 * ic];
                        acc += x1[ic] * w1[2 * ic];
                    }
                    out[oc] = acc;
                }
            }

            cache.hubert_codes.resize(out_frames);
            for (int t = 0; t < out_frames; ++t) {
                float min_dist = 1e30f;
                int best_k = 0;
                const float* frame = projected_data.data() + t * code_dim;
                for (int k = 0; k < codebook_size; ++k) {
                    float dot_prod = 0.0f;
                    const float* code = cb_data.data() + k * code_dim;
                    for (int i = 0; i < code_dim; ++i) {
                        dot_prod += frame[i] * code[i];
                    }
                    float dist = cb_norms[k] - 2.0f * dot_prod;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_k = k;
                    }
                }
                cache.hubert_codes[t] = best_k;
            }
        }
        // 3. BERT Features already computed in Step 1

        // 4. Compute speaker embedding (ge) via ref_enc from reference audio STFT spectrogram
        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: Computing speaker embedding (ge) via ref_enc..." << std::endl;
        {
            int n_frames = 0;
            std::vector<float> ref_enc_input = dsp::compute_stft_spectrogram(ref_audio_data, ref_audio_len, n_frames);
            if (n_frames > 0) {
                const int n_ref_enc = 704;

                struct ggml_init_params ge_init_params = {
                    /* .mem_size   = */ 128 * 1024 * 1024,
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true
                };

                struct ggml_context* ctx_ge = ggml_init(ge_init_params);
                // Create mel_spec tensor [n_ref_enc, n_frames]

                struct ggml_tensor* mel_spec_tensor = ggml_new_tensor_2d(ctx_ge, GGML_TYPE_F32, n_ref_enc, n_frames);

                ggml_backend_buffer_t ge_input_buf = ggml_backend_alloc_ctx_tensors(ctx_ge, impl->vits_target_backend);

                if (ge_input_buf) {
                    ggml_backend_tensor_set(mel_spec_tensor, ref_enc_input.data(), 0, ref_enc_input.size() * sizeof(float));
                }
                // Create graph context for ref_enc compute

                struct ggml_init_params ge_graph_params = {
                    /* .mem_size   = */ 256 * 1024 * 1024,
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true
                };

                struct ggml_context* ctx_ge_graph = ggml_init(ge_graph_params);

                struct ggml_tensor* ge_tensor = impl->vits->compute_speaker_embedding(ctx_ge_graph, mel_spec_tensor, impl->vits_target_backend);

                if (ge_tensor) {
                    struct ggml_cgraph* ge_graph = ggml_new_graph_custom(ctx_ge_graph, 65536, false);

                    ggml_build_forward_expand(ge_graph, ge_tensor);

                    ggml_backend_buffer_t ge_buf = ggml_backend_alloc_ctx_tensors(ctx_ge_graph, impl->vits_target_backend);

                    if (ge_buf) {
                        ggml_backend_graph_compute(impl->vits_target_backend, ge_graph);
                        // Extract ge (speaker embedding) [512] from ge_tensor [1, 512] or [512]
                        int64_t ge_nelems = ggml_nelements(ge_tensor);
                        cache.speaker_embedding.resize(ge_nelems);
                        ggml_backend_tensor_get(ge_tensor, cache.speaker_embedding.data(), 0, ge_nelems * sizeof(float));
                        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: Speaker embedding computed! shape=[" << ge_tensor->ne[0] << ", " << ge_tensor->ne[1] << "]" << std::endl;
                        float ge_sum = 0.0f;
                        for (float v : cache.speaker_embedding) ge_sum += std::abs(v);
                        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: ge L1 norm=" << ge_sum << std::endl;
                        ggml_backend_buffer_free(ge_buf);
                    }
                }
                ggml_free(ctx_ge_graph);
                if (ge_input_buf) ggml_backend_buffer_free(ge_input_buf);
                ggml_free(ctx_ge);
            } else {
                if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: Warning: ref audio too short for STFT (" << ref_audio_len << " samples). Using zero ge." << std::endl;
                cache.speaker_embedding.assign(512, 0.0f);
            }
        }

        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: ge computation complete." << std::endl;
        // Clean up graph context

        ggml_free(ctx_graph);
        impl->prompt_caches[cid] = cache;
    }

    impl->offload_model(0);
    impl->offload_model(1);
    impl->offload_model(3);
}

static const float* gpt_sovits_synthesize_single_segment_with_cache(
    gpt_sovits_engine_t engine,
    const char* text,
    const char* language,
    const char* cache_id,
    float speed,
    int* out_num_samples

) {
    if (out_num_samples) *out_num_samples = 0;
    if (!engine || !text || !language || !cache_id) return nullptr;
    Impl* impl = (Impl*)engine;
    impl->load_model(1); // BERT
    impl->load_model(2); // T2S
    impl->load_model(3); // VITS
    phonemizer::PhonemizerResult target_res;
    struct ggml_tensor* target_bert_out = nullptr;
    std::string cid(cache_id);
    auto it = impl->prompt_caches.find(cid);
    if (it == impl->prompt_caches.end()) {
        std::cerr << "[GPT-SoVITS] Error: Prompt cache ID not found: " << cid << "\n";
        impl->offload_model(1);
        impl->offload_model(2);
        impl->offload_model(3);
        return nullptr;
    }
    const PromptCache& cached_prompt = it->second;
    // 1. Process target text or load overrides
    std::vector<int32_t> target_phone_ids;
    std::vector<int32_t> prompt_phone_ids;
    std::vector<float> fused_bert_aligned;
    int target_len = 0;

    int prompt_len = (int)cached_prompt.prompt_phones.size();
    int text_len = 0;
    bool is_overridden = false;
    const char* override_phones = std::getenv("T2S_OVERRIDE_PHONES_FILE");
    const char* override_bert = std::getenv("T2S_OVERRIDE_BERT_FILE");

    if (override_phones && override_bert) {
        if (g_log_enabled) std::cout << "[GPT-SoVITS] T2S_OVERRIDE_PHONES_FILE and T2S_OVERRIDE_BERT_FILE are set. Loading preprocessed features..." << std::endl;
        // Load all phones
        std::ifstream f_phones(override_phones, std::ios::binary);
        if (f_phones.is_open()) {
            f_phones.seekg(0, std::ios::end);
            size_t size = f_phones.tellg();
            f_phones.seekg(0, std::ios::beg);
            std::vector<int32_t> py_all_phones(size / sizeof(int32_t));
            f_phones.read(reinterpret_cast<char*>(py_all_phones.data()), size);
            text_len = (int)py_all_phones.size();
            target_len = text_len - prompt_len;
            prompt_phone_ids.assign(py_all_phones.begin(), py_all_phones.begin() + prompt_len);
            target_phone_ids.assign(py_all_phones.begin() + prompt_len, py_all_phones.end());
            if (g_log_enabled) std::cout << "[GPT-SoVITS] Loaded " << text_len << " phones. prompt_len=" << prompt_len << ", target_len=" << target_len << std::endl;
        } else {
            std::cerr << "[GPT-SoVITS] Error: Failed to open override phones file: " << override_phones << std::endl;
        }
        // Load BERT
        std::ifstream f_bert(override_bert, std::ios::binary);
        if (f_bert.is_open()) {
            f_bert.seekg(0, std::ios::end);
            size_t size = f_bert.tellg();
            f_bert.seekg(0, std::ios::beg);
            fused_bert_aligned.resize(size / sizeof(float));
            f_bert.read(reinterpret_cast<char*>(fused_bert_aligned.data()), size);
            if (g_log_enabled) std::cout << "[GPT-SoVITS] Loaded " << fused_bert_aligned.size() << " floats from override BERT file." << std::endl;
            is_overridden = true;
        } else {
            std::cerr << "[GPT-SoVITS] Error: Failed to open override BERT file: " << override_bert << std::endl;
        }
    }
    struct ggml_context* ctx_graph = nullptr;
    struct ggml_tensor* bert_features_tensor = nullptr;

    if (is_overridden) {
        // Create a minimal graph context for bert_features_tensor
        struct ggml_init_params init_params = {
            /* .mem_size   = */ 64 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false
        };
        ctx_graph = ggml_init(init_params);
        bert_features_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, text_len);
        std::memcpy(bert_features_tensor->data, fused_bert_aligned.data(), 1024 * text_len * sizeof(float));
    } else {
        int64_t t_phonemizer_bert_start = ggml_time_us();

        std::string lang_str(language);
        struct ggml_init_params init_params = {
            /* .mem_size   = */ 256 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false
        };
        ctx_graph = ggml_init(init_params);

        std::vector<float> target_bert_aligned;
        FrontendResult front_res;
        impl->frontend->process(std::string(text), lang_str, impl->bert.get(), ctx_graph, impl->bert_backend, front_res);

        target_res.phones = front_res.phones;
        target_res.word2ph = front_res.word2ph;
        target_bert_aligned = front_res.bert_features;
        target_phone_ids = front_res.phone_ids;

        prompt_phone_ids.clear();
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[Pipeline Debug] Mapping prompt phone strings to IDs:\n";
        }
        for (const auto& ph : cached_prompt.prompt_phones) {
            int32_t id = impl->frontend->phone_to_id(ph);
            prompt_phone_ids.push_back(id);
            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "  ph='" << ph << "' len=" << ph.length() << " -> ID=" << id << "\n";
            }
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::fflush(stdout);
        }

        if (target_phone_ids.empty()) {
            std::cerr << "[GPT-SoVITS] Error: Target text contains no valid phonemes! Cannot synthesize.\n";
            ggml_free(ctx_graph);
            impl->offload_model(1);
            impl->offload_model(2);
            impl->offload_model(3);
            return nullptr;
        }

        target_len = target_phone_ids.size();
        text_len = prompt_len + target_len;

        fused_bert_aligned.assign(1024 * text_len, 0.0f);
        if (!cached_prompt.bert_features.empty()) {
            std::memcpy(fused_bert_aligned.data(), cached_prompt.bert_features.data(), 1024 * prompt_len * sizeof(float));
        }
        if (!target_bert_aligned.empty()) {
            std::memcpy(fused_bert_aligned.data() + 1024 * prompt_len, target_bert_aligned.data(), 1024 * target_len * sizeof(float));
        }

        bert_features_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, text_len);
        std::memcpy(bert_features_tensor->data, fused_bert_aligned.data(), 1024 * text_len * sizeof(float));
        if (lang_str == "zh" || lang_str == "zh_en") {
            target_bert_out = bert_features_tensor; // set to non-null
        }

        int64_t t_phonemizer_bert_end = ggml_time_us();
    }

    // 3. Predict semantic codes using T2S (autoregressive transformer decoder)
    std::vector<int32_t> pred_semantics;

    const char* override_file = std::getenv("T2S_OVERRIDE_CODES_FILE");

    if (override_file) {
        if (g_log_enabled) std::cout << "[GPT-SoVITS] T2S_OVERRIDE_CODES_FILE is set. Loading codes from: " << override_file << std::endl;
        std::ifstream f(override_file);
        if (f.is_open()) {
            std::string line;
            if (std::getline(f, line)) {
                std::stringstream ss(line);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    if (!token.empty()) {
                        pred_semantics.push_back(std::stoi(token));
                    }
                }
            }

            if (g_log_enabled) std::cout << "[GPT-SoVITS] Loaded " << pred_semantics.size() << " override semantic codes." << std::endl;
        } else {
            std::cerr << "[GPT-SoVITS] Error: Failed to open override codes file: " << override_file << std::endl;
        }
    }

    if (pred_semantics.empty()) {
        const char* override_prompt_semantic = std::getenv("T2S_OVERRIDE_PROMPT_SEMANTIC_FILE");
        std::vector<int32_t> hubert_codes = cached_prompt.hubert_codes;
        if (override_prompt_semantic) {
            if (g_log_enabled) std::cout << "[GPT-SoVITS] Loading override prompt semantics from: " << override_prompt_semantic << std::endl;
            std::ifstream f_prompt(override_prompt_semantic, std::ios::binary);
            if (f_prompt.is_open()) {
                f_prompt.seekg(0, std::ios::end);
                size_t size = f_prompt.tellg();
                f_prompt.seekg(0, std::ios::beg);
                hubert_codes.resize(size / sizeof(int32_t));
                f_prompt.read(reinterpret_cast<char*>(hubert_codes.data()), size);
                if (g_log_enabled) std::cout << "[GPT-SoVITS] Loaded " << hubert_codes.size() << " prompt semantic tokens." << std::endl;
            } else {
                std::cerr << "[GPT-SoVITS] Error: Failed to open override prompt semantic file: " << override_prompt_semantic << std::endl;
            }
        }

        int64_t t_t2s_start = ggml_time_us();

        pred_semantics = impl->t2s->forward(
            ctx_graph,
            prompt_phone_ids,
            target_phone_ids,
            hubert_codes,
            bert_features_tensor,
            target_res.word2ph,
            512, // max_len
            impl->t2s_backend // Run T2S on CPU/GPU depending on backend configuration
        );

        int64_t t_t2s_end = ggml_time_us();
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[GPT-SoVITS Debug] Generated tokens count: " << pred_semantics.size() << "\n";
            std::cout << "[GPT-SoVITS Debug] First 20 tokens: ";
            for (size_t i = 0; i < std::min((size_t)20, pred_semantics.size()); ++i) {
                std::cout << pred_semantics[i] << " ";
            }
            std::cout << "\n";
        }
    }
    {
        std::ofstream tokens_file("scratch/cpp_t2s_tokens.bin", std::ios::binary);
        if (tokens_file.is_open()) {
            tokens_file.write(reinterpret_cast<const char*>(pred_semantics.data()), pred_semantics.size() * sizeof(int32_t));
            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "[T2S Dump] Saved " << pred_semantics.size() << " semantic tokens to scratch/cpp_t2s_tokens.bin\n";
            }
        }
    }
    // Dump T2S outputs for reverse cross-testing
    // 4. Run SoVITS VITS Decoder to synthesize audio
    // Create dedicated context for VITS execution with no_alloc = true to allow backend allocation.

    // We use a single context for both inputs and model graph to ensure broad backend compatibility (including SYCL/CUDA/CPU).

    struct ggml_init_params vits_init_params = {
        /* .mem_size   = */ (size_t)1544 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };

    struct ggml_context* ctx_vits = ggml_init(vits_init_params);
    int word2ph_size = is_overridden ? 1 : (int)target_res.word2ph.size();
    int bert_out_len = is_overridden ? target_len : (int)target_res.phones.size();
    
    // Set pre-allocated inputs on the VITS model
    impl->vits->phone_ids.set(target_phone_ids.data(), target_phone_ids.size() * sizeof(int32_t));
    std::vector<int32_t> dummy_word2ph(1, 1);
    const int32_t* word2ph_ptr = is_overridden ? dummy_word2ph.data() : target_res.word2ph.data();
    impl->vits->word2ph.set(word2ph_ptr, word2ph_size * sizeof(int32_t));
    impl->vits->prompt_semantics.set(pred_semantics.data(), pred_semantics.size() * sizeof(int32_t));
    
    if (is_overridden || target_bert_out) {
        impl->vits->bert_features.set(fused_bert_aligned.data() + 1024 * prompt_len, target_len * 1024 * sizeof(float));
    } else {
        std::vector<float> zero_bert(1024 * bert_out_len, 0.0f);
        impl->vits->bert_features.set(zero_bert.data(), zero_bert.size() * sizeof(float));
    }

    const int ge_size = (int)cached_prompt.speaker_embedding.size();
    if (ge_size > 0) {
        impl->vits->refer_audio.set(cached_prompt.speaker_embedding.data(), ge_size * sizeof(float));
    } else {
        std::vector<float> zero_ge(512, 0.0f);
        impl->vits->refer_audio.set(zero_ge.data(), 512 * sizeof(float));
    }

    // Get input tensor views
    struct ggml_tensor* target_phone_tensor = impl->vits->phone_ids.view_1d(ctx_vits, target_phone_ids.size());
    struct ggml_tensor* target_word2ph_tensor = impl->vits->word2ph.view_1d(ctx_vits, word2ph_size);
    struct ggml_tensor* pred_semantics_tensor = impl->vits->prompt_semantics.view_1d(ctx_vits, pred_semantics.size());
    struct ggml_tensor* target_bert_out_gpu = impl->vits->bert_features.view_2d(ctx_vits, 1024, bert_out_len);
    struct ggml_tensor* ge_tensor = impl->vits->refer_audio.view_2d(ctx_vits, 512, 1);

    if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: ge_size=" << ge_size << ", calling VITS forward with cached speaker embedding..." << std::endl;

    int64_t t_vits_start = ggml_time_us();

    struct ggml_tensor* synth_audio = impl->vits->forward(
        ctx_vits,
        target_phone_tensor,
        nullptr, // phone_lengths
        target_word2ph_tensor,
        target_bert_out_gpu,
        pred_semantics_tensor,
        ge_tensor, // pass speaker embedding (ge) as refer_audio
        speed,
        impl->vits_target_backend
    );
    // 5. Build and evaluate the graph using backend

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_vits, 262144, false);

    ggml_build_forward_expand(gf, synth_audio);
    // Use persistent or newly created graph allocator for memory planning and overlaying intermediate activations

    if (!impl->vits_galloc) {
        impl->vits_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));
    }

    if (!impl->vits_galloc) {
        std::cerr << "[GPT-SoVITS] Error: Failed to create graph allocator (gallocr)!\n";
    } else {
        if (!ggml_gallocr_alloc_graph(impl->vits_galloc, gf)) {
            std::cerr << "[GPT-SoVITS] Error: Failed to allocate VITS graph using gallocr!\n";
        }
    }

    // Upload pending tensor data (flip matrices, interp indices, etc.)

    impl->vits->upload_pending_data(impl->vits_target_backend);
    // Compute on the backend

    ggml_backend_graph_compute(impl->vits_target_backend, gf);
    int64_t t_vits_end = ggml_time_us();

    // Convert synthesized tensor to final PCM float array in the resident memory
    int out_samples = (int)ggml_nelements(synth_audio);
    impl->last_synthesized_audio.resize(out_samples);
    ggml_backend_tensor_get(synth_audio, impl->last_synthesized_audio.data(), 0, out_samples * sizeof(float));

    // Cleanup VITS contexts and buffers (impl->vits_galloc is persistent, so DO NOT free it here)

    ggml_free(ctx_vits);
    ggml_free(ctx_graph);
    if (out_num_samples) *out_num_samples = out_samples;
    impl->offload_model(1);
    impl->offload_model(2);
    impl->offload_model(3);

    return impl->last_synthesized_audio.data();
}

const float* gpt_sovits_synthesize_with_cache(
    gpt_sovits_engine_t engine,
    const char* text,
    const char* language,
    const char* cache_id,
    float speed,
    int* out_num_samples

) {
    gpt_sovits::CoutSilencer silencer(!gpt_sovits::g_log_enabled);
    if (out_num_samples) *out_num_samples = 0;
    if (!engine || !text || !language || !cache_id) return nullptr;
    Impl* impl = (Impl*)engine;
    impl->load_model(1); // BERT
    impl->load_model(2); // T2S
    impl->load_model(3); // VITS
    impl->bypass_offload = true;
    std::string split_method = "cut5";
    const char* env_cut = std::getenv("T2S_CUT");

    if (env_cut) {
        std::string env_s(env_cut);
        if (env_s == "0" || env_s == "cut0") split_method = "cut0";
        else if (env_s == "1" || env_s == "cut1") split_method = "cut1";
        else if (env_s == "2" || env_s == "cut2") split_method = "cut2";
        else if (env_s == "3" || env_s == "cut3") split_method = "cut3";
        else if (env_s == "4" || env_s == "cut4") split_method = "cut4";
        else if (env_s == "5" || env_s == "cut5") split_method = "cut5";
    }

    std::vector<std::string> segments = impl->frontend->split_text(text, split_method);

    if (segments.empty()) {
        std::cerr << "[GPT-SoVITS] Warning: No segments parsed for synthesis.\n";
        impl->bypass_offload = false;

        impl->offload_model(1);
        impl->offload_model(2);
        impl->offload_model(3);
        return nullptr;
    }

    if (segments.size() == 1) {
        const float* res = gpt_sovits_synthesize_single_segment_with_cache(
            engine,
            segments[0].c_str(),
            language,
            cache_id,
            speed,
            out_num_samples
        );
        impl->bypass_offload = false;

        impl->offload_model(1);
        impl->offload_model(2);
        impl->offload_model(3);
        return res;
    }

    if (g_log_enabled) std::cout << "[GPT-SoVITS Split] Splitting paragraph into " << segments.size()
              << " segments using method '" << split_method << "'...\n";
    std::vector<float> combined_audio;
    const size_t pause_samples = 9600;

    for (size_t idx = 0; idx < segments.size(); ++idx) {
        const auto& seg_utf8 = segments[idx];

        if (g_log_enabled) std::cout << "[GPT-SoVITS Split] Synthesizing segment [" << (idx + 1) << "/" << segments.size()
                  << "]: \"" << seg_utf8 << "\"\n";
        int segment_samples = 0;

        const float* synth_audio = gpt_sovits_synthesize_single_segment_with_cache(
            engine,
            seg_utf8.c_str(),
            language,
            cache_id,
            speed,
            &segment_samples
        );

        if (synth_audio && segment_samples > 0) {
            combined_audio.insert(combined_audio.end(), synth_audio, synth_audio + segment_samples);

            if (idx + 1 < segments.size()) {
                combined_audio.insert(combined_audio.end(), pause_samples, 0.0f);
            }
        } else {
            std::cerr << "[GPT-SoVITS Split] Warning: Segment [" << (idx + 1) << "] synthesized empty or failed.\n";
        }
    }
    impl->last_synthesized_audio = combined_audio;

    if (out_num_samples) *out_num_samples = (int)impl->last_synthesized_audio.size();
    impl->bypass_offload = false;
    impl->offload_model(1);
    impl->offload_model(2);
    impl->offload_model(3);

    return impl->last_synthesized_audio.data();
}

const float* gpt_sovits_synthesize(
    gpt_sovits_engine_t engine,
    const char* text,
    const char* language,
    const float* ref_audio_data,
    size_t ref_audio_len,
    const char* ref_text,
    const char* ref_language,
    float speed,
    int* out_num_samples

) {
    gpt_sovits::CoutSilencer silencer(!gpt_sovits::g_log_enabled);

    if (out_num_samples) *out_num_samples = 0;

    Impl* impl = (Impl*)engine;
    impl->bypass_offload = true;
    impl->load_model(0);
    impl->load_model(1);
    impl->load_model(2);
    impl->load_model(3);
    // Standard pathway: extract prompt features on the fly, then synthesize using local cache

    gpt_sovits_get_or_create_prompt_cache(engine, "temp_prompt_cache", ref_audio_data, ref_audio_len, ref_text, ref_language);

    const float* res = gpt_sovits_synthesize_with_cache(engine, text, language, "temp_prompt_cache", speed, out_num_samples);
    impl->bypass_offload = false;
    impl->offload_model(0);
    impl->offload_model(1);
    impl->offload_model(2);
    impl->offload_model(3);
    return res;
}

const float* gpt_sovits_debug_vits_from_latent(
    gpt_sovits_engine_t engine,
    const float* latent_data,
    size_t latent_floats,
    const float* speaker_embedding,
    size_t speaker_floats,
    int* out_num_samples

) {
    if (out_num_samples) *out_num_samples = 0;

    if (!engine || !latent_data || latent_floats == 0 || (latent_floats % 192) != 0) {
        std::cerr << "[GPT-SoVITS] Invalid VITS debug inputs.\n";
        return nullptr;
    }

    Impl* impl = (Impl*)engine;

    impl->load_model(3);

    if (!impl->vits) {
        std::cerr << "[GPT-SoVITS] VITS model is not loaded.\n";
        return nullptr;
    }

    const int latent_frames = (int)(latent_floats / 192);

    if (speaker_embedding && speaker_floats != 512) {
        std::cerr << "[GPT-SoVITS] Speaker embedding must contain exactly 512 floats.\n";
        return nullptr;
    }

    struct ggml_init_params init_params = {
        /* .mem_size   = */ 512 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };

    struct ggml_context* ctx = ggml_init(init_params);

    if (!ctx) {
        std::cerr << "[GPT-SoVITS] Failed to create ggml context for VITS debug.\n";
        return nullptr;
    }

    struct ggml_tensor* latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 192, latent_frames);
    struct ggml_tensor* speaker = nullptr;

    if (speaker_embedding && speaker_floats == 512) {
        speaker = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    }

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx, impl->vits_target_backend);

    if (!input_buffer) {
        std::cerr << "[GPT-SoVITS] Failed to allocate backend tensors for VITS debug input.\n";

        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_tensor_set(latent, latent_data, 0, latent_floats * sizeof(float));

    if (speaker && speaker_embedding) {
        ggml_backend_tensor_set(speaker, speaker_embedding, 0, speaker_floats * sizeof(float));
    }

    struct ggml_tensor* audio = impl->vits->forward_from_latent(ctx, latent, speaker, impl->vits_target_backend);

    if (!audio) {
        std::cerr << "[GPT-SoVITS] VITS forward_from_latent returned null.\n";

        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        return nullptr;
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 262144, false);

    ggml_build_forward_expand(graph, audio);

    // Use the graph allocator (ggml_gallocr) for memory planning and overlaying intermediate activations

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));

    if (!galloc) {
        std::cerr << "[GPT-SoVITS] Failed to create graph allocator (gallocr) for VITS debug.\n";
        ggml_free(ctx);
        return nullptr;
    }

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        std::cerr << "[GPT-SoVITS] Failed to allocate VITS graph using gallocr for VITS debug.\n";
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return nullptr;
    }

    impl->vits->upload_pending_data(impl->vits_target_backend);
    ggml_backend_graph_compute(impl->vits_target_backend, graph);
    const int out_samples = (int)ggml_nelements(audio);
    impl->last_synthesized_audio.resize(out_samples);
    ggml_backend_tensor_get(audio, impl->last_synthesized_audio.data(), 0, out_samples * sizeof(float));

    if (out_num_samples) *out_num_samples = out_samples;

    if (galloc) {
        ggml_gallocr_free(galloc);
    }

    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    impl->offload_model(3);
    return impl->last_synthesized_audio.data();
}

const float* gpt_sovits_debug_full_pipeline(
    gpt_sovits_engine_t engine,
    const int* token_ids, size_t n_tokens,
    const int* phone_ids, size_t n_phones,
    const float* ge_data, size_t ge_size,
    float speed,
    int* out_num_samples

) {
    if (out_num_samples) *out_num_samples = 0;
    if (!engine || !token_ids || n_tokens == 0 || !phone_ids || n_phones == 0) return nullptr;
    Impl* impl = (Impl*)engine;
    impl->load_model(3);
    if (!impl->vits) return nullptr;

    // Use SINGLE context with two allocation passes (same pattern as debug_vits_from_latent)
    struct ggml_init_params init_params = { (size_t)16384*1024*1024, nullptr, true };
    struct ggml_context* ctx = ggml_init(init_params);
    struct ggml_tensor* sem_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_tokens);
    struct ggml_tensor* phone_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)n_phones);
    struct ggml_tensor* ge_tensor = nullptr;

    if (ge_data && ge_size > 0) {
        ge_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)ge_size);
    }
    // First alloc: input tensors only
    ggml_backend_buffer_t input_buf = ggml_backend_alloc_ctx_tensors(ctx, impl->vits_target_backend);
    ggml_backend_tensor_set(sem_tensor, token_ids, 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(phone_tensor, phone_ids, 0, n_phones * sizeof(int32_t));
    if (ge_tensor) {
        ggml_backend_tensor_set(ge_tensor, ge_data, 0, ge_size * sizeof(float));
    }
    // Enable encoder debug dumps via env var
    // Build graph in same context

    struct ggml_tensor* audio = impl->vits->forward(
        ctx, phone_tensor, nullptr, nullptr, nullptr,
        sem_tensor, ge_tensor, speed, impl->vits_target_backend);

    if (!audio) {
        ggml_backend_buffer_free(input_buf);

        ggml_free(ctx);
        return nullptr;
    }

    struct ggml_cgraph* graph = ggml_new_graph_custom(ctx, 524288, false);
    ggml_build_forward_expand(graph, audio);

    // Use the graph allocator (ggml_gallocr) for memory planning and overlaying intermediate activations
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));

    if (!galloc) {
        std::cerr << "[GPT-SoVITS] Failed to create graph allocator (gallocr) for full pipeline debug.\n";
        ggml_backend_buffer_free(input_buf);
        ggml_free(ctx);
        return nullptr;
    }

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        std::cerr << "[GPT-SoVITS] Failed to allocate VITS graph using gallocr for full pipeline debug.\n";
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(input_buf);
        ggml_free(ctx);
        return nullptr;
    }

    // Upload pending tensor data (flip matrices, interp indices, etc.)
    impl->vits->upload_pending_data(impl->vits_target_backend);
    ggml_backend_graph_compute(impl->vits_target_backend, graph);

    // Dump encoder intermediates
    int n_samples = (int)ggml_nelements(audio);
    impl->last_synthesized_audio.resize(n_samples);
    ggml_backend_tensor_get(audio, impl->last_synthesized_audio.data(), 0, n_samples * sizeof(float));
    if (out_num_samples) *out_num_samples = n_samples;

    if (galloc) {
        ggml_gallocr_free(galloc);
    }

    ggml_backend_buffer_free(input_buf);
    ggml_free(ctx);
    impl->offload_model(3);

    return impl->last_synthesized_audio.data();
}

const float* gpt_sovits_debug_ref_enc(
    gpt_sovits_engine_t engine,
    const float* mel_data,
    size_t mel_floats,
    int* out_dim

) {
    if (out_dim) *out_dim = 0;

    if (!engine || !mel_data || mel_floats == 0 || (mel_floats % 704) != 0) {
        std::cerr << "[GPT-SoVITS] Invalid ref_enc debug inputs.\n";
        return nullptr;
    }

    Impl* impl = (Impl*)engine;
    impl->load_model(3);

    if (!impl->vits) {
        std::cerr << "[GPT-SoVITS] VITS model is not loaded for ref_enc test.\n";
        return nullptr;
    }

    int T = (int)(mel_floats / 704);  // 704 = n_mel channels

    struct ggml_init_params init_params = {
        /* .mem_size   = */ 512 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true
    };

    struct ggml_context* ctx = ggml_init(init_params);
    if (!ctx) return nullptr;

    struct ggml_tensor* mel_spec = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 704, T);

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx, impl->vits_target_backend);

    if (!input_buffer) {
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_tensor_set(mel_spec, mel_data, 0, mel_floats * sizeof(float));
    struct ggml_tensor* ge = impl->vits->compute_speaker_embedding(ctx, mel_spec, impl->vits_target_backend);

    if (!ge) {
        ggml_backend_buffer_free(input_buffer);

        ggml_free(ctx);
        return nullptr;
    }

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ge);

    // Use the graph allocator (ggml_gallocr) for memory planning and overlaying intermediate activations
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));

    if (!galloc) {
        std::cerr << "[GPT-SoVITS] Failed to create graph allocator (gallocr) for ref_enc debug.\n";
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        return nullptr;
    }

    if (!ggml_gallocr_alloc_graph(galloc, graph)) {
        std::cerr << "[GPT-SoVITS] Failed to allocate graph using gallocr for ref_enc debug.\n";
        ggml_gallocr_free(galloc);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_graph_compute(impl->vits_target_backend, graph);

    // Dump debug tensors if VITS_ALIGNMENT is set
    int out_samples = (int)ggml_nelements(ge);
    impl->last_synthesized_audio.resize(out_samples);
    ggml_backend_tensor_get(ge, impl->last_synthesized_audio.data(), 0, out_samples * sizeof(float));

    if (out_dim) *out_dim = out_samples;

    if (galloc) {
        ggml_gallocr_free(galloc);
    }

    ggml_backend_buffer_free(input_buffer);
    ggml_free(ctx);
    impl->offload_model(3);
    return impl->last_synthesized_audio.data();
}

void gpt_sovits_set_model_config(
    gpt_sovits_engine_t engine,
    int model_type,
    const char* model_path,
    const char* device_name,
    bool is_resident

) {
    if (!engine || model_type < 0 || model_type >= 4) return;
    Impl* impl = (Impl*)engine;
    if (model_path) {
        impl->slots[model_type].path = model_path;
    }

    if (device_name) {
        impl->slots[model_type].device = device_name;
    }
    impl->slots[model_type].is_resident = is_resident;
}

bool gpt_sovits_load_model(gpt_sovits_engine_t engine, int model_type) {
    if (!engine || model_type < 0 || model_type >= 4) return false;
    Impl* impl = (Impl*)engine;
    return impl->load_model(model_type);
}

void gpt_sovits_offload_model(gpt_sovits_engine_t engine, int model_type) {
    if (!engine || model_type < 0 || model_type >= 4) return;

    Impl* impl = (Impl*)engine;
    bool prev_resident = impl->slots[model_type].is_resident;
    impl->slots[model_type].is_resident = false;

    impl->offload_model(model_type);
    impl->slots[model_type].is_resident = prev_resident;
}

bool gpt_sovits_is_model_loaded(gpt_sovits_engine_t engine, int model_type) {
    if (!engine || model_type < 0 || model_type >= 4) return false;

    Impl* impl = (Impl*)engine;
    return impl->slots[model_type].is_loaded;
}

void gpt_sovits_set_log_enabled(bool enabled) {
    g_log_enabled = enabled;
}

#include <mutex>

namespace ggml_ops_ext {
namespace cpu { void register_backend(); }
#ifdef GGML_USE_CUDA
namespace cuda { void register_backend(); }
#endif
#ifdef GGML_USE_SYCL
namespace sycl { void register_backend(); }
#endif
}

static void register_all_backends_once() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        ggml_ops_ext::cpu::register_backend();
#ifdef GGML_USE_CUDA
        ggml_ops_ext::cuda::register_backend();
#endif
#ifdef GGML_USE_SYCL
        ggml_ops_ext::sycl::register_backend();
#endif
    });
}

extern "C" {

void gpt_sovits_configure_sycl_cache(bool enable_cache, const char* cache_dir) {
    if (!enable_cache) {
        set_env_var("SYCL_CACHE_PERSISTENT", "0");
        std::cout << "[GPT-SoVITS SYCL Cache] JIT persistent cache disabled programmatically." << std::endl;
        return;
    }

    set_env_var("SYCL_CACHE_PERSISTENT", "1");
    std::string dir_str = cache_dir ? cache_dir : "";
    if (dir_str.empty()) {
        try {
            std::filesystem::path default_path = std::filesystem::current_path() / "sycl_cache";
            dir_str = default_path.string();
        } catch (const std::exception& e) {
            dir_str = "sycl_cache";
        }
    }

    set_env_var("SYCL_CACHE_DIR", dir_str);
    try {
        std::filesystem::path p(dir_str);
        if (!std::filesystem::exists(p)) {
            std::filesystem::create_directories(p);
            std::cout << "[GPT-SoVITS SYCL Cache] Created JIT cache directory programmatically: " << std::filesystem::absolute(p).string() << std::endl;
        } else {
            std::cout << "[GPT-SoVITS SYCL Cache] Using JIT cache directory programmatically: " << std::filesystem::absolute(p).string() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[GPT-SoVITS SYCL Cache] Warning: Failed to check/create cache directory " << dir_str << ": " << e.what() << std::endl;
    }
}

gpt_sovits_engine_t gpt_sovits_init(
    const char* dict_dir,
    const char* hubert_model_path,
    const char* bert_model_path,
    const char* t2s_model_path,
    const char* vits_model_path,
    int n_threads,
    bool use_gpu
) {
    return gpt_sovits_init_ext(
        dict_dir,
        hubert_model_path,
        bert_model_path,
        t2s_model_path,
        vits_model_path,
        n_threads,
        use_gpu ? 1 : 0
    );
}



gpt_sovits_engine_t gpt_sovits_init_ext(
    const char* dict_dir,
    const char* hubert_model_path,
    const char* bert_model_path,
    const char* t2s_model_path,
    const char* vits_model_path,
    int n_threads,
    int backend_mode
) {
    try {
        register_all_backends_once();
        Impl* engine = new Impl(
            dict_dir,
            hubert_model_path,
            bert_model_path,
            t2s_model_path,
            vits_model_path,
            n_threads,
            backend_mode,
            nullptr
        );
        return (gpt_sovits_engine_t)engine;
    } catch (const std::exception& e) {
        std::cerr << "[gpt_sovits_init_ext] Exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

gpt_sovits_engine_t gpt_sovits_init_with_device(
    const char* dict_dir,
    const char* hubert_model_path,
    const char* bert_model_path,
    const char* t2s_model_path,
    const char* vits_model_path,
    int n_threads,
    int backend_mode,
    const char* device_name
) {
    try {
        register_all_backends_once();
        Impl* engine = new Impl(
            dict_dir,
            hubert_model_path,
            bert_model_path,
            t2s_model_path,
            vits_model_path,
            n_threads,
            backend_mode,
            device_name
        );
        return (gpt_sovits_engine_t)engine;
    } catch (const std::exception& e) {
        std::cerr << "[gpt_sovits_init_with_device] Exception caught: " << e.what() << std::endl;
        return nullptr;
    }
}

void gpt_sovits_free(gpt_sovits_engine_t engine) {
    if (engine) {
        Impl* impl = (Impl*)engine;
        delete impl;
    }
}

bool gpt_sovits_load_speaker(
    gpt_sovits_engine_t engine,
    const char* t2s_model_path,
    const char* vits_model_path
) {
    if (!engine) return false;
    Impl* impl = (Impl*)engine;

    if (g_log_enabled) {
        std::cout << "[gpt_sovits_load_speaker] Hot-swapping speaker via slot configuration...\n";
    }

    if (t2s_model_path) {
        impl->slots[2].path = t2s_model_path;
        if (impl->slots[2].is_loaded || impl->slots[2].is_resident) {
            impl->slots[2].is_loaded = false; // force reload
            if (!impl->load_model(2)) {
                std::cerr << "[gpt_sovits_load_speaker] Failed to load new T2S model\n";
                return false;
            }
        }
    }

    if (vits_model_path) {
        impl->slots[3].path = vits_model_path;
        if (impl->slots[3].is_loaded || impl->slots[3].is_resident) {
            impl->slots[3].is_loaded = false; // force reload
            if (!impl->load_model(3)) {
                std::cerr << "[gpt_sovits_load_speaker] Failed to load new VITS model\n";
                return false;
            }
        }
    }

    if (g_log_enabled) {
        std::cout << "[gpt_sovits_load_speaker] Swapped successfully!\n";
    }
    return true;
}

} // extern "C"

