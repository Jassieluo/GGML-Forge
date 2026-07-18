#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include "gpt_sovits_internal.h"
#include "provider.h"
#include "../tts_provider.h"
#include "dsp.h"
#include "providers/gpt_sovits/frontend/text_utils.h"
#include "providers/gpt_sovits/frontend/gpt_sovits_frontend.h"
#include "phonemizer.h"
#include "frontend/symbols.h"
#define GGML_COMMON_DECL_CPP
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ops/cpu.h"
#include "ggml-cpu.h"
#include "ggml-common.h"
#include "gguf.h"
#include "ggml-impl.h"
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
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <random>

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

template <typename Fn>
static void parallel_for_ranges(int count, int requested_threads, Fn&& fn) {
    const int worker_count = std::max(1, std::min(count, requested_threads));
    if (worker_count == 1) {
        fn(0, count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        const int begin = count * worker / worker_count;
        const int end = count * (worker + 1) / worker_count;
        workers.emplace_back([begin, end, &fn]() { fn(begin, end); });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
}

static std::string select_split_method() {
    if (const char* env_cut = std::getenv("T2S_CUT")) {
        const std::string value(env_cut);
        if (value == "0" || value == "cut0") return "cut0";
        if (value == "1" || value == "cut1") return "cut1";
        if (value == "2" || value == "cut2") return "cut2";
        if (value == "3" || value == "cut3") return "cut3";
        if (value == "4" || value == "cut4") return "cut4";
        if (value == "5" || value == "cut5") return "cut5";
    }

    return "cut5";
}

static int32_t get_backend_device_type(ggml_backend_t backend) {
    if (!backend) return 0; // CPU
    const char* name = ggml_backend_name(backend);
    if (!name) return 0;
    std::string sname(name);
    if (sname.find("CUDA") != std::string::npos) return 1;
    if (sname.find("SYCL") != std::string::npos) return 2;
    return 0; // CPU
}

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
    size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> raw_bytes(nbytes);

    if (tensor->buffer == nullptr) {
        std::memcpy(raw_bytes.data(), tensor->data, nbytes);
    } else {
        ggml_backend_tensor_get(tensor, raw_bytes.data(), 0, nbytes);
    }

    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(data.data(), raw_bytes.data(), nelements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* fp16_ptr = (const ggml_fp16_t*)raw_bytes.data();
        for (int64_t i = 0; i < nelements; ++i) {
            data[i] = ggml_fp16_to_fp32(fp16_ptr[i]);
        }
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        typedef struct {
            ggml_half d;
            int8_t qs[32];
        } block_q8_0_local;
        const block_q8_0_local* blocks = (const block_q8_0_local*)raw_bytes.data();
        int64_t nblocks = nelements / 32;
        for (int64_t b = 0; b < nblocks; ++b) {
            float d = ggml_fp16_to_fp32(blocks[b].d);
            for (int i = 0; i < 32; ++i) {
                data[b * 32 + i] = d * blocks[b].qs[i];
            }
        }
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        typedef struct {
            ggml_half d;
            uint8_t qs[16];
        } block_q4_0_local;
        const block_q4_0_local* blocks = (const block_q4_0_local*)raw_bytes.data();
        int64_t nblocks = nelements / 32;
        for (int64_t b = 0; b < nblocks; ++b) {
            float d = ggml_fp16_to_fp32(blocks[b].d);
            for (int i = 0; i < 16; ++i) {
                data[b * 32 + i] = d * ((blocks[b].qs[i] & 0x0F) - 8.0f);
                data[b * 32 + i + 16] = d * ((blocks[b].qs[i] >> 4) - 8.0f);
            }
        }
    } else {
        std::cerr << "[get_tensor_as_float] Error: Unsupported tensor type " << tensor->type << std::endl;
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
#define GPT_SOVITS_DEBUG_ENABLED() (gpt_sovits::is_debug_enabled())
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
                    ggml_ops_ext_cpu_set_n_threads(b, params.n_threads);
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

static int t2s_family_from_model(const T2SModel* model) {
    return model ? model->family : 0;
}

bool Impl::validate_t2s_vits_compatibility() const {
    if (defer_model_compat_validation) {
        return true;
    }
    if (!t2s || !vits) {
        return true;
    }

    const int expected_family = vits->profile.expected_t2s_family;
    const int actual_family = t2s_family_from_model(t2s.get());
    if (expected_family <= 0 || actual_family <= 0) {
        std::cerr << "[GPT-SoVITS load_model] Missing T2S/VITS family metadata"
                  << " (T2S version='" << t2s->version_string
                  << "', VITS profile='" << vits->profile.profile_id << "')." << std::endl;
        return false;
    }

    if (expected_family == actual_family) {
        if (g_log_enabled) {
            std::cout << "[GPT-SoVITS load_model] T2S/VITS compatibility OK"
                      << " (T2S family=v" << actual_family
                      << ", VITS expects=v" << expected_family << ")." << std::endl;
        }
        return true;
    }

    std::cerr << "[GPT-SoVITS load_model] Error: T2S/VITS family mismatch. T2S version='"
              << t2s->version_string << "' maps to family v" << actual_family
              << ", but VITS profile '" << vits->profile.exact_version
              << "' expects T2S family v" << expected_family << "." << std::endl;
    return false;
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
        if (!hubert && shared_static_artifacts && slot.is_resident && !GPT_SOVITS_DEBUG_ENABLED()) {
            const std::string key = slot.path + "\n" + slot.device;
            std::lock_guard<std::mutex> lock(shared_static_artifacts->mutex);
            hubert = shared_static_artifacts->hubert[key].lock();
            if (!hubert) {
                hubert = std::make_shared<HubertModel>();
                if (hubert->load(slot.path, backend)) shared_static_artifacts->hubert[key] = hubert;
                else hubert.reset();
            }
            ok = static_cast<bool>(hubert);
        } else {
            if (!hubert) hubert = std::make_shared<HubertModel>();
            ok = hubert->load(slot.path, backend);
        }
        vits_backend = backend;
    } else if (model_type == 1) {
        if (!bert && shared_static_artifacts && slot.is_resident && !GPT_SOVITS_DEBUG_ENABLED()) {
            const std::string key = slot.path + "\n" + slot.device;
            std::lock_guard<std::mutex> lock(shared_static_artifacts->mutex);
            bert = shared_static_artifacts->bert[key].lock();
            if (!bert) {
                bert = std::make_shared<BertModel>();
                if (bert->load(slot.path, backend)) shared_static_artifacts->bert[key] = bert;
                else bert.reset();
            }
            ok = static_cast<bool>(bert);
        } else {
            if (!bert) bert = std::make_shared<BertModel>();
            ok = bert->load(slot.path, backend);
        }
        bert_backend = backend;
    } else if (model_type == 2) {
        if (!t2s) t2s = std::make_unique<T2SModel>();
        ok = t2s->load(slot.path, backend);
        t2s_backend = backend;
    } else if (model_type == 3) {
        if (!vits) vits = VITSModel::create(slot.path);
        if (vits) {
            ok = vits->load(slot.path, backend);
        } else {
            ok = false;
        }
        vits_target_backend = backend;
        if (ok && !validate_t2s_vits_compatibility()) {
            ok = false;
            vits.reset();
            vits_target_backend = nullptr;
        }
        if (ok && frontend) {
            struct ggml_tensor* text_emb_w = vits->semantic.text_embedding.weight.local_tensor();
            const int frontend_ver = vits->profile.symbol_version;
            frontend->set_symbol_version(frontend_ver);
            vits_version = vits->profile.vits_version;
            if (g_log_enabled) {
                std::cout << "[GPT-SoVITS load_model] VITS profile: " << vits->profile.summary()
                          << " (vocabulary size: " << (text_emb_w ? text_emb_w->ne[1] : 0)
                          << ", frontend symbols=v" << frontend_ver << ")\n";
            }
        }
    }

    if (ok) {
        if (model_type == 2 && !validate_t2s_vits_compatibility()) {
            ok = false;
            t2s.reset();
            t2s_backend = nullptr;
        }
    }

    if (ok) {
        slot.is_loaded = true;
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
    const char* device_name,
    const tts::RuntimeContext* runtime_context,
    std::shared_ptr<SharedStaticArtifacts> shared_artifacts
) {
    ggml_ops_ext::acquire_ops_hook();
    configure_sycl_cache_impl();

    if (dict_dir) params.dict_dir = dict_dir;
    if (hubert_model_path) params.hubert_model_path = hubert_model_path;
    if (bert_model_path) params.bert_model_path = bert_model_path;
    if (t2s_model_path) params.t2s_model_path = t2s_model_path;
    if (vits_model_path) params.vits_model_path = vits_model_path;
    params.n_threads = std::max(1, n_threads);
    params.use_gpu = (backend_mode > 0);
    shared_static_artifacts = std::move(shared_artifacts);
    if (const char* seed = std::getenv("T2S_RANDOM_SEED")) {
        default_rng.seed(static_cast<uint32_t>(std::strtoul(seed, nullptr, 10)));
    }

    frontend = std::make_unique<GPTSoVITSFrontend>(params.dict_dir);
    if (!frontend->initialize()) {
        std::cerr << "[GPT-SoVITS] Failed to initialize text frontend." << std::endl;
        return;
    }

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
    slots[0].is_resident = true;
    slots[1].is_resident = true;
    slots[2].is_resident = true;
    slots[3].is_resident = true;

    if (runtime_context) {
        static constexpr const char* component_names[4] = {"hubert", "bert", "t2s", "vits"};
        for (int i = 0; i < 4; ++i) {
            slots[i].device = runtime_context->device_name;
            const auto policy = runtime_context->components.find(component_names[i]);
            if (policy == runtime_context->components.end()) continue;
            if (!policy->second.device.empty()) slots[i].device = policy->second.device;
            slots[i].is_resident = policy->second.residency == tts::ComponentResidency::resident;
        }
    }

    initialized = true;
    for (int i = 0; i < 4; ++i) {
        if (slots[i].path.empty() || (slots[i].is_resident && !load_model(i))) {
            initialized = false;
            break;
        }
    }
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
    ggml_ops_ext::release_ops_hook();
}

} // namespace gpt_sovits

using namespace gpt_sovits;

void gpt_sovits_get_or_create_prompt_cache(
    gpt_sovits_engine_t engine,
    const char* cache_id,
    const float* ref_audio_data,
    size_t ref_audio_len,
    int ref_audio_sample_rate,
    const char* ref_text,
    const char* ref_language,
    const float* sv_emb_data,
    size_t sv_emb_len
) {
    gpt_sovits::CoutSilencer silencer(!gpt_sovits::g_log_enabled);

    if (!engine || !cache_id || !ref_audio_data || ref_audio_len == 0 ||
        ref_audio_sample_rate <= 0 || !ref_text || !ref_language) return;

    Impl* impl = (Impl*)engine;

    const float* source_ref_audio_data = ref_audio_data;
    const size_t source_ref_audio_len = ref_audio_len;
    const int source_ref_audio_sample_rate = ref_audio_sample_rate;

    std::vector<float> ref_audio_16k;
    if (ref_audio_sample_rate == 16000) {
        ref_audio_16k.assign(ref_audio_data, ref_audio_data + ref_audio_len);
    } else {
        ref_audio_16k = dsp::resample_audio(ref_audio_data, ref_audio_len, ref_audio_sample_rate, 16000);
    }
    ref_audio_data = ref_audio_16k.data();
    ref_audio_len = ref_audio_16k.size();

    std::string cid(cache_id);

    auto it = impl->prompt_caches.find(cid);

    if (it != impl->prompt_caches.end()) {
        return;
    }

    if (!impl->load_model(0) || !impl->load_model(1) || !impl->load_model(3)) {
        std::cerr << "[GPT-SoVITS] Error: Failed to load models required for prompt extraction.\n";
        return;
    }
    const int model_ref_rate = impl->vits->profile.reference_sampling_rate;
    std::vector<float> ref_audio_model;
    if (source_ref_audio_sample_rate == model_ref_rate) {
        ref_audio_model.assign(source_ref_audio_data, source_ref_audio_data + source_ref_audio_len);
    } else {
        ref_audio_model = dsp::resample_audio(
            source_ref_audio_data, source_ref_audio_len, source_ref_audio_sample_rate, model_ref_rate);
    }
    if (impl->vits && !impl->vits_galloc) {
        impl->vits_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));
    }

    bool is_sycl = false;
    if (impl->vits_target_backend) {
        const char* bname = ggml_backend_name(impl->vits_target_backend);
        if (bname && std::string(bname).find("SYCL") != std::string::npos) {
            is_sycl = true;
        }
    }
    ggml_gallocr_t active_galloc = is_sycl ? nullptr : impl->vits_galloc;

    {
        PromptCache cache;
        cache.prompt_text = ref_text;
        cache.prompt_lang = ref_language;
        if (sv_emb_data && sv_emb_len > 0) {
            cache.sv_emb.assign(sv_emb_data, sv_emb_data + sv_emb_len);
        }
        if (impl->vits) {
            cache.vits_version = impl->vits->profile.vits_version;
            cache.ge_dim = impl->vits->profile.ge_dim;
            cache.model_profile_id = impl->vits->profile.profile_id;
            cache.model_version = impl->vits->profile.exact_version;
            cache.requires_sv_emb = impl->vits->profile.requires_sv_emb;
            cache.sv_emb_dim = impl->vits->profile.sv_emb_dim;
            cache.ref_enc_channels = impl->vits->profile.ref_enc_channels;
            cache.output_sampling_rate = impl->vits->profile.output_sampling_rate;
            if (cache.requires_sv_emb && cache.sv_emb.size() != static_cast<size_t>(cache.sv_emb_dim)) {
                std::cerr << "[GPT-SoVITS] Warning: model profile " << cache.model_version
                          << " requires a " << cache.sv_emb_dim
                          << "-float sv_emb, but prompt cache was created with "
                          << cache.sv_emb.size() << " floats. Output will not match Python V2Pro conditioning." << std::endl;
            }
        }
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
        impl->frontend->process(cache.prompt_text, cache.prompt_lang, impl->bert.get(), ctx_graph, impl->bert_backend, front_res, active_galloc);
        cache.prompt_phones = front_res.phones;
        cache.prompt_word2ph = front_res.word2ph;
        cache.bert_features = front_res.bert_features;

        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 1 finished. Phonemes count: " << front_res.phones.size() << std::endl;

        if (g_log_enabled) {
            std::cout << "[GPT-SoVITS] Prompt Phones: ";
            for (const auto& ph : front_res.phones) std::cout << "'" << ph << "' ";
            std::cout << "\n";
        }

        // 2. CNHuBERT codes
        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 2: Running CNHuBERT..." << std::endl;

        // Convert ref_audio to ggml_tensor with zero padding at the end matching PyTorch's padding (4800 samples)
        size_t pad_samples = 9600; // 0.6s padding to match PyTorch's appending of 32kHz zero_wav to 16kHz audio
        size_t total_samples = ref_audio_len + pad_samples;

        std::vector<float> padded_audio(total_samples, 0.0f);
        for (size_t i = 0; i < ref_audio_len; ++i) {
            padded_audio[i] = ref_audio_data[i];
        }

        // Python calls cnhuhbert_model.model(...) directly here, bypassing the
        // Wav2Vec2FeatureExtractor normalization performed by CNHubert.forward().
        HubertRunner hubert_runner(*impl->hubert, impl->vits_backend);
        struct ggml_tensor* ssl_content = hubert_runner.forward(
            ctx_graph, padded_audio.data(), static_cast<int>(total_samples));
        if (!ssl_content) {
            std::cerr << "[GPT-SoVITS] Error: HuBERT inference failed." << std::endl;
            ggml_free(ctx_graph);
            return;
        }
        // Project and quantize using SoVITS VITS quantizer to get hubert_codes
        int n_frames = ssl_content->ne[1]; // seq_len
        cache.hubert_codes.resize(n_frames);

        struct ggml_tensor* cb = impl->vits->quantizer.codebook.weight.local_tensor();
        const nn::Parameter& ssl_projection = impl->vits->prompt_ssl_projection.weight;
        struct ggml_tensor* ssl_proj_w = ssl_projection.local_tensor();
        struct ggml_tensor* ssl_proj_b = impl->vits->prompt_ssl_projection.bias.local_tensor();

        if (!cb || !ssl_proj_w || !ssl_proj_b) {
            std::cerr << "[GPT-SoVITS] Error: VITS quantizer tensors are missing." << std::endl;
            ggml_free(ctx_graph);
            return;
        } else {
            const int code_dim = (int)cb->ne[0];
            const int codebook_size = (int)cb->ne[1];
            const int ssl_dim = (int)ssl_content->ne[0];
            const int kernel_size = static_cast<int>(ssl_projection.logical_shape().at(0));
            const bool channel_rows = ggml_is_quantized(ssl_proj_w->type);
            const int stride = impl->vits->profile.semantic_frame_stride;
            const int out_frames = (n_frames - kernel_size) / stride + 1;

            if (impl->prompt_vq_profile_id != impl->vits->profile.profile_id) {
                impl->prompt_vq_codebook = get_tensor_as_float(cb);
                impl->prompt_vq_projection_weights = get_tensor_as_float(ssl_proj_w);
                impl->prompt_vq_projection_bias = get_tensor_as_float(ssl_proj_b);
                impl->prompt_vq_codebook_norms.assign(codebook_size, 0.0f);
                for (int k = 0; k < codebook_size; ++k) {
                    const float* code = impl->prompt_vq_codebook.data() + k * code_dim;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < code_dim; ++i) {
                        sum_sq += code[i] * code[i];
                    }
                    impl->prompt_vq_codebook_norms[k] = sum_sq;
                }
                impl->prompt_vq_profile_id = impl->vits->profile.profile_id;
            }
            const std::vector<float>& cb_data = impl->prompt_vq_codebook;
            const std::vector<float>& cb_norms = impl->prompt_vq_codebook_norms;

            // Get Hubert features back to CPU to set to the graph input
            std::vector<float> ssl_content_cpu(ssl_dim * n_frames);
            safe_ggml_backend_tensor_get(ssl_content, ssl_content_cpu.data(), 0, ssl_dim * n_frames * sizeof(float));
            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::ofstream f("scratch/ssl_content.bin", std::ios::binary);
                if (f.is_open()) {
                    f.write((char*)ssl_content_cpu.data(), ssl_content_cpu.size() * sizeof(float));
                }
            }

            std::vector<float> projected_data(code_dim * out_frames);
            const std::vector<float>& projection_weights = impl->prompt_vq_projection_weights;
            const std::vector<float>& projection_bias = impl->prompt_vq_projection_bias;

            // Prompt extraction is cached, so prefer a deterministic CPU accumulation
            // order over backend-dependent GEMM rounding at VQ decision boundaries.
            parallel_for_ranges(out_frames, impl->params.n_threads, [&](int begin, int end) {
                for (int t = begin; t < end; ++t) {
                    float* output = projected_data.data() + t * code_dim;
                    for (int oc = 0; oc < code_dim; ++oc) {
                        float acc = projection_bias[oc];
                        for (int kernel = 0; kernel < kernel_size; ++kernel) {
                            const float* input = ssl_content_cpu.data() + (stride * t + kernel) * ssl_dim;
                            for (int ic = 0; ic < ssl_dim; ++ic) {
                                const size_t weight_index = channel_rows
                                    ? static_cast<size_t>(ic) + ssl_dim * (kernel + kernel_size * oc)
                                    : static_cast<size_t>(kernel) + kernel_size * (ic + ssl_dim * oc);
                                acc += input[ic] * projection_weights[weight_index];
                            }
                        }
                        output[oc] = acc;
                    }
                }
            });

            // Compute L2 distances on CPU to find the argmin
            cache.hubert_codes.resize(out_frames);
            parallel_for_ranges(out_frames, impl->params.n_threads, [&](int begin, int end) {
                for (int t = begin; t < end; ++t) {
                    float min_dist = 1e30f;
                    int best_k = 0;
                    const float* frame = projected_data.data() + t * code_dim;
                    for (int k = 0; k < codebook_size; ++k) {
                        const float* code = cb_data.data() + k * code_dim;
                        float dot_prod = 0.0f;
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
            });

            if (GPT_SOVITS_DEBUG_ENABLED()) {
                std::cout << "[GPT-SoVITS Debug] Extracted prompt semantic len: " << cache.hubert_codes.size() << "\n";
                std::cout << "[GPT-SoVITS Debug] Extracted prompt semantic first 20: ";
                for (size_t i = 0; i < std::min((size_t)20, cache.hubert_codes.size()); ++i) {
                    std::cout << cache.hubert_codes[i] << " ";
                }
                std::cout << "\n";
            }
        }
        // 3. BERT Features already computed in Step 1

        // 4. Compute speaker embedding (ge) via ref_enc from reference audio STFT spectrogram
        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: Computing speaker embedding (ge) via ref_enc..." << std::endl;
        {
            int n_frames = 0;
            int n_ref_enc = impl->vits ? impl->vits->profile.ref_enc_channels : 704;
            std::vector<float> ref_enc_input = dsp::compute_stft_spectrogram(
                ref_audio_model.data(), ref_audio_model.size(), n_ref_enc, n_frames);
            if (n_frames > 0) {
                struct ggml_init_params ge_init_params = {
                    /* .mem_size   = */ 128 * 1024 * 1024,
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true
                };

                struct ggml_context* ctx_ge = ggml_init(ge_init_params);
                nn::Context ge_context = nn::Context::borrow(ctx_ge);
                struct ggml_tensor* mel_spec_tensor = ge_context.input<float>(
                    "vits.reference_spectrogram", {n_ref_enc, n_frames}, nn::data::borrow(ref_enc_input));
                struct ggml_tensor* sv_emb_tensor = nullptr;
                if (sv_emb_data && sv_emb_len > 0) {
                    sv_emb_tensor = ge_context.input<float>(
                        "vits.speaker_vector", {sv_emb_len, 1}, nn::data::borrow(sv_emb_data, sv_emb_len));
                }

                ggml_backend_buffer_t ge_input_buf = ggml_backend_alloc_ctx_tensors(ctx_ge, impl->vits_target_backend);

                if (ge_input_buf) {
                    ge_context.materialize();
                }
                // Create graph context for ref_enc compute

                struct ggml_init_params ge_graph_params = {
                    /* .mem_size   = */ 256 * 1024 * 1024,
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true
                };

                struct ggml_context* ctx_ge_graph = ggml_init(ge_graph_params);
                nn::Context ge_graph_context = nn::Context::borrow(ctx_ge_graph);

                struct ggml_tensor* ge_tensor = impl->vits->compute_speaker_embedding(
                    ge_graph_context, mel_spec_tensor, sv_emb_tensor, impl->vits_target_backend);

                if (ge_tensor) {
                    struct ggml_cgraph* ge_graph = ggml_new_graph_custom(ctx_ge_graph, 65536, false);

                    ggml_build_forward_expand(ge_graph, ge_tensor);

                    ggml_backend_buffer_t ge_buf = ggml_backend_alloc_ctx_tensors(ctx_ge_graph, impl->vits_target_backend);

                    if (ge_buf) {
                        ggml_ops_ext::ops_backend_graph_compute(impl->vits_target_backend, ge_graph);
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
                if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: Warning: ref audio too short for STFT (" << ref_audio_model.size() << " samples). Using zero ge." << std::endl;
                int current_ge_dim = impl->vits->profile.ge_dim;
                cache.speaker_embedding.assign(current_ge_dim, 0.0f);
            }
        }

        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: ge computation complete." << std::endl;

        // Compute prompt Mel spectrogram for CFM models (v3/v4)
        if (impl->vits->profile.uses_cfm) {
            int out_frames = 0;
            int sampling_rate = impl->vits->profile.prompt_mel_sampling_rate;
            int n_fft = impl->vits->profile.filter_length;
            int hop_size = impl->vits->profile.hop_length;
            int win_size = impl->vits->profile.win_length;
            int mel_channels = impl->vits->profile.prompt_mel_channels > 0 ? impl->vits->profile.prompt_mel_channels : 100;
            std::vector<float> prompt_audio;
            if (source_ref_audio_sample_rate == sampling_rate) {
                prompt_audio.assign(source_ref_audio_data, source_ref_audio_data + source_ref_audio_len);
            } else {
                prompt_audio = dsp::resample_audio(
                    source_ref_audio_data, source_ref_audio_len, source_ref_audio_sample_rate, sampling_rate);
            }
            cache.prompt_mel = dsp::compute_mel_spectrogram(
                prompt_audio.data(),
                prompt_audio.size(),
                sampling_rate,
                n_fft,
                hop_size,
                win_size,
                mel_channels,
                out_frames
            );
            // Python applies norm_spec before using the reference mel as CFM prompt.
            for (float& value : cache.prompt_mel) {
                value = (value + 12.0f) / 7.0f - 1.0f;
            }
            if (g_log_enabled) {
                std::cout << "[GPT-SoVITS] Computed prompt Mel spectrogram. Frames: " << out_frames
                          << ", size: " << cache.prompt_mel.size() << std::endl;
            }
        }
        // 5. If FlowMatching (V3/V4), run VITS encoder on prompt text to get prompt_fea_ref
        if (impl->vits && impl->vits->profile.uses_cfm) {
            if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 5: Pre-computing prompt VITS encoder features..." << std::endl;
            
            // Map prompt phonemes to IDs
            std::vector<int32_t> prompt_phone_ids;
            for (const auto& ph : cache.prompt_phones) {
                prompt_phone_ids.push_back(impl->frontend->phone_to_id(ph));
            }
            
            int current_ge_dim = impl->vits->profile.ge_dim;
            std::vector<float> prompt_ge(current_ge_dim, 0.0f);
            std::copy_n(cache.speaker_embedding.data(),
                        std::min(cache.speaker_embedding.size(), prompt_ge.size()), prompt_ge.data());
            
            // 5.2 Build graph
            nn::Context prompt_vits_context(256 * 1024 * 1024);
            struct ggml_context* ctx_prompt_vits = prompt_vits_context.native_handle();
            struct ggml_tensor* prompt_phone_tensor = prompt_vits_context.input<int32_t>(
                "prompt_phone_ids", {static_cast<int64_t>(prompt_phone_ids.size())}, nn::data::borrow(prompt_phone_ids));
            struct ggml_tensor* prompt_semantics_tensor = prompt_vits_context.input<int32_t>(
                "prompt_semantics", {static_cast<int64_t>(cache.hubert_codes.size())}, nn::data::borrow(cache.hubert_codes));
            struct ggml_tensor* ge_tensor = prompt_vits_context.input<float>(
                "prompt_speaker_embedding", {current_ge_dim, 1}, nn::data::borrow(prompt_ge));
            
            VITSModel::EncodeResult enc_res = impl->vits->encode_semantic_base(
                prompt_vits_context,
                prompt_phone_tensor,
                prompt_semantics_tensor,
                ge_tensor,
                impl->vits_target_backend
            );
            
            if (enc_res.y2) {
                VITSModelCFM* cfm_model = dynamic_cast<VITSModelCFM*>(impl->vits.get());
                if (cfm_model) {
                    struct ggml_tensor* cond_text = cfm_model->condition_features(
                        prompt_vits_context, enc_res.y2, enc_res.ge,
                        impl->vits_target_backend);
                    if (!cond_text) {
                        ggml_free(ctx_prompt_vits);
                        return;
                    }
                    
                    struct ggml_cgraph* prompt_graph = ggml_new_graph_custom(ctx_prompt_vits, 65536, false);
                    ggml_build_forward_expand(prompt_graph, cond_text);
                    
                    ggml_gallocr_t prompt_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));
                    if (prompt_galloc) {
                        ggml_gallocr_alloc_graph(prompt_galloc, prompt_graph);
                        prompt_vits_context.materialize();
                        ggml_ops_ext::ops_backend_graph_compute(impl->vits_target_backend, prompt_graph);
                        
                        int64_t prompt_fea_nelems = ggml_nelements(cond_text);
                        cache.prompt_fea_ref.resize(prompt_fea_nelems);
                        ggml_backend_tensor_get(cond_text, cache.prompt_fea_ref.data(), 0, prompt_fea_nelems * sizeof(float));
                        
                        if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 5: Prompt encoder features computed! size=" << cache.prompt_fea_ref.size() << std::endl;
                        
                        ggml_gallocr_free(prompt_galloc);
                    }
                }
            }
        }
        // Clean up graph context

        ggml_free(ctx_graph);
        cache.vits_version = impl->vits->profile.vits_version;
        cache.ge_dim = impl->vits->profile.ge_dim;
        cache.model_profile_id = impl->vits->profile.profile_id;
        cache.model_version = impl->vits->profile.exact_version;
        cache.requires_sv_emb = impl->vits->profile.requires_sv_emb;
        cache.sv_emb_dim = impl->vits->profile.sv_emb_dim;
        cache.ref_enc_channels = impl->vits->profile.ref_enc_channels;
        cache.output_sampling_rate = impl->vits->profile.output_sampling_rate;
        cache.device_type = get_backend_device_type(impl->vits_target_backend);
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
    if (!impl->load_model(1) || !impl->load_model(2) || !impl->load_model(3)) {
        std::cerr << "[GPT-SoVITS] Error: Failed to load models required for synthesis.\n";
        return nullptr;
    }
    if (impl->vits && !impl->vits_galloc) {
        impl->vits_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl->vits_target_backend));
    }

    bool is_sycl = false;
    if (impl->vits_target_backend) {
        const char* bname = ggml_backend_name(impl->vits_target_backend);
        if (bname && std::string(bname).find("SYCL") != std::string::npos) {
            is_sycl = true;
        }
    }
    ggml_gallocr_t active_galloc = is_sycl ? nullptr : impl->vits_galloc;
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
        nn::Context graph_context = nn::Context::borrow(ctx_graph);
        bert_features_tensor = graph_context.empty<float>("bert.features", {1024, text_len});
        graph_context.write(bert_features_tensor, fused_bert_aligned.data(), fused_bert_aligned.size());
    } else {
        std::string lang_str(language);
        struct ggml_init_params init_params = {
            /* .mem_size   = */ 256 * 1024 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false
        };
        ctx_graph = ggml_init(init_params);

        std::vector<float> target_bert_aligned;
        FrontendResult front_res;
        int64_t t_front_start = ggml_time_us();
        impl->frontend->process(std::string(text), lang_str, impl->bert.get(), ctx_graph, impl->bert_backend, front_res, active_galloc);
        int64_t t_front_end = ggml_time_us();
        if (g_log_enabled) {
            std::cout << "[GPT-SoVITS] Frontend process (phonemes + BERT) took: " << (t_front_end - t_front_start) / 1000.0 << " ms" << std::endl;
        }

        target_res.phones = front_res.phones;
        target_res.word2ph = front_res.word2ph;
        target_bert_aligned = front_res.bert_features;
        target_phone_ids = front_res.phone_ids;

        if (g_log_enabled) {
            std::cout << "[GPT-SoVITS] Target Phones: ";
            for (const auto& ph : target_res.phones) std::cout << "'" << ph << "' ";
            std::cout << "\n";
        }

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

        nn::Context graph_context = nn::Context::borrow(ctx_graph);
        bert_features_tensor = graph_context.empty<float>("bert.features", {1024, text_len});
        graph_context.write(bert_features_tensor, fused_bert_aligned.data(), fused_bert_aligned.size());
        if (lang_str == "zh" || lang_str == "zh_en") {
            target_bert_out = bert_features_tensor; // set to non-null
        }
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
            impl->t2s_backend, // Run T2S on CPU/GPU depending on backend configuration
            impl->active_rng ? *impl->active_rng : impl->default_rng,
            active_galloc
        );

        int64_t t_t2s_end = ggml_time_us();
        if (g_log_enabled) {
            std::cout << "[GPT-SoVITS] T2S forward took: " << (t_t2s_end - t_t2s_start) / 1000.0 << " ms" << std::endl;
        }
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[GPT-SoVITS Debug] Generated tokens count: " << pred_semantics.size() << "\n";
            std::cout << "[GPT-SoVITS Debug] First 20 tokens: ";
            for (size_t i = 0; i < std::min((size_t)20, pred_semantics.size()); ++i) {
                std::cout << pred_semantics[i] << " ";
            }
            std::cout << "\n";
        }
    }
    if (pred_semantics.empty()) {
        std::cerr << "[GPT-SoVITS] Error: T2S produced no semantic tokens.\n";
        ggml_free(ctx_graph);
        impl->offload_model(1);
        impl->offload_model(2);
        impl->offload_model(3);
        return nullptr;
    }
    // 4. Run SoVITS VITS Decoder to synthesize audio
    // Create dedicated context for VITS execution with no_alloc = true to allow backend allocation.

    // We use a single context for both inputs and model graph to ensure broad backend compatibility (including SYCL/CUDA/CPU).

    nn::Context vits_context((size_t)1544 * 1024 * 1024);
    struct ggml_context* ctx_vits = vits_context.native_handle();
    int word2ph_size = is_overridden ? 1 : (int)target_res.word2ph.size();
    int bert_out_len = is_overridden ? target_len : (int)target_res.phones.size();
    
    std::vector<int32_t> dummy_word2ph(1, 1);
    const int32_t* word2ph_ptr = is_overridden ? dummy_word2ph.data() : target_res.word2ph.data();
    std::vector<float> zero_bert;
    const float* bert_data = nullptr;
    if (is_overridden || target_bert_out) {
        bert_data = fused_bert_aligned.data() + 1024 * prompt_len;
    } else {
        zero_bert.assign(1024 * bert_out_len, 0.0f);
        bert_data = zero_bert.data();
    }

    int current_ge_dim = impl->vits->profile.ge_dim;
    VITSRunState vits_run;
    const int ge_size = (int)cached_prompt.speaker_embedding.size();
    if (GPT_SOVITS_DEBUG_ENABLED()) {
        float ge_l1 = 0.0f;
        for (float val : cached_prompt.speaker_embedding) ge_l1 += std::abs(val);
        float mel_l1 = 0.0f;
        for (float val : cached_prompt.prompt_mel) mel_l1 += std::abs(val);
        float fea_l1 = 0.0f;
        for (float val : cached_prompt.prompt_fea_ref) fea_l1 += std::abs(val);
        std::cout << "[DEBUG Pipeline] Speaker embedding size=" << ge_size << ", L1=" << ge_l1 << std::endl;
        std::cout << "[DEBUG Pipeline] Prompt Mel size=" << cached_prompt.prompt_mel.size() << ", L1=" << mel_l1 << std::endl;
        std::cout << "[DEBUG Pipeline] Prompt fea_ref size=" << cached_prompt.prompt_fea_ref.size() << ", L1=" << fea_l1 << std::endl;
    }

    std::vector<float> ge_input(current_ge_dim, 0.0f);
    std::copy_n(cached_prompt.speaker_embedding.data(), std::min(ge_size, current_ge_dim), ge_input.data());

    if (impl->vits->profile.uses_cfm && cached_prompt.prompt_mel.size() > 0) {
        vits_run.prompt_mel = cached_prompt.prompt_mel;
        vits_run.prompt_features = cached_prompt.prompt_fea_ref;
    }

    struct ggml_tensor* target_phone_tensor = vits_context.input<int32_t>(
        "target_phone_ids", {static_cast<int64_t>(target_phone_ids.size())}, nn::data::borrow(target_phone_ids));
    struct ggml_tensor* target_word2ph_tensor = vits_context.input<int32_t>(
        "target_word2ph", {word2ph_size}, nn::data::borrow(word2ph_ptr, word2ph_size));
    struct ggml_tensor* pred_semantics_tensor = vits_context.input<int32_t>(
        "pred_semantics", {static_cast<int64_t>(pred_semantics.size())}, nn::data::borrow(pred_semantics));
    struct ggml_tensor* target_bert_out_gpu = vits_context.input<float>(
        "target_bert", {1024, bert_out_len}, nn::data::borrow(bert_data, 1024 * bert_out_len));
    struct ggml_tensor* ge_tensor = vits_context.input<float>(
        "speaker_embedding", {current_ge_dim, 1}, nn::data::borrow(ge_input));

    if (g_log_enabled) std::cout << "[GPT-SoVITS] Step 4: ge_size=" << ge_size << ", calling VITS forward with cached speaker embedding..." << std::endl;

    int64_t t_vits_start = ggml_time_us();

    struct ggml_tensor* synth_audio = impl->vits->forward(
        vits_context,
        target_phone_tensor,
        nullptr, // phone_lengths
        target_word2ph_tensor,
        target_bert_out_gpu,
        pred_semantics_tensor,
        ge_tensor, // pass speaker embedding (ge) as refer_audio
        vits_run,
        speed,
        impl->vits_target_backend
    );
    if (!synth_audio) {
        std::cerr << "[GPT-SoVITS] Error: VITS failed to build the synthesis graph.\n";
        ggml_free(ctx_graph);
        impl->offload_model(1);
        impl->offload_model(2);
        impl->offload_model(3);
        return nullptr;
    }
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

    vits_context.materialize();
    ggml_ops_ext::ops_backend_graph_compute(impl->vits_target_backend, gf);
    int64_t t_vits_end = ggml_time_us();
    if (g_log_enabled) {
        std::cout << "[GPT-SoVITS] VITS forward took: " << (t_vits_end - t_vits_start) / 1000.0 << " ms" << std::endl;
    }

    // Convert synthesized tensor to final PCM float array in the resident memory
    int out_samples = (int)ggml_nelements(synth_audio);
    impl->last_synthesized_audio.resize(out_samples);
    ggml_backend_tensor_get(synth_audio, impl->last_synthesized_audio.data(), 0, out_samples * sizeof(float));

    // Cleanup VITS contexts and buffers (impl->vits_galloc is persistent, so DO NOT free it here)

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
    const std::string split_method = select_split_method();

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
    int ref_audio_sample_rate,
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

    gpt_sovits_get_or_create_prompt_cache(engine, "temp_prompt_cache", ref_audio_data, ref_audio_len,
                                          ref_audio_sample_rate, ref_text, ref_language, nullptr, 0);

    const float* res = gpt_sovits_synthesize_with_cache(engine, text, language, "temp_prompt_cache", speed, out_num_samples);
    impl->bypass_offload = false;
    impl->offload_model(0);
    impl->offload_model(1);
    impl->offload_model(2);
    impl->offload_model(3);
    return res;
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
        ggml_backend_load_all();
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

    const bool batch_swap =
        t2s_model_path && t2s_model_path[0] != '\0' &&
        vits_model_path && vits_model_path[0] != '\0';
    impl->defer_model_compat_validation = batch_swap;

    if (t2s_model_path) {
        impl->slots[2].path = t2s_model_path;
        if (impl->slots[2].is_loaded || impl->slots[2].is_resident) {
            impl->slots[2].is_loaded = false; // force reload
            if (!impl->load_model(2)) {
                impl->defer_model_compat_validation = false;
                if (batch_swap) {
                    impl->t2s.reset();
                    impl->slots[2].is_loaded = false;
                    impl->t2s_backend = nullptr;
                }
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
                impl->defer_model_compat_validation = false;
                if (batch_swap) {
                    impl->t2s.reset();
                    impl->vits.reset();
                    impl->slots[2].is_loaded = false;
                    impl->slots[3].is_loaded = false;
                    impl->t2s_backend = nullptr;
                    impl->vits_target_backend = nullptr;
                }
                std::cerr << "[gpt_sovits_load_speaker] Failed to load new VITS model\n";
                return false;
            }
        }
    }

    impl->defer_model_compat_validation = false;
    if (!impl->validate_t2s_vits_compatibility()) {
        if (batch_swap) {
            impl->t2s.reset();
            impl->vits.reset();
            impl->slots[2].is_loaded = false;
            impl->slots[3].is_loaded = false;
            impl->t2s_backend = nullptr;
            impl->vits_target_backend = nullptr;
        }
        std::cerr << "[gpt_sovits_load_speaker] New T2S/VITS models are not compatible\n";
        return false;
    }

    if (g_log_enabled) {
        std::cout << "[gpt_sovits_load_speaker] Swapped successfully!\n";
    }
    return true;
}

void gpt_sovits_set_cfm_steps(gpt_sovits_engine_t engine, int steps) {
    if (engine) {
        Impl* impl = (Impl*)engine;
        if (impl->vits) {
            impl->vits->cfm_steps = steps;
            if (g_log_enabled) {
                std::cout << "[gpt_sovits_set_cfm_steps] Set CFM ODE steps to: " << steps << "\n";
            }
        }
    }
}

int gpt_sovits_get_version(gpt_sovits_engine_t engine) {
    if (engine) {
        Impl* impl = (Impl*)engine;
        if (impl->vits) {
            return impl->vits->profile.vits_version;
        }
        return impl->vits_version;
    }
    return 0;
}

int gpt_sovits_get_sampling_rate(gpt_sovits_engine_t engine) {
    if (engine) {
        Impl* impl = (Impl*)engine;
        if (impl->vits) {
            return impl->vits->profile.output_sampling_rate;
        }
        return 0;
    }
    return 0;
}

} // extern "C"

namespace tts {

class GPTSoVITSModel;

class GPTSoVITSSession final : public ITTSSession {
public:
    GPTSoVITSSession(GPTSoVITSModel& model, std::string cache_id)
        : model_(model), cache_id_(std::move(cache_id)) {
        if (const char* seed = std::getenv("T2S_RANDOM_SEED")) {
            rng_.seed(static_cast<uint32_t>(std::strtoul(seed, nullptr, 10)));
        }
    }

    std::vector<float> synthesize(const SynthesisRequest& request) override;
    bool synthesize_streaming(const SynthesisRequest& request, AudioChunkCallback callback) override;
    bool set_reference(const VoiceReference& reference) override;
    int32_t output_sample_rate() const override { return output_sample_rate_; }

private:
    friend class GPTSoVITSModel;

    GPTSoVITSModel& model_;
    std::string cache_id_;
    std::optional<gpt_sovits::PromptCache> prompt_cache_;
    size_t voice_signature_ = 0;
    bool has_voice_signature_ = false;
    std::mt19937 rng_{42u};
    VoiceReference reference_;
    int32_t output_sample_rate_ = 0;
};

class GPTSoVITSModel final : public ITTSModel {
private:
    friend class GPTSoVITSSession;

    struct ExecutionLane {
        std::unique_ptr<gpt_sovits::Impl> impl;
        bool busy = false;
    };

    class LaneLease {
    public:
        LaneLease() = default;
        LaneLease(GPTSoVITSModel* owner, ExecutionLane* lane) : owner_(owner), lane_(lane) {}
        LaneLease(const LaneLease&) = delete;
        LaneLease& operator=(const LaneLease&) = delete;
        LaneLease(LaneLease&& other) noexcept : owner_(other.owner_), lane_(other.lane_) {
            other.owner_ = nullptr;
            other.lane_ = nullptr;
        }
        ~LaneLease() { if (owner_ && lane_) owner_->release_lane(lane_); }

        explicit operator bool() const { return lane_ && lane_->impl; }
        gpt_sovits::Impl& impl() const { return *lane_->impl; }

    private:
        GPTSoVITSModel* owner_ = nullptr;
        ExecutionLane* lane_ = nullptr;
    };

    std::vector<std::unique_ptr<ExecutionLane>> lanes_;
    std::shared_ptr<gpt_sovits::SharedStaticArtifacts> shared_static_artifacts_ =
        std::make_shared<gpt_sovits::SharedStaticArtifacts>();
    std::mutex lanes_mutex_;
    std::condition_variable lanes_cv_;
    RuntimeContext runtime_;
    std::string dict_dir_;
    std::string hubert_path_;
    std::string bert_path_;
    std::string t2s_path_;
    std::string vits_path_;
    std::atomic<uint64_t> next_session_id_{1};

    std::unique_ptr<gpt_sovits::Impl> create_impl() const {
        auto result = std::make_unique<gpt_sovits::Impl>(
            dict_dir_.c_str(), hubert_path_.c_str(), bert_path_.c_str(),
            t2s_path_.c_str(), vits_path_.c_str(),
            static_cast<int>(runtime_.n_threads), 0, runtime_.device_name.c_str(), &runtime_,
            shared_static_artifacts_);
        if (!result->initialized) return nullptr;
        return result;
    }

    LaneLease acquire_lane() {
        std::unique_lock<std::mutex> lock(lanes_mutex_);
        for (;;) {
            for (const auto& lane : lanes_) {
                if (!lane->busy && lane->impl) {
                    lane->busy = true;
                    return LaneLease(this, lane.get());
                }
            }

            if (lanes_.size() < runtime_.max_concurrency) {
                auto lane = std::make_unique<ExecutionLane>();
                lane->busy = true;
                ExecutionLane* lane_ptr = lane.get();
                lanes_.push_back(std::move(lane));
                lock.unlock();
                auto impl = create_impl();
                lock.lock();
                if (!impl) {
                    lanes_.erase(std::remove_if(lanes_.begin(), lanes_.end(),
                        [lane_ptr](const auto& item) { return item.get() == lane_ptr; }), lanes_.end());
                    lanes_cv_.notify_all();
                    return {};
                }
                lane_ptr->impl = std::move(impl);
                lock.unlock();
                return LaneLease(this, lane_ptr);
            }
            lanes_cv_.wait(lock);
        }
    }

    void release_lane(ExecutionLane* lane) {
        std::lock_guard<std::mutex> lock(lanes_mutex_);
        lane->busy = false;
        lanes_cv_.notify_one();
    }

    class PromptCacheAttachment {
    public:
        PromptCacheAttachment(gpt_sovits::Impl& impl, GPTSoVITSSession& session)
            : impl_(impl), session_(session) {
            if (session_.prompt_cache_) {
                impl_.prompt_caches.emplace(session_.cache_id_, std::move(*session_.prompt_cache_));
                session_.prompt_cache_.reset();
            }
        }

        ~PromptCacheAttachment() {
            auto node = impl_.prompt_caches.extract(session_.cache_id_);
            if (!node.empty()) session_.prompt_cache_ = std::move(node.mapped());
        }

    private:
        gpt_sovits::Impl& impl_;
        GPTSoVITSSession& session_;
    };

    class RngAttachment {
    public:
        RngAttachment(gpt_sovits::Impl& impl, std::mt19937& rng)
            : impl_(impl), previous_(impl.active_rng) {
            impl_.active_rng = &rng;
        }
        ~RngAttachment() { impl_.active_rng = previous_; }

    private:
        gpt_sovits::Impl& impl_;
        std::mt19937* previous_;
    };

    class ModelResidencyScope {
    public:
        ModelResidencyScope(gpt_sovits::Impl& impl, std::initializer_list<int> model_types)
            : impl_(impl), previous_bypass_(impl.bypass_offload), model_types_(model_types) {
            impl_.bypass_offload = true;
            for (int model_type : model_types_) {
                if (!impl_.load_model(model_type)) {
                    ready_ = false;
                    break;
                }
            }
        }

        ~ModelResidencyScope() {
            impl_.bypass_offload = previous_bypass_;
            if (!previous_bypass_) {
                for (auto it = model_types_.rbegin(); it != model_types_.rend(); ++it) {
                    impl_.offload_model(*it);
                }
            }
        }

        explicit operator bool() const { return ready_; }

    private:
        gpt_sovits::Impl& impl_;
        bool previous_bypass_;
        std::vector<int> model_types_;
        bool ready_ = true;
    };

    class CFMStepsScope {
    public:
        CFMStepsScope(gpt_sovits::Impl& impl, const std::unordered_map<std::string, float>& options)
            : impl_(impl) {
            if (!impl_.vits) return;
            previous_ = impl_.vits->cfm_steps;
            const auto steps = options.find("cfm_steps");
            if (steps != options.end() && steps->second > 0.0f) {
                impl_.vits->cfm_steps = static_cast<int>(steps->second);
                active_ = true;
            }
        }

        ~CFMStepsScope() {
            if (active_ && impl_.vits) impl_.vits->cfm_steps = previous_;
        }

    private:
        gpt_sovits::Impl& impl_;
        int previous_ = 0;
        bool active_ = false;
    };

    static size_t voice_signature(const SynthesisRequest& request) {
        size_t hash = 1469598103934665603ull;
        auto append = [&](const void* data, size_t size) {
            const auto* bytes = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < size; ++i) {
                hash ^= bytes[i];
                hash *= 1099511628211ull;
            }
        };
        if (!request.ref_audio.empty()) {
            append(request.ref_audio.data(), request.ref_audio.size() * sizeof(float));
        }
        const auto text = request.string_params.find("ref_text");
        if (text != request.string_params.end()) append(text->second.data(), text->second.size());
        const auto lang = request.string_params.find("ref_language");
        if (lang != request.string_params.end()) append(lang->second.data(), lang->second.size());
        return hash;
    }

    void prepare_voice(GPTSoVITSSession& session, const SynthesisRequest& request) {
        if (request.ref_audio.empty()) return;
        const size_t signature = voice_signature(request);
        if (session.has_voice_signature_ && session.voice_signature_ != signature) {
            session.prompt_cache_.reset();
        }
        session.voice_signature_ = signature;
        session.has_voice_signature_ = true;
    }

    bool set_reference_session(GPTSoVITSSession& session, const VoiceReference& reference) {
        if (reference.audio.empty() || reference.sample_rate <= 0) return false;
        session.reference_ = reference;
        session.prompt_cache_.reset();
        session.has_voice_signature_ = false;
        return true;
    }

public:
    GPTSoVITSModel() = default;
    ~GPTSoVITSModel() override {
        while (lanes_.size() > 1) lanes_.pop_back();
        shared_static_artifacts_.reset();
        lanes_.clear();
    }

    bool initialize(const ModelConfig& config, const RuntimeContext& runtime) {
        std::string dict_dir;
        std::string hubert;
        std::string bert;
        std::string t2s;
        std::string vits;

        if (!config.adapters.empty()) {
            std::cerr << "[GPT-SoVITS Provider] Adapters are declared but not supported yet." << std::endl;
            return false;
        }

        auto resolve_required = [&](const char* key, std::string& output) {
            const auto it = config.models.find(key);
            if (it == config.models.end()) {
                std::cerr << "[GPT-SoVITS Provider] Missing required model entry: " << key << std::endl;
                return false;
            }
            const std::filesystem::path resolved = config.resolve_path(it->second);
            if (!std::filesystem::exists(resolved)) {
                std::cerr << "[GPT-SoVITS Provider] Model entry '" << key
                          << "' does not exist: " << resolved.string() << std::endl;
                return false;
            }
            output = resolved.u8string();
            return true;
        };

        if (!resolve_required("dict", dict_dir) ||
            !resolve_required("hubert", hubert) ||
            !resolve_required("bert", bert) ||
            !resolve_required("t2s", t2s) ||
            !resolve_required("vits", vits)) {
            return false;
        }

        runtime_ = runtime;
        dict_dir_ = std::move(dict_dir);
        hubert_path_ = std::move(hubert);
        bert_path_ = std::move(bert);
        t2s_path_ = std::move(t2s);
        vits_path_ = std::move(vits);

        auto lane = std::make_unique<ExecutionLane>();
        lane->impl = create_impl();
        if (!lane->impl) return false;
        lanes_.push_back(std::move(lane));
        return true;
    }

    std::unique_ptr<ITTSSession> create_session() override {
        if (lanes_.empty()) return nullptr;
        const uint64_t id = next_session_id_.fetch_add(1, std::memory_order_relaxed);
        return std::make_unique<GPTSoVITSSession>(*this, "tts-session-" + std::to_string(id));
    }

    const TTSCapabilities& capabilities() const override {
        static const TTSCapabilities value = {
            /* streaming = */ true,
            /* voice_cloning = */ true,
            /* speaker_id = */ false,
            /* speaker_embedding = */ false,
            /* emotion = */ false,
            /* deterministic_seed = */ false,
            /* speed_control = */ true,
        };
        return value;
    }

    std::vector<float> synthesize_session(GPTSoVITSSession& session, const SynthesisRequest& request) {
        LaneLease lane = acquire_lane();
        if (!lane) return {};
        auto& impl = lane.impl();
        prepare_voice(session, request);
        PromptCacheAttachment cache_attachment(impl, session);
        RngAttachment rng_attachment(impl, session.rng_);

        const bool request_has_reference = !request.ref_audio.empty();
        const std::vector<float>& ref_audio = request_has_reference ? request.ref_audio : session.reference_.audio;
        const float* ref_audio_data = ref_audio.data();
        size_t ref_audio_len = ref_audio.size();
        const int ref_audio_sample_rate = request_has_reference ? 16000 : session.reference_.sample_rate;

        std::string ref_text = "";
        auto it_text = request.string_params.find("ref_text");
        if (it_text != request.string_params.end()) {
            ref_text = it_text->second;
        } else if (!request_has_reference) {
            ref_text = session.reference_.text;
        }

        std::string ref_lang = "zh";
        auto it_lang = request.string_params.find("ref_language");
        if (it_lang != request.string_params.end()) {
            ref_lang = it_lang->second;
        } else if (!request_has_reference && !session.reference_.language.empty()) {
            ref_lang = session.reference_.language;
        }

        float speed = 1.0f;
        auto it_speed = request.float_params.find("speed");
        if (it_speed != request.float_params.end()) {
            speed = it_speed->second;
        }

        int out_samples = 0;
        ModelResidencyScope residency(impl, {0, 1, 2, 3});
        if (!residency) return {};

        if (impl.vits) {
            session.output_sample_rate_ = impl.vits->profile.output_sampling_rate;
        }
        CFMStepsScope cfm_steps(impl, request.float_params);

        gpt_sovits_get_or_create_prompt_cache(&impl, session.cache_id_.c_str(), ref_audio_data, ref_audio_len, ref_audio_sample_rate, ref_text.c_str(), ref_lang.c_str(), nullptr, 0);

        const float* res = gpt_sovits_synthesize_with_cache(&impl, request.text.c_str(), request.language.c_str(), session.cache_id_.c_str(), speed, &out_samples);

        if (res && out_samples > 0) {
            return std::vector<float>(res, res + out_samples);
        }
        return {};
    }

    bool synthesize_streaming_session(
        GPTSoVITSSession& session,
        const SynthesisRequest& request,
        AudioChunkCallback callback
    ) {
        LaneLease lane = acquire_lane();
        if (!lane) return false;
        auto& impl = lane.impl();
        prepare_voice(session, request);
        PromptCacheAttachment cache_attachment(impl, session);
        RngAttachment rng_attachment(impl, session.rng_);

        const bool request_has_reference = !request.ref_audio.empty();
        const std::vector<float>& ref_audio = request_has_reference ? request.ref_audio : session.reference_.audio;
        const float* ref_audio_data = ref_audio.data();
        size_t ref_audio_len = ref_audio.size();
        const int ref_audio_sample_rate = request_has_reference ? 16000 : session.reference_.sample_rate;

        std::string ref_text = "";
        auto it_text = request.string_params.find("ref_text");
        if (it_text != request.string_params.end()) {
            ref_text = it_text->second;
        } else if (!request_has_reference) {
            ref_text = session.reference_.text;
        }

        std::string ref_lang = "zh";
        auto it_lang = request.string_params.find("ref_language");
        if (it_lang != request.string_params.end()) {
            ref_lang = it_lang->second;
        } else if (!request_has_reference && !session.reference_.language.empty()) {
            ref_lang = session.reference_.language;
        }

        float speed = 1.0f;
        auto it_speed = request.float_params.find("speed");
        if (it_speed != request.float_params.end()) {
            speed = it_speed->second;
        }

        ModelResidencyScope residency(impl, {0, 1, 2, 3});
        if (!residency) return false;

        if (impl.vits) {
            session.output_sample_rate_ = impl.vits->profile.output_sampling_rate;
        }
        CFMStepsScope cfm_steps(impl, request.float_params);

        std::string split_method = select_split_method();
        const auto it_split = request.string_params.find("split_method");
        if (it_split != request.string_params.end()) {
            split_method = it_split->second;
        }

        std::vector<std::string> segments = impl.frontend->split_text(request.text, split_method);
        if (segments.empty()) {
            return false;
        }

        // Process prompt cache once
        gpt_sovits_get_or_create_prompt_cache(&impl, session.cache_id_.c_str(), ref_audio_data, ref_audio_len, ref_audio_sample_rate, ref_text.c_str(), ref_lang.c_str(), nullptr, 0);

        const size_t pause_samples = 9600;
        std::vector<float> silence(pause_samples, 0.0f);

        for (size_t idx = 0; idx < segments.size(); ++idx) {
            const auto& seg_utf8 = segments[idx];
            int segment_samples = 0;

            const float* synth_audio = gpt_sovits_synthesize_single_segment_with_cache(
                &impl,
                seg_utf8.c_str(),
                request.language.c_str(),
                session.cache_id_.c_str(),
                speed,
                &segment_samples
            );

            if (synth_audio && segment_samples > 0) {
                // Call back immediately with the synthesized segment's audio!
                callback(synth_audio, segment_samples);

                // If there are more segments, send a short pause callback
                if (idx + 1 < segments.size()) {
                    callback(silence.data(), silence.size());
                }
            }
        }

        return true;
    }
};

std::vector<float> GPTSoVITSSession::synthesize(const SynthesisRequest& request) {
    return model_.synthesize_session(*this, request);
}

bool GPTSoVITSSession::set_reference(const VoiceReference& reference) {
    return model_.set_reference_session(*this, reference);
}

bool GPTSoVITSSession::synthesize_streaming(
    const SynthesisRequest& request,
    AudioChunkCallback callback
) {
    return model_.synthesize_streaming_session(*this, request, std::move(callback));
}

namespace gpt_sovits_provider {

std::shared_ptr<ITTSModel> load_model(
    const ModelConfig& config,
    const RuntimeContext& runtime
) {
    auto model = std::make_shared<GPTSoVITSModel>();
    return model->initialize(config, runtime) ? std::move(model) : nullptr;
}

} // namespace gpt_sovits_provider

} // namespace tts


