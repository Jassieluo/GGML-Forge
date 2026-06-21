#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include "gpt_sovits.h"
#include "models.h"
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

static bool g_log_enabled = true;

struct CoutSilencer {
    std::streambuf* old_buf;
    std::stringstream null_stream;
    bool active;

    CoutSilencer(bool silencer_active) : old_buf(nullptr), active(silencer_active) {
        if (active) {
            old_buf = std::cout.rdbuf();
            std::cout.rdbuf(null_stream.rdbuf());
        }
    }

    ~CoutSilencer() {
        if (active && old_buf) {
            std::cout.rdbuf(old_buf);
        }
    }
};



#ifndef GPT_SOVITS_DEBUG_ENABLED
#define GPT_SOVITS_DEBUG_ENABLED() (std::getenv("GPT_SOVITS_DEBUG") != nullptr)
#endif



#define GPT_SOVITS_DEBUG_PRINT(x) do { if (GPT_SOVITS_DEBUG_ENABLED()) { std::cout << "[GPT-SoVITS Debug] " << x << std::endl; std::fflush(stdout); } } while (0)



static std::u32string replace_all_u32(std::u32string str, const std::u32string& from, const std::u32string& to) {

    size_t start_pos = 0;

    while((start_pos = str.find(from, start_pos)) != std::u32string::npos) {

        str.replace(start_pos, from.length(), to);

        start_pos += to.length();

    }

    return str;

}



static const std::unordered_set<char32_t> splits_dict = {

    U'，', U'。', U'？', U'！', U',', U'.', U'?', U'!', U'~', U':', U'：', U'—', U'…', U'、', U';', U'；'

};



static bool is_subset_of_punctuation(const std::u32string& str) {

    static const std::unordered_set<char32_t> puncs = {

        U'!', U'?', U'…', U',', U'.', U'-', U' ', U'\t', U'\r', U'\n',

        U'，', U'。', U'？', U'！', U'~', U':', U'：', U'—', U'、', U';', U'；'

    };

    for (char32_t cp : str) {

        if (puncs.find(cp) == puncs.end()) {

            return false;

        }

    }

    return true;

}



static bool is_decimal_point(const std::u32string& inp, size_t i) {

    if (inp[i] != U'.') return false;

    if (i == 0 || i + 1 >= inp.size()) return false;

    char32_t prev = inp[i - 1];

    if (prev < U'0' || prev > U'9') return false;

    

    // Scan forward to see if the next significant character is a digit

    for (size_t j = i + 1; j < inp.size(); ++j) {

        char32_t next = inp[j];

        if (next >= U'0' && next <= U'9') {

            return true;

        }

        // Stop scanning if we hit letters, Chinese characters, or sentence/clause punctuation

        if ((next >= 0x4E00 && next <= 0x9FFF) || 

            (next >= U'A' && next <= U'Z') || 

            (next >= U'a' && next <= U'z') ||

            next == U'。' || next == U'？' || next == U'！' || next == U'?' || next == U'!' ||

            next == U'，' || next == U',' || next == U'、' || next == U'；' || next == U';') {

            break;

        }

    }

    return false;

}



static std::u32string clean_formatted_decimals(const std::u32string& text) {

    std::u32string result;

    result.reserve(text.size());

    for (size_t i = 0; i < text.size(); ++i) {

        char32_t cp = text[i];

        result.push_back(cp);

        

        // If we just pushed a period U'.' and the previous char was a digit

        if (cp == U'.' && result.size() >= 2 && result[result.size() - 2] >= U'0' && result[result.size() - 2] <= U'9') {

            // Check if there is a digit ahead, possibly separated by whitespace, newlines, or vertical bars

            size_t next_digit_idx = 0;

            for (size_t j = i + 1; j < text.size(); ++j) {

                char32_t n = text[j];

                if (n >= U'0' && n <= U'9') {

                    next_digit_idx = j;

                    break;

                }

                // Only skip whitespace, newlines, vertical bars, or common layout symbols

                if (n != U' ' && n != U'\t' && n != U'\r' && n != U'\n' && n != U'|' && n != U'│') {

                    break;

                }

            }

            // If we found a digit ahead, skip all intermediate formatting characters!

            if (next_digit_idx > 0) {

                i = next_digit_idx - 1; // Advance the outer loop index

            }

        }

    }

    return result;

}



static std::vector<std::u32string> split_python(std::u32string todo_text) {

    if (todo_text.empty()) return {};

    todo_text = replace_all_u32(todo_text, U"……", U"。");

    todo_text = replace_all_u32(todo_text, U"——", U"，");

    if (splits_dict.find(todo_text.back()) == splits_dict.end()) {

        todo_text += U"。";

    }

    size_t i_split_head = 0;

    size_t i_split_tail = 0;

    size_t len_text = todo_text.size();

    std::vector<std::u32string> todo_texts;

    while (true) {

        if (i_split_head >= len_text) break;

        if (splits_dict.find(todo_text[i_split_head]) != splits_dict.end()) {

            i_split_head++;

            todo_texts.push_back(todo_text.substr(i_split_tail, i_split_head - i_split_tail));

            i_split_tail = i_split_head;

        } else {

            i_split_head++;

        }

    }

    return todo_texts;

}



static std::vector<std::u32string> cut0(const std::u32string& inp) {

    if (!is_subset_of_punctuation(inp)) return { inp };

    return {};

}



static std::vector<std::u32string> cut1(const std::u32string& inp) {

    std::vector<std::u32string> inps = split_python(inp);

    if (inps.empty()) return {};

    std::vector<std::u32string> opts;

    for (size_t idx = 0; idx < inps.size(); idx += 4) {

        std::u32string merged = U"";

        for (size_t k = 0; k < 4 && idx + k < inps.size(); ++k) {

            merged += inps[idx + k];

        }

        opts.push_back(merged);

    }

    std::vector<std::u32string> filtered_opts;

    for (const auto& item : opts) {

        if (!is_subset_of_punctuation(item)) filtered_opts.push_back(item);

    }

    return filtered_opts;

}



static std::vector<std::u32string> cut2(const std::u32string& inp) {

    std::vector<std::u32string> inps = split_python(inp);

    if (inps.size() < 2) {

        if (!is_subset_of_punctuation(inp)) return { inp };

        return {};

    }

    std::vector<std::u32string> opts;

    size_t summ = 0;

    std::u32string tmp_str = U"";

    for (size_t i = 0; i < inps.size(); ++i) {

        summ += inps[i].size();

        tmp_str += inps[i];

        if (summ > 50) {

            summ = 0;

            opts.push_back(tmp_str);

            tmp_str = U"";

        }

    }

    if (!tmp_str.empty()) opts.push_back(tmp_str);

    if (opts.size() > 1 && opts.back().size() < 50) {

        opts[opts.size() - 2] = opts[opts.size() - 2] + opts.back();

        opts.pop_back();

    }

    std::vector<std::u32string> filtered_opts;

    for (const auto& item : opts) {

        if (!is_subset_of_punctuation(item)) filtered_opts.push_back(item);

    }

    return filtered_opts;

}



static std::vector<std::u32string> cut3(const std::u32string& inp) {

    std::vector<std::u32string> opts;

    std::u32string current = U"";

    for (char32_t cp : inp) {

        if (cp == U'。') {

            if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);

            current = U"";

        } else {

            current += cp;

        }

    }

    if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);

    return opts;

}



static std::vector<std::u32string> cut4(const std::u32string& inp) {

    std::vector<std::u32string> opts;

    std::u32string current = U"";

    for (size_t i = 0; i < inp.size(); ++i) {

        char32_t cp = inp[i];

        if (cp == U'.') {

            if (is_decimal_point(inp, i)) {

                current += cp;

            } else {

                if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);

                current = U"";

            }

        } else {

            current += cp;

        }

    }

    if (!current.empty() && !is_subset_of_punctuation(current)) opts.push_back(current);

    return opts;

}



static std::vector<std::u32string> cut5(const std::u32string& inp) {

    static const std::unordered_set<char32_t> punds = {

        U',', U'.', U';', U'?', U'!', U'、', U'，', U'。', U'？', U'！', U'；', U'：', U'…'

    };

    std::vector<std::u32string> mergeitems;

    std::u32string items = U"";

    for (size_t i = 0; i < inp.size(); ++i) {

        char32_t cp = inp[i];

        if (punds.find(cp) != punds.end()) {

            if (is_decimal_point(inp, i)) {

                items += cp;

            } else {

                items += cp;

                mergeitems.push_back(items);

                items = U"";

            }

        } else {

            items += cp;

        }

    }

    if (!items.empty()) mergeitems.push_back(items);

    std::vector<std::u32string> opts;

    for (const auto& item : mergeitems) {

        if (!is_subset_of_punctuation(item)) opts.push_back(item);

    }

    return opts;

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

    std::vector<float> host_data(nelements);

    if (tensor->buffer == nullptr) {

        if (tensor->type == GGML_TYPE_F32) {

            std::memcpy(host_data.data(), tensor->data, nelements * sizeof(float));

        } else if (tensor->type == GGML_TYPE_F16) {

            const ggml_fp16_t* temp = (const ggml_fp16_t*)tensor->data;

            for (int64_t i = 0; i < nelements; ++i) {

                host_data[i] = ggml_fp16_to_fp32(temp[i]);

            }

        } else {

            std::cerr << "[GPT-SoVITS] Error: Unsupported CPU tensor type " << tensor->type << " for float extraction" << std::endl;

        }

    } else {

        if (tensor->type == GGML_TYPE_F32) {

            ggml_backend_tensor_get(tensor, host_data.data(), 0, nelements * sizeof(float));

        } else if (tensor->type == GGML_TYPE_F16) {

            std::vector<ggml_fp16_t> temp_f16(nelements);

            ggml_backend_tensor_get(tensor, temp_f16.data(), 0, nelements * sizeof(ggml_fp16_t));

            for (int64_t i = 0; i < nelements; ++i) {

                host_data[i] = ggml_fp16_to_fp32(temp_f16[i]);

            }

        } else {

            std::cerr << "[GPT-SoVITS] Error: Unsupported backend tensor type " << tensor->type << " for float extraction" << std::endl;

        }

    }

    return host_data;

}



// Cooley-Tukey in-place FFT (power-of-2 length only)

static void fft_inplace(std::vector<std::complex<float>>& x) {

    const int n = (int)x.size();

    // Bit-reversal permutation

    for (int i = 1, j = 0; i < n; ++i) {

        int bit = n >> 1;

        for (; j & bit; bit >>= 1) j ^= bit;

        j ^= bit;

        if (i < j) std::swap(x[i], x[j]);

    }

    // Butterfly stages

    for (int len = 2; len <= n; len <<= 1) {

        float ang = -2.0f * (float)M_PI / len;

        std::complex<float> wlen(std::cos(ang), std::sin(ang));

        for (int i = 0; i < n; i += len) {

            std::complex<float> w(1.0f, 0.0f);

            for (int j = 0; j < len / 2; ++j) {

                std::complex<float> u = x[i + j];

                std::complex<float> v = x[i + j + len/2] * w;

                x[i + j]         = u + v;

                x[i + j + len/2] = u - v;

                w *= wlen;

            }

        }

    }

}





// Internal Prompt Cache representation in resident VRAM

struct PromptCache {

    std::string prompt_text;

    std::string prompt_lang;

    

    std::vector<std::string> prompt_phones;

    std::vector<int> prompt_word2ph;

    

    // Extracted HuBERT codes/indices (1D int sequence)

    std::vector<int32_t> hubert_codes;

    

    // Extracted BERT features for the prompt: [1024, len_phones]

    std::vector<float> bert_features;

    

    // Extracted Speaker embedding for style conditioning: [512]

    std::vector<float> speaker_embedding;

};



// Internal implementation of GPT_SoVITS

class Impl {

public:

    struct InternalParams {

        std::string dict_dir;

        std::string hubert_model_path;

        std::string bert_model_path;

        std::string t2s_model_path;

        std::string vits_model_path;

        int n_threads = 4;

        bool use_gpu = true;

    } params;

    

    std::unique_ptr<phonemizer::Phonemizer> phonemizer;

    std::unordered_map<std::string, int32_t> bert_vocab;

    std::unordered_map<std::string, int32_t> phone_to_id;

    

    // GGML Backends

    ggml_backend_t static_backend = nullptr;

    ggml_backend_t dynamic_backend = nullptr;

    ggml_backend_t vits_backend = nullptr;

    ggml_backend_t vits_target_backend = nullptr;

    ggml_backend_t bert_backend = nullptr;

    ggml_backend_t t2s_backend = nullptr;

    

    struct ggml_threadpool* static_threadpool = nullptr;

    struct ggml_threadpool* dynamic_threadpool = nullptr;

    struct ggml_threadpool* vits_threadpool = nullptr;

    struct ggml_threadpool* bert_threadpool = nullptr;

    struct ggml_threadpool* t2s_threadpool = nullptr;

    

    // persistent Static base models (CNHuBERT and RoBERTa BERT)

    std::unique_ptr<HubertModel> hubert;

    std::unique_ptr<BertModel> bert;

    

    // Dynamic speaker weights (hot-swapped)

    std::unique_ptr<T2SModel> t2s;

    std::unique_ptr<VITSModel> vits;

    

    // Double buffers for glitch-free speaker hot-swapping

    std::unique_ptr<T2SModel> t2s_standby;

    std::unique_ptr<VITSModel> vits_standby;

    

    // Resident Cache in VRAM

    std::unordered_map<std::string, PromptCache> prompt_caches;

    

    // Engine-resident output buffer for stable C-pointer access

    std::vector<float> last_synthesized_audio;



    // Persistent VITS graph allocator

    ggml_gallocr_t vits_galloc = nullptr;



    // Model Slots and Backend Registry

    struct ModelSlot {

        std::string path;

        std::string device = "cpu";

        bool is_resident = true;

        bool is_loaded = false;

    };

    ModelSlot slots[4];

    std::unordered_map<std::string, ggml_backend_t> device_backends;

    struct ggml_threadpool* shared_cpu_threadpool = nullptr;

    bool bypass_offload = false;



    ggml_backend_t get_backend_for_device(const std::string& device_name) {

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

                if (GPT_SOVITS_DEBUG_ENABLED()) {

                    std::cout << "[GPT-SoVITS Backend Registry] Initialized device backend: '" << ggml_backend_dev_name(found_dev) << "' for query '" << device_name << "'" << std::endl;

                }

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

                            if (GPT_SOVITS_DEBUG_ENABLED()) {

                                std::cout << "[GPT-SoVITS Backend Registry] Fallback initialized GPU device backend: '" << dev_name << "' for query '" << device_name << "'" << std::endl;

                            }

                            return b;

                        }

                    }

                }

            }

        }

        

        return get_backend_for_device("cpu");

    }



    void create_and_bind_shared_threadpool(ggml_backend_t backend) {

        if (!shared_cpu_threadpool) {

            auto * d = ggml_backend_get_device(backend);

            if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) {

                auto * reg = ggml_backend_dev_backend_reg(d);

                auto * ggml_threadpool_new_fn = (struct ggml_threadpool * (*)(const struct ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");

                if (ggml_threadpool_new_fn) {

                    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(params.n_threads);

                    shared_cpu_threadpool = ggml_threadpool_new_fn(&tpp);

                    if (shared_cpu_threadpool && GPT_SOVITS_DEBUG_ENABLED()) {

                        std::cout << "[GPT-SoVITS] Created shared CPU threadpool with " << params.n_threads << " threads" << std::endl;

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

                    if (GPT_SOVITS_DEBUG_ENABLED()) {

                        std::cout << "[GPT-SoVITS] Bound shared CPU threadpool to backend: " << backend << std::endl;

                    }

                }

            }

        }

    }



    bool load_model(int model_type) {

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

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS load_model] Loading model slot " << model_type 

                      << " (" << slot.path << ") onto backend device '" << slot.device << "'..." << std::endl;

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

            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::cout << "[GPT-SoVITS load_model] Model slot " << model_type << " loaded successfully." << std::endl;

            }

        } else {

            std::cerr << "[GPT-SoVITS load_model] Failed to load model slot " << model_type << "." << std::endl;

        }

        return ok;

    }



    void offload_model(int model_type) {

        if (model_type < 0 || model_type >= 4) return;

        if (bypass_offload) return;

        ModelSlot& slot = slots[model_type];

        if (!slot.is_loaded) return;

        if (slot.is_resident) {

            return;

        }

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS offload_model] Offloading model slot " << model_type 

                      << " to reclaim VRAM/memory..." << std::endl;

        }

                  

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

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS offload_model] Model slot " << model_type << " offloaded successfully." << std::endl;

        }

    }



    static void configure_sycl_jit_cache() {

        std::string s_persistent = get_env_var("SYCL_CACHE_PERSISTENT");

        if (!s_persistent.empty()) {

            std::cout << "[GPT-SoVITS SYCL Cache] Using programmatically configured settings: SYCL_CACHE_PERSISTENT=" 

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

            std::cout << "[GPT-SoVITS SYCL Cache] JIT persistent cache disabled via GPT_SOVITS_SYCL_CACHE_PERSISTENT." << std::endl;

            return;

        }



        set_env_var("SYCL_CACHE_PERSISTENT", "1");



        // Get cache directory

        std::string cache_dir = get_env_var("GPT_SOVITS_SYCL_CACHE_DIR");

        if (cache_dir.empty()) {

            // Fallback to existing SYCL_CACHE_DIR if set

            cache_dir = get_env_var("SYCL_CACHE_DIR");

        }



        if (cache_dir.empty()) {

            // Default to "./sycl_cache" in current working directory

            try {

                std::filesystem::path default_path = std::filesystem::current_path() / "sycl_cache";

                cache_dir = default_path.string();

            } catch (const std::exception& e) {

                cache_dir = "sycl_cache";

            }

        }



        // Set SYCL_CACHE_DIR

        set_env_var("SYCL_CACHE_DIR", cache_dir);



        // Create the directory if it doesn't exist

        try {

            std::filesystem::path p(cache_dir);

            if (!std::filesystem::exists(p)) {

                std::filesystem::create_directories(p);

                std::cout << "[GPT-SoVITS SYCL Cache] Created JIT cache directory: " << std::filesystem::absolute(p).string() << std::endl;

            } else {

                std::cout << "[GPT-SoVITS SYCL Cache] Using existing JIT cache directory: " << std::filesystem::absolute(p).string() << std::endl;

            }

        } catch (const std::exception& e) {

            std::cerr << "[GPT-SoVITS SYCL Cache] Warning: Failed to check/create cache directory " << cache_dir << ": " << e.what() << std::endl;

        }

    }



    Impl(

        const char* dict_dir,

        const char* hubert_model_path,

        const char* bert_model_path,

        const char* t2s_model_path,

        const char* vits_model_path,

        int n_threads,

        int backend_mode,

        const char* device_name = nullptr

    ) {

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Impl constructor start" << std::endl;

        }

        configure_sycl_jit_cache();

        ggml_backend_load_all(); // Load backends unconditionally first

        

        if (dict_dir) params.dict_dir = dict_dir;

        if (hubert_model_path) params.hubert_model_path = hubert_model_path;

        if (bert_model_path) params.bert_model_path = bert_model_path;

        if (t2s_model_path) params.t2s_model_path = t2s_model_path;

        if (vits_model_path) params.vits_model_path = vits_model_path;

        params.n_threads = n_threads;

        params.use_gpu = (backend_mode > 0);

        

        phonemizer = std::make_unique<phonemizer::Phonemizer>(params.dict_dir);

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Phonemizer frontend loaded" << std::endl;

        }

        

        const auto& syms = get_phone_symbols();

        for (size_t i = 0; i < syms.size(); ++i) {

            phone_to_id[syms[i]] = (int32_t)i;

        }

        

        load_bert_vocab(params.dict_dir + "/bert_vocab.txt");

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] BERT vocab loaded" << std::endl;

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

        

        bool is_sycl = (default_gpu_name.rfind("SYCL", 0) == 0);

        if (is_sycl) {

            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::cout << "[GPT-SoVITS] SYCL GPU detected. Routing all model slots to GPU." << std::endl;

            }

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



    ~Impl() {

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



    void load_bert_vocab(const std::string& path) {

        std::ifstream file(path);

        if (!file.is_open()) {

            std::cerr << "[GPT-SoVITS] Failed to open BERT vocabulary: " << path << "\n";

            return;

        }

        std::string line;

        int32_t idx = 0;

        while (std::getline(file, line)) {

            if (!line.empty() && line.back() == '\r') line.pop_back();

            bert_vocab[line] = idx++;

        }

        std::cout << "[GPT-SoVITS] Loaded " << bert_vocab.size() << " BERT vocab items\n";

    }



    std::vector<int32_t> bert_tokenize(const std::string& text) {

        std::vector<int32_t> ids;

        ids.push_back(101); // [CLS]

        

        std::u32string u32_chars = phonemizer::utf8_to_utf32(text);

        for (char32_t c : u32_chars) {

            std::string utf8_char = phonemizer::utf32_char_to_utf8(c);

            auto it = bert_vocab.find(utf8_char);

            if (it != bert_vocab.end()) {

                ids.push_back(it->second);

            } else {

                ids.push_back(100); // [UNK]

            }

        }

        

        ids.push_back(102); // [SEP]

        return ids;

    }



    std::vector<int32_t> phones_to_ids(const std::vector<std::string>& phones) {

        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {

            std::cout << "[GPT-SoVITS Debug] phones_to_ids called with phones count=" << phones.size() 

                      << ", phone_to_id size=" << phone_to_id.size() << std::endl; std::fflush(stdout);

        }

        std::vector<int32_t> ids;

        for (size_t idx = 0; idx < phones.size(); ++idx) {

            const auto& ph = phones[idx];

            if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {

                std::cout << "[GPT-SoVITS Debug] phones_to_ids: looking up ph[" << idx << "]=\"" << ph << "\"" << std::endl; std::fflush(stdout);

            }

            auto it = phone_to_id.find(ph);

            if (it != phone_to_id.end()) {

                ids.push_back(it->second);

            } else {

                ids.push_back(0); // padding/unk

            }

        }

        if (std::getenv("GPT_SOVITS_DEBUG") != nullptr) {

            std::cout << "[GPT-SoVITS Debug] phones_to_ids finished successfully." << std::endl; std::fflush(stdout);

        }

        return ids;

    }

};



static bool is_vits_3d_conv_weight(const std::string& name) {

    if (name.find("emb_rel_") != std::string::npos) return true;

    if (name.find(".weight") == std::string::npos) return false;

    

    if (name.find("attn_layers") != std::string::npos ||

        name.find("conv_") != std::string::npos ||

        name.find("convs") != std::string::npos ||

        name.find("ups") != std::string::npos ||

        name.find("temporal") != std::string::npos ||

        name.find("in_layers") != std::string::npos ||

        name.find("res_skip_layers") != std::string::npos ||

        name.find("cond_layer") != std::string::npos ||

        name.find("pre.weight") != std::string::npos ||

        name.find("proj.weight") != std::string::npos ||

        name.find("post.weight") != std::string::npos ||

        name.find("ssl_proj") != std::string::npos ||

        name.find("cond.weight") != std::string::npos) {

        return true;

    }

    

    return false;

}



bool load_gguf_model(const std::string& path, GGUFModel& model, ggml_backend_t backend) {

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Loading GGUF model: " << path << std::endl;

    bool is_transposed_vits = false;

    

    // 1. Load weights metadata only (no_alloc = true)

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 1: Loading GGUF metadata..." << std::endl;

    struct ggml_context* ggml_ctx_backend = nullptr;

    struct gguf_init_params params_backend = {

        /* .no_alloc = */ true,

        /* .ctx      = */ &ggml_ctx_backend

    };

    struct gguf_context* ctx_gguf_backend = gguf_init_from_file(path.c_str(), params_backend);

    if (!ctx_gguf_backend) {

        fprintf(stderr, "[GPT-SoVITS] Failed to load GGUF metadata from %s\n", path.c_str());

        return false;

    }

    

    int kid = gguf_find_key(ctx_gguf_backend, "attention.head_count");

    if (kid >= 0) {

        model.n_heads = (int)gguf_get_val_u32(ctx_gguf_backend, kid);

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Found attention.head_count in GGUF: " << model.n_heads << std::endl;

    } else {

        std::string path_lower = path;

        for (auto& c : path_lower) c = std::tolower((unsigned char)c);

        if (path_lower.find("s1bert") != std::string::npos || 

            path_lower.find("base") != std::string::npos || 

            path_lower.find("369668") != std::string::npos) {

            model.n_heads = 16;

            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Warning: attention.head_count not found in GGUF. Detected v2base pattern in filename, configuring n_heads to 16." << std::endl;

        } else {

            model.n_heads = 8;

            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Warning: attention.head_count not found in GGUF. Defaulting to: 8" << std::endl;

        }

    }

    

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 1 finished. Backend context: " << ggml_ctx_backend << std::endl;

    

    // 1.5. Pre-adjust transposed convolution tensor types and shapes in metadata context

    // before allocating backend buffers to avoid out-of-bound writes when copying dequantized data.

    {

        int n_tensors_meta = (int)gguf_get_n_tensors(ctx_gguf_backend);

        for (int i = 0; i < n_tensors_meta; ++i) {

            std::string name = gguf_get_tensor_name(ctx_gguf_backend, i);

            struct ggml_tensor* t_backend = ggml_get_tensor(ggml_ctx_backend, name.c_str());

            if (t_backend) {

                bool is_transposed = is_transposed_vits && is_vits_3d_conv_weight(name);

                if (is_transposed) {

                    // Force the backend tensor allocation type to F16

                    t_backend->type = GGML_TYPE_F16;

                    

                    int64_t out_channels = t_backend->ne[0];

                    int64_t in_channels  = t_backend->ne[1];

                    int64_t kernel_size  = t_backend->ne[2];

                    

                    t_backend->ne[0] = kernel_size;

                    t_backend->ne[1] = in_channels;

                    t_backend->ne[2] = out_channels;

                    

                    t_backend->nb[0] = ggml_type_size(t_backend->type);

                    t_backend->nb[1] = t_backend->nb[0] * t_backend->ne[0];

                    t_backend->nb[2] = t_backend->nb[1] * t_backend->ne[1];

                    t_backend->nb[3] = t_backend->nb[2] * t_backend->ne[2];

                    

                    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Pre-adjusted metadata for transposed tensor: " << name

                              << " -> type to F16, shape to [" << kernel_size << ", " << in_channels << ", " << out_channels << "]" << std::endl;

                }

            }

        }

    }

    

    // 2. Allocate the tensors on the backend

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 2: Allocating backend buffer..." << std::endl;

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx_backend, backend);

    if (!buffer) {

        fprintf(stderr, "[GPT-SoVITS] Failed to allocate backend buffer for GGUF: %s\n", path.c_str());

        gguf_free(ctx_gguf_backend);

        return false;

    }

    model.backend_buffer = buffer;

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 2 finished. Buffer: " << buffer << " (size: " << (ggml_backend_buffer_get_size(buffer) / (1024.0 * 1024.0)) << " MB)" << std::endl;

    

    // 3. Load weight data on CPU (no_alloc = false)

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 3: Loading GGUF CPU weights data..." << std::endl;

    struct ggml_context* ggml_ctx_cpu = nullptr;

    struct gguf_init_params params_cpu = {

        /* .no_alloc = */ false,

        /* .ctx      = */ &ggml_ctx_cpu

    };

    struct gguf_context* ctx_gguf_cpu = gguf_init_from_file(path.c_str(), params_cpu);

    if (!ctx_gguf_cpu) {

        fprintf(stderr, "[GPT-SoVITS] Failed to load GGUF data on CPU: %s\n", path.c_str());

        gguf_free(ctx_gguf_backend);

        return false;

    }

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 3 finished. CPU context: " << ggml_ctx_cpu << std::endl;

    

    // 4. Copy weight data from CPU tensors to backend tensors

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 4: Copying weights from CPU to backend..." << std::endl;

    int n_tensors = (int)gguf_get_n_tensors(ctx_gguf_backend);

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Total tensors to copy: " << n_tensors << std::endl;

    for (int i = 0; i < n_tensors; ++i) {

        std::string name = gguf_get_tensor_name(ctx_gguf_backend, i);

        struct ggml_tensor* t_backend = ggml_get_tensor(ggml_ctx_backend, name.c_str());

        struct ggml_tensor* t_cpu = ggml_get_tensor(ggml_ctx_cpu, name.c_str());

        if (t_backend && t_cpu) {

            bool is_transposed = is_transposed_vits && is_vits_3d_conv_weight(name);

            if (is_transposed) {

                int64_t out_channels = t_cpu->ne[0];

                int64_t in_channels  = t_cpu->ne[1];

                int64_t kernel_size  = t_cpu->ne[2];

                

                int64_t nelems = ggml_nelements(t_cpu);

                std::vector<float> f32_buf(nelems);

                

                if (t_cpu->type == GGML_TYPE_F32) {

                    std::memcpy(f32_buf.data(), t_cpu->data, nelems * sizeof(float));

                } else if (t_cpu->type == GGML_TYPE_F16) {

                    ggml_fp16_to_fp32_row((const ggml_fp16_t*)t_cpu->data, f32_buf.data(), nelems);

                } else if (t_cpu->type == GGML_TYPE_Q8_0) {

                    int64_t block_size = 32;

                    int64_t n_blocks = nelems / block_size;

                    struct q8_0_block {

                        ggml_fp16_t d;

                        int8_t qs[32];

                    };

                    const q8_0_block* blocks = (const q8_0_block*)t_cpu->data;

                    for (int64_t b = 0; b < n_blocks; ++b) {

                        float scale = ggml_fp16_to_fp32(blocks[b].d);

                        for (int j = 0; j < 32; ++j) {

                            f32_buf[b * 32 + j] = blocks[b].qs[j] * scale;

                        }

                    }

                } else {

                    std::cerr << "[load_gguf_model] Warning: unsupported transposed tensor type: " << t_cpu->type << std::endl;

                    ggml_backend_tensor_set(t_backend, t_cpu->data, 0, ggml_nbytes(t_cpu));

                    continue;

                }

                

                std::vector<float> transposed_buf(nelems);

                for (int64_t oc = 0; oc < out_channels; ++oc) {

                    for (int64_t ic = 0; ic < in_channels; ++ic) {

                        for (int64_t k = 0; k < kernel_size; ++k) {

                            int64_t src_idx = k * (in_channels * out_channels) + ic * out_channels + oc;

                            int64_t dst_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;

                            transposed_buf[dst_idx] = f32_buf[src_idx];

                        }

                    }

                }

                

                if (t_backend->type == GGML_TYPE_F16) {

                    std::vector<ggml_fp16_t> f16_buf(nelems);

                    ggml_fp32_to_fp16_row(transposed_buf.data(), f16_buf.data(), nelems);

                    ggml_backend_tensor_set(t_backend, f16_buf.data(), 0, nelems * sizeof(ggml_fp16_t));

                } else {

                    ggml_backend_tensor_set(t_backend, transposed_buf.data(), 0, nelems * sizeof(float));

                }

                

                t_backend->ne[0] = kernel_size;

                t_backend->ne[1] = in_channels;

                t_backend->ne[2] = out_channels;

                

                t_backend->nb[0] = ggml_type_size(t_backend->type);

                t_backend->nb[1] = t_backend->nb[0] * t_backend->ne[0];

                t_backend->nb[2] = t_backend->nb[1] * t_backend->ne[1];

                t_backend->nb[3] = t_backend->nb[2] * t_backend->ne[2];

                

                if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Successfully dequantized, transposed, and restored shape for weight tensor: " << name 

                          << " | Original shape: [" << kernel_size << ", " << in_channels << ", " << out_channels << "]" << std::endl;

            } else {

                ggml_backend_tensor_set(t_backend, t_cpu->data, 0, ggml_nbytes(t_cpu));

            }

        }

    }

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 4 finished." << std::endl;

    

    // 5. Free temporary CPU contexts

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 5: Freeing CPU context..." << std::endl;

    ggml_free(ggml_ctx_cpu);

    gguf_free(ctx_gguf_cpu);

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 5 finished." << std::endl;

    

    // 6. Save backend context and populate tensors map

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] Step 6: Populating tensors map..." << std::endl;

    model.ctx = ggml_ctx_backend;

    for (int i = 0; i < n_tensors; ++i) {

        std::string name = gguf_get_tensor_name(ctx_gguf_backend, i);

        struct ggml_tensor* tensor = ggml_get_tensor(ggml_ctx_backend, name.c_str());

        if (tensor) {

            model.tensors[name] = tensor;

        }

    }

    

    gguf_free(ctx_gguf_backend);

    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[load_gguf_model] GGUF loaded successfully." << std::endl;

    return true;

}



} // namespace gpt_sovits



using namespace gpt_sovits;



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

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

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

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[gpt_sovits_load_speaker] Swapped successfully!\n";

    }

    return true;

}

struct LangSegment {
    std::string text;
    std::string lang;
};

static std::vector<LangSegment> split_zh_en(const std::u32string& text) {
    if (text.empty()) return {};
    
    std::vector<LangSegment> segments;
    std::u32string current_text;
    std::string current_lang = "";
    
    auto is_cjk = [](char32_t cp) {
        if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
        if (cp >= 0x3400 && cp <= 0x4DBF) return true;
        if (cp == U'，' || cp == U'。' || cp == U'？' || cp == U'！' || cp == U'、' || 
            cp == U'；' || cp == U'：' || cp == U'“' || cp == U'”' || cp == U'…') {
            return true;
        }
        return false;
    };
    
    auto is_english_char = [](char32_t cp) {
        if (cp >= U'A' && cp <= U'Z') return true;
        if (cp >= U'a' && cp <= U'z') return true;
        return false;
    };
    
    for (char32_t cp : text) {
        std::string char_lang = "";
        if (is_cjk(cp)) {
            char_lang = "zh";
        } else if (is_english_char(cp)) {
            char_lang = "en";
        } else {
            char_lang = current_lang.empty() ? "zh" : current_lang;
        }
        
        if (current_lang.empty()) {
            current_lang = char_lang;
            current_text.push_back(cp);
        } else if (current_lang == char_lang) {
            current_text.push_back(cp);
        } else {
            segments.push_back({phonemizer::utf32_to_utf8(current_text), current_lang});
            current_text = {cp};
            current_lang = char_lang;
        }
    }
    if (!current_text.empty()) {
        segments.push_back({phonemizer::utf32_to_utf8(current_text), current_lang});
    }
    
    std::vector<LangSegment> merged;
    for (const auto& seg : segments) {
        if (merged.empty()) {
            merged.push_back(seg);
        } else if (merged.back().lang == seg.lang) {
            merged.back().text += seg.text;
        } else {
            merged.push_back(seg);
        }
    }
    return merged;
}

static bool process_text_mixed(
    Impl* impl,
    struct ggml_context* ctx_graph,
    const std::string& text,
    const std::string& language,
    phonemizer::PhonemizerResult& out_res,
    std::vector<float>& out_bert_features
) {
    if (language != "zh" && language != "zh_en") {
        out_res = impl->phonemizer->process(text, language);
        out_bert_features.assign(1024 * out_res.phones.size(), 0.0f);
        return false;
    }
    
    // Mixed Mode segmentation
    std::u32string u32_text = phonemizer::utf8_to_utf32(text);
    std::vector<LangSegment> segments = split_zh_en(u32_text);
    
    std::vector<std::string> combined_phones;
    std::vector<int> combined_word2ph;
    std::string combined_norm_text;
    std::vector<float> combined_bert_aligned;
    
    for (const auto& seg : segments) {
        phonemizer::PhonemizerResult seg_res = impl->phonemizer->process(seg.text, seg.lang);
        
        combined_phones.insert(combined_phones.end(), seg_res.phones.begin(), seg_res.phones.end());
        combined_word2ph.insert(combined_word2ph.end(), seg_res.word2ph.begin(), seg_res.word2ph.end());
        combined_norm_text += seg_res.norm_text;
        
        int seg_phone_len = (int)seg_res.phones.size();
        
        if (seg.lang == "zh") {
            std::vector<int32_t> target_bert_ids = impl->bert_tokenize(seg_res.norm_text);
            struct ggml_tensor* seg_bert_out = impl->bert->forward(ctx_graph, target_bert_ids, impl->bert_backend);
            
            int seg_bert_len = seg_bert_out ? seg_bert_out->ne[1] : 0;
            std::vector<float> seg_bert_data(1024 * seg_bert_len, 0.0f);
            if (seg_bert_out) {
                safe_ggml_backend_tensor_get(seg_bert_out, seg_bert_data.data(), 0, 1024 * seg_bert_len * sizeof(float));
            }
            
            std::vector<int> seg_phone_to_word;
            for (size_t w_idx = 0; w_idx < seg_res.word2ph.size(); ++w_idx) {
                int num_phones = seg_res.word2ph[w_idx];
                for (int p = 0; p < num_phones; ++p) {
                    seg_phone_to_word.push_back((int)w_idx);
                }
            }
            
            for (int p_idx = 0; p_idx < seg_phone_len; ++p_idx) {
                int w_idx = 0;
                if (p_idx < (int)seg_phone_to_word.size()) {
                    w_idx = seg_phone_to_word[p_idx] + 1; // +1 to skip [CLS]
                }
                if (w_idx >= seg_bert_len) w_idx = seg_bert_len - 1;
                if (w_idx < 0) w_idx = 0;
                
                size_t start_idx = combined_bert_aligned.size();
                combined_bert_aligned.resize(start_idx + 1024);
                std::memcpy(combined_bert_aligned.data() + start_idx, seg_bert_data.data() + w_idx * 1024, 1024 * sizeof(float));
            }
        } else {
            size_t start_idx = combined_bert_aligned.size();
            combined_bert_aligned.resize(start_idx + 1024 * seg_phone_len, 0.0f);
        }
    }
    
    out_res.phones = combined_phones;
    out_res.word2ph = combined_word2ph;
    out_res.norm_text = combined_norm_text;
    out_bert_features = combined_bert_aligned;
    return true;
}

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

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Hit resident feature cache for ID: " << cid << "\n";

        }

        return;

    }

    

    impl->load_model(0); // Hubert

    impl->load_model(1); // BERT

    impl->load_model(3); // VITS

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS] Cache miss for ID: " << cid << ". Extracting features..." << std::endl;

    }

    

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
        std::cout << "[GPT-SoVITS] Step 1: Processing phonemes..." << std::endl;
        phonemizer::PhonemizerResult phone_res;
        std::vector<float> aligned_bert;
        process_text_mixed(impl, ctx_graph, cache.prompt_text, cache.prompt_lang, phone_res, aligned_bert);

        cache.prompt_phones = phone_res.phones;
        cache.prompt_word2ph = phone_res.word2ph;

        std::cout << "[GPT-SoVITS] Step 1 finished. Phonemes count: " << phone_res.phones.size() << std::endl;

        std::cout << "[GPT-SoVITS] Prompt Phones: ";

        for (const auto& ph : phone_res.phones) std::cout << "'" << ph << "' ";

        std::cout << "\n";

        

        // 2. CNHuBERT codes

        std::cout << "[GPT-SoVITS] Step 2: Running CNHuBERT..." << std::endl;

        

        // Convert ref_audio to ggml_tensor with zero padding at the end matching PyTorch's padding (9600 samples)

        size_t pad_samples = 9600; 

        size_t total_samples = ref_audio_len + pad_samples;

        struct ggml_tensor* input_audio = ggml_new_tensor_1d(ctx_graph, GGML_TYPE_F32, total_samples);

        std::memcpy(input_audio->data, ref_audio_data, ref_audio_len * sizeof(float));

        std::memset((float*)input_audio->data + ref_audio_len, 0, pad_samples * sizeof(float));

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            float min_val = 1e30f;

            float max_val = -1e30f;

            for (size_t i = 0; i < ref_audio_len; ++i) {

                float v = ref_audio_data[i];

                if (v < min_val) min_val = v;

                if (v > max_val) max_val = v;

            }

            std::cout << "[GPT-SoVITS Debug] C++ raw ref_audio min: " << min_val << " max: " << max_val << std::endl;

        }

        

        // Match PyTorch's bypassed feature extractor behavior: feed raw audio directly to CNHuBERT without normalization!

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Normalization bypassed to match PyTorch model requirements." << std::endl;

        }

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            float* audio_ptr = (float*)input_audio->data;

            std::cout << "[GPT-SoVITS Debug] C++ input_audio[:10] (normalized): ";

            for (int i = 0; i < 10; ++i) {

                std::cout << audio_ptr[i] << " ";

            }

            std::cout << std::endl;

        }

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Step 2: Calling HubertModel::forward..." << std::endl;

        }

        struct ggml_tensor* ssl_content = impl->hubert->forward(ctx_graph, input_audio, impl->vits_backend);

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] HubertModel::forward returned successfully!" << std::endl;

        }

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            float* ssl_content_cpu = (float*)ssl_content->data;

            std::cout << "[GPT-SoVITS Debug] C++ ssl_content[0, 0, :10]: ";

            for (int i = 0; i < 10; ++i) {

                std::cout << ssl_content_cpu[i * 768] << " ";

            }

            std::cout << std::endl;

        }



        // Project and quantize using SoVITS VITS quantizer to get hubert_codes

        int n_frames = ssl_content->ne[1]; // seq_len

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            float* ssl_content_cpu = (float*)ssl_content->data;

            std::ofstream ssl_file("scratch/pipeline_alignment_cpp_ssl_content.f32", std::ios::binary);

            if (ssl_file.is_open()) {

                ssl_file.write(reinterpret_cast<const char*>(ssl_content_cpu), 768 * n_frames * sizeof(float));

            }

        }

        cache.hubert_codes.resize(n_frames);

        

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Step 2: Retrieving VITS quantizer tensors..." << std::endl;

        }

        struct ggml_tensor* cb = impl->vits->get_tensor("quantizer.vq.layers.0._codebook.embed");

        struct ggml_tensor* ssl_proj_w = impl->vits->get_tensor("ssl_proj.weight");

        struct ggml_tensor* ssl_proj_b = impl->vits->get_tensor("ssl_proj.bias");

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] Tensors retrieved:\n"

                      << "  cb: " << cb << " (type=" << (cb ? cb->type : -1) << ", buffer=" << (cb ? cb->buffer : nullptr) << ")\n"

                      << "  ssl_proj_w: " << ssl_proj_w << " (type=" << (ssl_proj_w ? ssl_proj_w->type : -1) << ", buffer=" << (ssl_proj_w ? ssl_proj_w->buffer : nullptr) << ")\n"

                      << "  ssl_proj_b: " << ssl_proj_b << " (type=" << (ssl_proj_b ? ssl_proj_b->type : -1) << ", buffer=" << (ssl_proj_b ? ssl_proj_b->buffer : nullptr) << ")" << std::endl;

        }

        

        if (!cb || !ssl_proj_w || !ssl_proj_b) {

            std::cerr << "[GPT-SoVITS] Warning: Missing VITS quantizer GGUF tensors! Falling back to mock codes." << std::endl;

            for (int i = 0; i < n_frames; ++i) {

                cache.hubert_codes[i] = i % 1024; // Mock fallback

            }

        } else {

            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::cout << "[GPT-SoVITS] Running CPU FP32 VITS projection & Vector Quantization..." << std::endl;

            }



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

            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::ofstream w_file("scratch/pipeline_alignment_cpp_ssl_proj_weight.f32", std::ios::binary);

                if (w_file.is_open()) {

                    w_file.write(reinterpret_cast<const char*>(w_data.data()), w_data.size() * sizeof(float));

                }

                std::ofstream b_file("scratch/pipeline_alignment_cpp_ssl_proj_bias.f32", std::ios::binary);

                if (b_file.is_open()) {

                    b_file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));

                }

            }



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



            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::ofstream proj_file("scratch/pipeline_alignment_cpp_ssl_proj.f32", std::ios::binary);

                if (proj_file.is_open()) {

                    proj_file.write(reinterpret_cast<const char*>(projected_data.data()), projected_data.size() * sizeof(float));

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



            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::cout << "[GPT-SoVITS] First 10 semantic codes: ";

                for (int i = 0; i < std::min(10, out_frames); ++i) {

                    std::cout << cache.hubert_codes[i] << " ";

                }

                std::cout << std::endl;



                std::ofstream codes_file("scratch/pipeline_alignment_cpp_prompt_codes.txt");

                if (codes_file.is_open()) {

                    for (size_t i = 0; i < cache.hubert_codes.size(); ++i) {

                        codes_file << cache.hubert_codes[i] << (i + 1 == cache.hubert_codes.size() ? "" : ",");

                    }

                }

            }



            if (GPT_SOVITS_DEBUG_ENABLED()) {

                std::cout << "[GPT-SoVITS] Vector Quantization completed successfully. Generated "

                          << out_frames << " real semantic codes." << std::endl;

            }

        }

        

        // 3. BERT Features
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[GPT-SoVITS] Step 3: Assigning precomputed BERT features..." << std::endl;
        }
        cache.bert_features = aligned_bert;

        

        // 4. Compute speaker embedding (ge) via ref_enc from reference audio STFT spectrogram

        // Python: refer = spectrogram_torch(audio, n_fft=2048, sr=32000, hop=640, win=2048, center=False)

        //         ge = ref_enc(refer[:, :704] * refer_mask, refer_mask)

        // We compute STFT magnitude spectrogram then slice first 704 bins.

        std::cout << "[GPT-SoVITS] Step 4: Computing speaker embedding (ge) via ref_enc..." << std::endl;

        {

            // STFT parameters matching Python

            const int n_fft     = 2048;

            const int hop_len   = 640;   // at 32kHz

            // const int n_spec    = n_fft / 2 + 1;  // 1025

            const int n_ref_enc = 704;            // ref_enc uses first 704 bins



            // The reference audio is at 16kHz but VITS expects 32kHz.

            // Upsample 16kHz -> 32kHz via linear interpolation

            std::vector<float> audio32k(ref_audio_len * 2);

            for (size_t i = 0; i < ref_audio_len; ++i) {

                audio32k[2 * i]     = ref_audio_data[i];

                audio32k[2 * i + 1] = (i + 1 < ref_audio_len)

                    ? 0.5f * (ref_audio_data[i] + ref_audio_data[i + 1])

                    : ref_audio_data[i];

            }

            // Clamp to [-1, 1]

            float maxx = 0.0f;

            for (float v : audio32k) maxx = std::max(maxx, std::abs(v));

            if (maxx > 1.0f) for (float& v : audio32k) v /= std::min(2.0f, maxx);



            // Compute STFT frames (center=False: no padding)

            const size_t n_samples32k = audio32k.size();

            int n_frames = (int)((n_samples32k - n_fft) / hop_len) + 1;

            if (n_frames % 2 != 0) {

                n_frames -= 1;

            }



            if (n_frames > 0) {

                // Build Hann window

                std::vector<float> window(n_fft);

                for (int i = 0; i < n_fft; ++i) {

                    window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n_fft)));

                }



                // Compute magnitude spectrogram using FFT-based STFT

                // Output layout: spec_data[f * n_spec + k] = magnitude of bin k at frame f

                // ggml tensor [ne0=n_ref_enc, ne1=n_frames]: data stored as [n_frames rows × n_ref_enc cols]

                // i.e., data[f * n_ref_enc + k] for k in [0, n_ref_enc)

                std::vector<std::complex<float>> fft_buf(n_fft);

                std::vector<float> ref_enc_input(n_ref_enc * n_frames, 0.0f);

                for (int f = 0; f < n_frames; ++f) {

                    int offset = f * hop_len;

                    // Load windowed frame into FFT buffer

                    for (int i = 0; i < n_fft; ++i) {

                        fft_buf[i] = std::complex<float>(audio32k[offset + i] * window[i], 0.0f);

                    }

                    // In-place FFT

                    fft_inplace(fft_buf);

                    // Extract magnitude for first n_ref_enc bins, store in row-major [f, k]

                    for (int k = 0; k < n_ref_enc; ++k) {

                        float re = fft_buf[k].real();

                        float im = fft_buf[k].imag();

                        ref_enc_input[f * n_ref_enc + k] = std::sqrt(re * re + im * im + 1e-8f);

                    }

                }



                // Create ggml context for ref_enc graph

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

                        std::cout << "[GPT-SoVITS] Step 4: Speaker embedding computed! shape=[" << ge_tensor->ne[0] << ", " << ge_tensor->ne[1] << "]" << std::endl;

                        float ge_sum = 0.0f;

                        for (float v : cache.speaker_embedding) ge_sum += std::abs(v);

                        std::cout << "[GPT-SoVITS] Step 4: ge L1 norm=" << ge_sum << std::endl;

                        ggml_backend_buffer_free(ge_buf);

                    }

                }

                ggml_free(ctx_ge_graph);

                if (ge_input_buf) ggml_backend_buffer_free(ge_input_buf);

                ggml_free(ctx_ge);

            } else {

                std::cout << "[GPT-SoVITS] Step 4: Warning: ref audio too short for STFT (" << n_samples32k << " samples, need >= " << n_fft << "). Using zero ge." << std::endl;

                cache.speaker_embedding.assign(512, 0.0f);

            }

        }

        std::cout << "[GPT-SoVITS] Step 4: ge computation complete." << std::endl;

        

        // Clean up graph context

        ggml_free(ctx_graph);

        

        impl->prompt_caches[cid] = cache;

        if (GPT_SOVITS_DEBUG_ENABLED()) {

            std::cout << "[GPT-SoVITS] [Diagnostic] Nest block completed. Destructors about to run..." << std::endl;

        }

    }

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS] [Diagnostic] Destructors executed successfully!" << std::endl;

        std::cout << "[GPT-SoVITS] PromptCache cached and returning successfully!" << std::endl;

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

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS Debug] Got PromptCache. prompt_text size: " << cached_prompt.prompt_text.size() << std::endl;

        std::cout << "[GPT-SoVITS Debug] Synthesizing: \"" << text << "\" using cached prompt...\n";

        std::fflush(stdout);

    }

    

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

        std::cout << "[GPT-SoVITS] T2S_OVERRIDE_PHONES_FILE and T2S_OVERRIDE_BERT_FILE are set. Loading preprocessed features..." << std::endl;

        

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

            std::cout << "[GPT-SoVITS] Loaded " << text_len << " phones. prompt_len=" << prompt_len << ", target_len=" << target_len << std::endl;

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

            std::cout << "[GPT-SoVITS] Loaded " << fused_bert_aligned.size() << " floats from override BERT file." << std::endl;

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
        if (lang_str == "zh" || lang_str == "zh_en") {
            struct ggml_init_params init_params = {
                /* .mem_size   = */ 256 * 1024 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ false
            };
            ctx_graph = ggml_init(init_params);
            
            std::vector<float> target_bert_aligned;
            process_text_mixed(impl, ctx_graph, std::string(text), lang_str, target_res, target_bert_aligned);
            
            target_phone_ids = impl->phones_to_ids(target_res.phones);
            prompt_phone_ids = impl->phones_to_ids(cached_prompt.prompt_phones);
            
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
            std::memcpy(fused_bert_aligned.data(), cached_prompt.bert_features.data(), 1024 * prompt_len * sizeof(float));
            std::memcpy(fused_bert_aligned.data() + 1024 * prompt_len, target_bert_aligned.data(), 1024 * target_len * sizeof(float));
            
            bert_features_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, text_len);
            std::memcpy(bert_features_tensor->data, fused_bert_aligned.data(), 1024 * text_len * sizeof(float));
            
            target_bert_out = bert_features_tensor; // set to non-null
        } else {
            target_res = impl->phonemizer->process(std::string(text), lang_str);
            target_phone_ids = impl->phones_to_ids(target_res.phones);
            prompt_phone_ids = impl->phones_to_ids(cached_prompt.prompt_phones);
            
            if (target_phone_ids.empty()) {
                std::cerr << "[GPT-SoVITS] Error: Target text contains no valid phonemes! Cannot synthesize.\n";
                impl->offload_model(1);
                impl->offload_model(2);
                impl->offload_model(3);
                return nullptr;
            }
            
            struct ggml_init_params init_params = {
                /* .mem_size   = */ 64 * 1024 * 1024,
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ false
            };
            ctx_graph = ggml_init(init_params);
            
            target_len = target_phone_ids.size();
            text_len = prompt_len + target_len;
            
            fused_bert_aligned.assign(1024 * text_len, 0.0f);
            std::memcpy(fused_bert_aligned.data(), cached_prompt.bert_features.data(), 1024 * prompt_len * sizeof(float));
            
            bert_features_tensor = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 1024, text_len);
            std::memcpy(bert_features_tensor->data, fused_bert_aligned.data(), 1024 * text_len * sizeof(float));
        }
        
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[GPT-SoVITS] Target Phones: ";
            for (const auto& ph : target_res.phones) std::cout << "'" << ph << "' ";
            std::cout << "\n";
        }
        
        int64_t t_phonemizer_bert_end = ggml_time_us();
        std::cout << "[Profile] Phonemizer & BERT took " << (t_phonemizer_bert_end - t_phonemizer_bert_start) / 1000.0 << " ms" << std::endl;
    }

    

    // 3. Predict semantic codes using T2S (autoregressive transformer decoder)

    std::vector<int32_t> pred_semantics;

    const char* override_file = std::getenv("T2S_OVERRIDE_CODES_FILE");

    if (override_file) {

        std::cout << "[GPT-SoVITS] T2S_OVERRIDE_CODES_FILE is set. Loading codes from: " << override_file << std::endl;

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

            std::cout << "[GPT-SoVITS] Loaded " << pred_semantics.size() << " override semantic codes." << std::endl;

        } else {

            std::cerr << "[GPT-SoVITS] Error: Failed to open override codes file: " << override_file << std::endl;

        }

    }



    if (pred_semantics.empty()) {

        const char* override_prompt_semantic = std::getenv("T2S_OVERRIDE_PROMPT_SEMANTIC_FILE");

        std::vector<int32_t> hubert_codes = cached_prompt.hubert_codes;

        if (override_prompt_semantic) {

            std::cout << "[GPT-SoVITS] Loading override prompt semantics from: " << override_prompt_semantic << std::endl;

            std::ifstream f_prompt(override_prompt_semantic, std::ios::binary);

            if (f_prompt.is_open()) {

                f_prompt.seekg(0, std::ios::end);

                size_t size = f_prompt.tellg();

                f_prompt.seekg(0, std::ios::beg);

                hubert_codes.resize(size / sizeof(int32_t));

                f_prompt.read(reinterpret_cast<char*>(hubert_codes.data()), size);

                std::cout << "[GPT-SoVITS] Loaded " << hubert_codes.size() << " prompt semantic tokens." << std::endl;

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

        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[Profile] T2S Autoregressive Transformer took " << (t_t2s_end - t_t2s_start) / 1000.0 << " ms" << std::endl;

    }

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS Debug] T2S forward finished. pred_semantics size=" << pred_semantics.size() << std::endl;

    }

    

    // Dump T2S outputs for reverse cross-testing

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::ofstream tokens_file("scratch/cpp_t2s_tokens.bin", std::ios::binary);

        if (tokens_file.is_open()) {

            tokens_file.write(reinterpret_cast<const char*>(pred_semantics.data()), pred_semantics.size() * sizeof(int32_t));

            std::cout << "[T2S Dump] Saved " << pred_semantics.size() << " semantic tokens to scratch/cpp_t2s_tokens.bin\n";

        }

        std::ofstream phones_file("scratch/cpp_t2s_phones.bin", std::ios::binary);

        if (phones_file.is_open()) {

            phones_file.write(reinterpret_cast<const char*>(target_phone_ids.data()), target_phone_ids.size() * sizeof(int32_t));

            std::cout << "[T2S Dump] Saved " << target_phone_ids.size() << " phones to scratch/cpp_t2s_phones.bin\n";

        }

    }

    

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

    

    struct ggml_tensor* target_phone_tensor = ggml_new_tensor_1d(ctx_vits, GGML_TYPE_I32, target_phone_ids.size());

    struct ggml_tensor* target_word2ph_tensor = ggml_new_tensor_1d(ctx_vits, GGML_TYPE_I32, word2ph_size);

    struct ggml_tensor* pred_semantics_tensor = ggml_new_tensor_1d(ctx_vits, GGML_TYPE_I32, pred_semantics.size());

    struct ggml_tensor* target_bert_out_gpu = ggml_new_tensor_2d(ctx_vits, GGML_TYPE_F32, 1024, bert_out_len);

    // ge tensor: [512, 1] speaker embedding from cached prompt

    const int ge_size = (int)cached_prompt.speaker_embedding.size();

    struct ggml_tensor* ge_tensor = ggml_new_tensor_2d(ctx_vits, GGML_TYPE_F32, ge_size > 0 ? ge_size : 512, 1);

    

    ggml_backend_buffer_t input_buffer = ggml_backend_alloc_ctx_tensors(ctx_vits, impl->vits_target_backend);

    if (input_buffer) {

        ggml_backend_tensor_set(target_phone_tensor, target_phone_ids.data(), 0, target_phone_ids.size() * sizeof(int32_t));

        

        std::vector<int32_t> dummy_word2ph(1, 1);

        const int32_t* word2ph_ptr = is_overridden ? dummy_word2ph.data() : target_res.word2ph.data();

        ggml_backend_tensor_set(target_word2ph_tensor, word2ph_ptr, 0, word2ph_size * sizeof(int32_t));

        

        ggml_backend_tensor_set(pred_semantics_tensor, pred_semantics.data(), 0, pred_semantics.size() * sizeof(int32_t));

        

        if (is_overridden || target_bert_out) {

            ggml_backend_tensor_set(target_bert_out_gpu, fused_bert_aligned.data() + 1024 * prompt_len, 0, target_len * 1024 * sizeof(float));

        } else {

            std::vector<float> zero_bert(1024 * bert_out_len, 0.0f);

            ggml_backend_tensor_set(target_bert_out_gpu, zero_bert.data(), 0, ggml_nbytes(target_bert_out_gpu));

        }

        if (ge_size > 0) {

            ggml_backend_tensor_set(ge_tensor, cached_prompt.speaker_embedding.data(), 0, ge_size * sizeof(float));

        } else {

            // fallback: zero embedding

            std::vector<float> zero_ge(512, 0.0f);

            ggml_backend_tensor_set(ge_tensor, zero_ge.data(), 0, 512 * sizeof(float));

        }

    }

    

    std::cout << "[GPT-SoVITS] Step 4: ge_size=" << ge_size << ", calling VITS forward with cached speaker embedding..." << std::endl;

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

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[Profile] VITS Decoder Graph execution took " << (t_vits_end - t_vits_start) / 1000.0 << " ms" << std::endl;

    }

    

    // Convert synthesized tensor to final PCM float array in the resident memory

    int out_samples = (int)ggml_nelements(synth_audio);

    impl->last_synthesized_audio.resize(out_samples);

    ggml_backend_tensor_get(synth_audio, impl->last_synthesized_audio.data(), 0, out_samples * sizeof(float));

    

    // Cleanup VITS contexts and buffers (impl->vits_galloc is persistent, so DO NOT free it here)

    ggml_free(ctx_vits);

    

    if (input_buffer) {

        ggml_backend_buffer_free(input_buffer);

    }

    ggml_free(ctx_graph);

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS] Synthesized " << impl->last_synthesized_audio.size() << " samples successfully!\n";

    }

    

    if (out_num_samples) *out_num_samples = out_samples;

    impl->offload_model(1);

    impl->offload_model(2);

    impl->offload_model(3);

    return impl->last_synthesized_audio.data();

}





static std::vector<std::u32string> split_by_sentence_ends(const std::u32string& inp) {

    static const std::unordered_set<char32_t> sentence_ends = {

        U'。', U'？', U'！', U'.', U'?', U'!'

    };

    std::vector<std::u32string> res;

    std::u32string current = U"";

    for (size_t i = 0; i < inp.size(); ++i) {

        char32_t cp = inp[i];

        current += cp;

        if (sentence_ends.find(cp) != sentence_ends.end()) {

            if (!is_decimal_point(inp, i)) {

                res.push_back(current);

                current = U"";

            }

        }

    }

    if (!current.empty()) {

        res.push_back(current);

    }

    return res;

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

    

    std::u32string u32_text = phonemizer::utf8_to_utf32(text);

    u32_text = clean_formatted_decimals(u32_text);

    std::vector<std::u32string> u32_segments;

    if (split_method == "cut0") {

        u32_segments = cut0(u32_text);

    } else if (split_method == "cut1") {

        u32_segments = cut1(u32_text);

    } else if (split_method == "cut2") {

        u32_segments = cut2(u32_text);

    } else if (split_method == "cut3") {

        u32_segments = cut3(u32_text);

    } else if (split_method == "cut4") {

        u32_segments = cut4(u32_text);

    } else if (split_method == "cut5") {

        u32_segments = cut5(u32_text);

    }

    

    // Post-process: split segments further by sentence-ending punctuation to prevent T2S early EOS aborts

    std::vector<std::u32string> final_segments;

    for (const auto& seg : u32_segments) {

        std::vector<std::u32string> sub_segs = split_by_sentence_ends(seg);

        for (const auto& sub : sub_segs) {

            if (!is_subset_of_punctuation(sub)) {

                final_segments.push_back(sub);

            }

        }

    }

    u32_segments = final_segments;

    

    if (u32_segments.empty()) {

        std::cerr << "[GPT-SoVITS] Warning: No segments parsed for synthesis.\n";

        impl->bypass_offload = false;

        impl->offload_model(1);

        impl->offload_model(2);

        impl->offload_model(3);

        return nullptr;

    }

    

    if (u32_segments.size() == 1) {

        std::string seg_utf8 = phonemizer::utf32_to_utf8(u32_segments[0]);

        const float* res = gpt_sovits_synthesize_single_segment_with_cache(

            engine,

            seg_utf8.c_str(),

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

    

    std::cout << "[GPT-SoVITS Split] Splitting paragraph into " << u32_segments.size() 

              << " segments using method '" << split_method << "'...\n";

              

    std::vector<float> combined_audio;

    const size_t pause_samples = 9600;

    

    for (size_t idx = 0; idx < u32_segments.size(); ++idx) {

        std::string seg_utf8 = phonemizer::utf32_to_utf8(u32_segments[idx]);

        std::cout << "[GPT-SoVITS Split] Synthesizing segment [" << (idx + 1) << "/" << u32_segments.size()

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

            if (idx + 1 < u32_segments.size()) {

                combined_audio.insert(combined_audio.end(), pause_samples, 0.0f);

            }

        } else {

            std::cerr << "[GPT-SoVITS Split] Warning: Segment [" << (idx + 1) << "] synthesized empty or failed.\n";

        }

    }

    

    impl->last_synthesized_audio = combined_audio;

    if (out_num_samples) *out_num_samples = (int)impl->last_synthesized_audio.size();

    

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS Split] Successfully concatenated " << u32_segments.size() 

                  << " segments. Total samples: " << impl->last_synthesized_audio.size() << "\n";

    }

              

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



    bool vits_align_mode = (std::getenv("VITS_ALIGNMENT") != nullptr);

    if (vits_align_mode) {

        if (impl->vits->debug_conv_pre) ggml_build_forward_expand(graph, impl->vits->debug_conv_pre);

        if (impl->vits->debug_cond) ggml_build_forward_expand(graph, impl->vits->debug_cond);

        for (int i = 0; i < 5; ++i) {

            if (impl->vits->debug_ups[i]) ggml_build_forward_expand(graph, impl->vits->debug_ups[i]);

        }

        for (int i = 0; i < 15; ++i) {

            if (impl->vits->debug_resblocks[i]) ggml_build_forward_expand(graph, impl->vits->debug_resblocks[i]);

        }

        for (int i = 0; i < 3; ++i) {

            if (impl->vits->debug_res0_convs1[i]) ggml_build_forward_expand(graph, impl->vits->debug_res0_convs1[i]);

            if (impl->vits->debug_res0_convs2[i]) ggml_build_forward_expand(graph, impl->vits->debug_res0_convs2[i]);

        }

        if (impl->vits->debug_conv_post) ggml_build_forward_expand(graph, impl->vits->debug_conv_post);

    }



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



    if (vits_align_mode) {

        auto dump_tensor_cpp = [&](struct ggml_tensor* t, const std::string& name) {

            if (!t) return;

            int64_t nelems = ggml_nelements(t);

            std::vector<float> data(nelems);

            ggml_backend_tensor_get(t, data.data(), 0, nelems * sizeof(float));

            std::ofstream file("scratch/vits_alignment_cpp_act_" + name + ".f32", std::ios::binary);

            if (file.is_open()) {

                file.write(reinterpret_cast<const char*>(data.data()), nelems * sizeof(float));

                std::cout << "[VITS Align CPP] Dumped " << name << " shape: [";

                for (int d = 0; d < 4; ++d) {

                    std::cout << t->ne[d] << (d == 3 ? "" : ", ");

                }

                std::cout << "]\n";

            }

        };



        dump_tensor_cpp(impl->vits->debug_conv_pre, "conv_pre");

        dump_tensor_cpp(impl->vits->debug_cond, "cond");

        for (int i = 0; i < 5; ++i) {

            dump_tensor_cpp(impl->vits->debug_ups[i], "ups_" + std::to_string(i));

        }

        for (int i = 0; i < 15; ++i) {

            dump_tensor_cpp(impl->vits->debug_resblocks[i], "resblock_" + std::to_string(i));

        }

        for (int i = 0; i < 3; ++i) {

            dump_tensor_cpp(impl->vits->debug_res0_convs1[i], "res0_convs1_" + std::to_string(i));

            dump_tensor_cpp(impl->vits->debug_res0_convs2[i], "res0_convs2_" + std::to_string(i));

        }

        dump_tensor_cpp(impl->vits->debug_conv_post, "conv_post");

    }



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

    bool dump_enc = true; // (std::getenv("ENC_ALIGNMENT") != nullptr);



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

    if (dump_enc) {

        auto d = [&](struct ggml_tensor* t, const char* nm) {

            if (!t) return;

            int64_t n = ggml_nelements(t);

            std::vector<float> buf(n);

            ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));

            std::string p = std::string("scratch/enc_cpp_") + nm + ".f32";

            std::ofstream f(p, std::ios::binary);

            f.write((const char*)buf.data(), n * sizeof(float));

            std::cout << "[EncDump] " << nm << " [" << t->ne[0] << "," << t->ne[1] << "," << t->ne[2] << "," << t->ne[3] << "]\n";

        };

        d(impl->vits->debug_enc_ssl_out, "encoder_ssl_out");

        // Also dump the first encoder layer output if we add the debug field

        if (impl->vits->debug_enc_m_p) {

            // Dump ssl_proj input (first thing in forward, before encoder)

        }

        d(impl->vits->debug_enc_text_out, "encoder_text_out");

        d(impl->vits->debug_enc_mrte_out, "mrte_out");

        d(impl->vits->debug_enc_enc2_out, "encoder2_out");

        d(impl->vits->debug_enc_m_p, "m_p");

        d(impl->vits->debug_ssl_proj, "ssl_proj");

        d(impl->vits->debug_decoded, "vq_decoded");

        d(impl->vits->debug_interp, "vq_interp");

        d(impl->vits->debug_enc_q, "encoder_q");

        d(impl->vits->debug_enc_attn, "encoder_attn");

        d(impl->vits->debug_enc_q_cont, "encoder_q_cont");

        d(impl->vits->debug_enc_fa_raw, "encoder_fa_raw");

        d(impl->vits->debug_enc_scores, "encoder_scores");

        d(impl->vits->debug_enc_attn_w, "encoder_attn_w");

        d(impl->vits->debug_enc_vt, "encoder_vt");

        d(impl->vits->debug_enc_out_raw, "encoder_out_raw");

        d(impl->vits->debug_enc_z, "encoder_z");

        d(impl->vits->debug_ref_enc_spectral_0, "mrte_c_pre");

        d(impl->vits->debug_ref_enc_spectral_3, "mrte_text_pre");

        d(impl->vits->debug_enc_q_cont, "mrte_q");

        d(impl->vits->debug_enc_fa_raw, "mrte_k");

        d(impl->vits->debug_ref_enc_pre_attn, "mrte_residual");

        d(impl->vits->debug_ref_enc_temporal_0, "mrte_cross_proj");

        d(impl->vits->debug_ref_enc_spectral_0, "flow_l6");

        d(impl->vits->debug_ref_enc_spectral_3, "flow_l4");

        d(impl->vits->debug_ref_enc_temporal_1, "flow_l2");

        d(impl->vits->debug_ref_enc_spectral_0, "flow_l6");

        d(impl->vits->debug_ref_enc_spectral_3, "flow_l4");

        // Coupling Layer 0 internals

        d(impl->vits->debug_ref_enc_temporal_0, "flow0_x0");

        d(impl->vits->debug_ref_enc_pre_attn, "flow0_pre");

        d(impl->vits->debug_ref_enc_post_attn, "flow0_wn");

        d(impl->vits->debug_enc_vt, "flow0_mean");

    }



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

    bool ref_enc_align = (std::getenv("REF_ENC_ALIGNMENT") != nullptr);

    if (ref_enc_align) {

        auto dump_tensor = [&](struct ggml_tensor* t, const std::string& name) {

            if (!t) return;

            int64_t nelems = ggml_nelements(t);

            std::vector<float> data(nelems);

            ggml_backend_tensor_get(t, data.data(), 0, nelems * sizeof(float));

            std::string path = "scratch/ref_enc_cpp_" + name + ".f32";

            std::ofstream file(path, std::ios::binary);

            file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));

            if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[RefEnc CPP] Dumped " << name << " shape: [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3] << "]\n";

        };



        dump_tensor(impl->vits->debug_ref_enc_spectral_0, "spectral_0");

        dump_tensor(impl->vits->debug_ref_enc_spectral_3, "spectral_3");

        dump_tensor(impl->vits->debug_ref_enc_temporal_0, "temporal_0");

        dump_tensor(impl->vits->debug_ref_enc_temporal_1, "temporal_1");

        dump_tensor(impl->vits->debug_ref_enc_pre_attn, "pre_attn");

        dump_tensor(impl->vits->debug_ref_enc_post_attn, "post_attn");

        dump_tensor(impl->vits->debug_ref_enc_post_fc, "post_fc");

        dump_tensor(impl->vits->debug_ref_enc_pre_pool, "pre_pool");

        dump_tensor(ge, "ge");

    }



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

    if (GPT_SOVITS_DEBUG_ENABLED()) {

        std::cout << "[GPT-SoVITS Config] Configured model slot " << model_type

                  << ": path='" << impl->slots[model_type].path

                  << "', device='" << impl->slots[model_type].device

                  << "', resident=" << is_resident << std::endl;

    }

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

// Emotion entry parsed from config.json
struct EmotionEntry {
    std::string audio;
    std::string text;
};

// Parser for the new emotions-based config.json format:
// {
//   "name": "doubao",
//   "lang": "zh",
//   "emotions": {
//     "安慰": { "audio": "安慰.wav", "text": "别难过啦..." },
//     ...
//   }
// }
static bool voice_manager_parse_emotions_config(
    const std::string& json_str,
    std::string& out_name,
    std::string& out_lang,
    std::unordered_map<std::string, EmotionEntry>& out_emotions
) {
    out_name.clear();
    out_lang.clear();
    out_emotions.clear();

    // Helper: extract a JSON string value (between quotes)
    auto extract_string = [](const std::string& s, size_t& pos) -> std::string {
        size_t start = s.find('"', pos);
        if (start == std::string::npos) return "";
        size_t end = s.find('"', start + 1);
        if (end == std::string::npos) return "";
        pos = end + 1;
        return s.substr(start + 1, end - start - 1);
    };

    // Helper: skip whitespace
    auto skip_ws = [](const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) pos++;
    };

    size_t pos = 0;
    skip_ws(json_str, pos);
    if (pos >= json_str.size() || json_str[pos] != '{') return false;
    pos++; // skip opening {

    for (int depth = 1; depth > 0 && pos < json_str.size(); ) {
        skip_ws(json_str, pos);
        if (pos >= json_str.size()) break;

        if (json_str[pos] == '}') { depth--; pos++; continue; }
        if (json_str[pos] == ',') { pos++; continue; }

        std::string key = extract_string(json_str, pos);
        if (key.empty()) break;

        skip_ws(json_str, pos);
        if (pos >= json_str.size() || json_str[pos] != ':') break;
        pos++; // skip :
        skip_ws(json_str, pos);

        if (key == "name" || key == "lang") {
            std::string val = extract_string(json_str, pos);
            if (key == "name") out_name = val;
            else out_lang = val;
        } else if (key == "emotions") {
            // Parse the emotions object
            if (pos >= json_str.size() || json_str[pos] != '{') break;
            pos++; // skip {
            while (pos < json_str.size()) {
                skip_ws(json_str, pos);
                if (pos >= json_str.size()) break;
                if (json_str[pos] == '}') { pos++; break; }
                if (json_str[pos] == ',') { pos++; continue; }

                std::string emo_name = extract_string(json_str, pos);
                if (emo_name.empty()) break;

                skip_ws(json_str, pos);
                if (pos >= json_str.size() || json_str[pos] != ':') break;
                pos++;
                skip_ws(json_str, pos);
                if (pos >= json_str.size() || json_str[pos] != '{') break;
                pos++; // skip inner {

                EmotionEntry entry;
                while (pos < json_str.size()) {
                    skip_ws(json_str, pos);
                    if (pos >= json_str.size()) break;
                    if (json_str[pos] == '}') { pos++; break; }
                    if (json_str[pos] == ',') { pos++; continue; }

                    std::string inner_key = extract_string(json_str, pos);
                    if (inner_key.empty()) break;
                    skip_ws(json_str, pos);
                    if (pos >= json_str.size() || json_str[pos] != ':') break;
                    pos++;
                    skip_ws(json_str, pos);
                    std::string inner_val = extract_string(json_str, pos);

                    if (inner_key == "audio") entry.audio = inner_val;
                    else if (inner_key == "text") entry.text = inner_val;
                }
                if (!entry.audio.empty() && !entry.text.empty()) {
                    out_emotions[emo_name] = entry;
                }
            }
        }
    }

    return !out_name.empty() && !out_lang.empty() && !out_emotions.empty();
}

// Helper to load a 16-bit PCM Mono WAV file into float vector
static std::vector<float> voice_manager_load_wav_file(const std::string& filename, int& sample_rate) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[VoiceManager WAV Loader] Failed to open WAV file: " << filename << "\n";
        return {};
    }

    char chunk_id[4];
    file.read(chunk_id, 4);
    if (std::strncmp(chunk_id, "RIFF", 4) != 0) {
        std::cerr << "[VoiceManager WAV Loader] Invalid RIFF header\n";
        return {};
    }

    file.seekg(8, std::ios::beg);
    char format_id[4];
    file.read(format_id, 4);
    if (std::strncmp(format_id, "WAVE", 4) != 0) {
        std::cerr << "[VoiceManager WAV Loader] Not a WAVE file\n";
        return {};
    }

    short num_channels = 0;
    int s_rate = 0;
    short bits_per_sample = 0;
    int data_size = 0;

    while (file) {
        char subchunk_id[4];
        file.read(subchunk_id, 4);
        if (!file) break;

        int subchunk_size = 0;
        file.read(reinterpret_cast<char*>(&subchunk_size), 4);
        if (!file) break;

        if (std::strncmp(subchunk_id, "fmt ", 4) == 0) {
            short audio_format = 0;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&s_rate), 4);
            file.seekg(6, std::ios::cur);
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (subchunk_size > 16) {
                file.seekg(subchunk_size - 16, std::ios::cur);
            }
        } else if (std::strncmp(subchunk_id, "data", 4) == 0) {
            data_size = subchunk_size;
            std::vector<float> audio_data;
            if (bits_per_sample == 16) {
                int num_samples = data_size / 2;
                std::vector<short> raw_samples(num_samples);
                file.read(reinterpret_cast<char*>(raw_samples.data()), data_size);
                audio_data.resize(num_samples);
                for (int i = 0; i < num_samples; ++i) {
                    audio_data[i] = raw_samples[i] / 32768.0f;
                }
            } else if (bits_per_sample == 32) {
                int num_samples = data_size / 4;
                audio_data.resize(num_samples);
                file.read(reinterpret_cast<char*>(audio_data.data()), data_size);
            } else {
                std::cerr << "[VoiceManager WAV Loader] Unsupported bits per sample: " << bits_per_sample << "\n";
                return {};
            }
            
            if (s_rate == 32000) {
                std::cout << "[VoiceManager WAV Loader] Downsampling 32000 Hz reference to 16000 Hz...\n";
                std::vector<float> downsampled;
                downsampled.reserve(audio_data.size() / 2);
                for (size_t i = 0; i < audio_data.size(); i += 2) {
                    downsampled.push_back(audio_data[i]);
                }
                audio_data = std::move(downsampled);
                s_rate = 16000;
            } else if (s_rate == 48000) {
                std::cout << "[VoiceManager WAV Loader] Downsampling 48000 Hz reference to 16000 Hz...\n";
                std::vector<float> downsampled;
                downsampled.reserve(audio_data.size() / 3);
                for (size_t i = 0; i < audio_data.size(); i += 3) {
                    downsampled.push_back(audio_data[i]);
                }
                audio_data = std::move(downsampled);
                s_rate = 16000;
            }
            
            sample_rate = s_rate;
            std::cout << "[VoiceManager WAV Loader] Loaded " << filename << " | Samples: " << audio_data.size() << "\n";
            return audio_data;
        } else {
            file.seekg(subchunk_size, std::ios::cur);
        }
    }
    return {};
}

static bool serialize_features(const std::string& filepath, const PromptCache& cache) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for writing: " << filepath << std::endl;
        return false;
    }
    
    uint32_t magic = 0x47535646;
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    uint32_t text_len = static_cast<uint32_t>(cache.prompt_text.size());
    out.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
    out.write(cache.prompt_text.data(), text_len);
    
    uint32_t lang_len = static_cast<uint32_t>(cache.prompt_lang.size());
    out.write(reinterpret_cast<const char*>(&lang_len), sizeof(lang_len));
    out.write(cache.prompt_lang.data(), lang_len);
    
    uint32_t phone_count = static_cast<uint32_t>(cache.prompt_phones.size());
    out.write(reinterpret_cast<const char*>(&phone_count), sizeof(phone_count));
    for (const auto& phone : cache.prompt_phones) {
        uint32_t len = static_cast<uint32_t>(phone.size());
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(phone.data(), len);
    }
    
    uint32_t word2ph_count = static_cast<uint32_t>(cache.prompt_word2ph.size());
    out.write(reinterpret_cast<const char*>(&word2ph_count), sizeof(word2ph_count));
    out.write(reinterpret_cast<const char*>(cache.prompt_word2ph.data()), word2ph_count * sizeof(int));
    
    uint32_t hubert_codes_count = static_cast<uint32_t>(cache.hubert_codes.size());
    out.write(reinterpret_cast<const char*>(&hubert_codes_count), sizeof(hubert_codes_count));
    out.write(reinterpret_cast<const char*>(cache.hubert_codes.data()), hubert_codes_count * sizeof(int32_t));
    
    uint32_t bert_features_count = static_cast<uint32_t>(cache.bert_features.size());
    out.write(reinterpret_cast<const char*>(&bert_features_count), sizeof(bert_features_count));
    out.write(reinterpret_cast<const char*>(cache.bert_features.data()), bert_features_count * sizeof(float));
    
    uint32_t speaker_embedding_count = static_cast<uint32_t>(cache.speaker_embedding.size());
    out.write(reinterpret_cast<const char*>(&speaker_embedding_count), sizeof(speaker_embedding_count));
    out.write(reinterpret_cast<const char*>(cache.speaker_embedding.data()), speaker_embedding_count * sizeof(float));
    
    return out.good();
}

static bool deserialize_features(const std::string& filepath, PromptCache& cache) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[VoiceManager] Failed to open file for reading: " << filepath << std::endl;
        return false;
    }
    
    uint32_t magic = 0;
    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    if (magic != 0x47535646 || version != 1) {
        std::cerr << "[VoiceManager] Invalid magic or version in features file: " << filepath << std::endl;
        return false;
    }
    
    uint32_t text_len = 0;
    in.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
    cache.prompt_text.resize(text_len);
    if (text_len > 0) {
        in.read(&cache.prompt_text[0], text_len);
    }
    
    uint32_t lang_len = 0;
    in.read(reinterpret_cast<char*>(&lang_len), sizeof(lang_len));
    cache.prompt_lang.resize(lang_len);
    if (lang_len > 0) {
        in.read(&cache.prompt_lang[0], lang_len);
    }
    
    uint32_t phone_count = 0;
    in.read(reinterpret_cast<char*>(&phone_count), sizeof(phone_count));
    cache.prompt_phones.resize(phone_count);
    for (uint32_t i = 0; i < phone_count; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        cache.prompt_phones[i].resize(len);
        if (len > 0) {
            in.read(&cache.prompt_phones[i][0], len);
        }
    }
    
    uint32_t word2ph_count = 0;
    in.read(reinterpret_cast<char*>(&word2ph_count), sizeof(word2ph_count));
    cache.prompt_word2ph.resize(word2ph_count);
    if (word2ph_count > 0) {
        in.read(reinterpret_cast<char*>(cache.prompt_word2ph.data()), word2ph_count * sizeof(int));
    }
    
    uint32_t hubert_codes_count = 0;
    in.read(reinterpret_cast<char*>(&hubert_codes_count), sizeof(hubert_codes_count));
    cache.hubert_codes.resize(hubert_codes_count);
    if (hubert_codes_count > 0) {
        in.read(reinterpret_cast<char*>(cache.hubert_codes.data()), hubert_codes_count * sizeof(int32_t));
    }
    
    uint32_t bert_features_count = 0;
    in.read(reinterpret_cast<char*>(&bert_features_count), sizeof(bert_features_count));
    cache.bert_features.resize(bert_features_count);
    if (bert_features_count > 0) {
        in.read(reinterpret_cast<char*>(cache.bert_features.data()), bert_features_count * sizeof(float));
    }
    
    uint32_t speaker_embedding_count = 0;
    in.read(reinterpret_cast<char*>(&speaker_embedding_count), sizeof(speaker_embedding_count));
    cache.speaker_embedding.resize(speaker_embedding_count);
    if (speaker_embedding_count > 0) {
        in.read(reinterpret_cast<char*>(cache.speaker_embedding.data()), speaker_embedding_count * sizeof(float));
    }
    
    return in.good();
}

struct VoiceManagerImpl {
    gpt_sovits_engine_t engine;
    // Map: "doubao/安慰" -> "voice_doubao_安慰" (cache_id)
    std::unordered_map<std::string, std::string> emo_to_cache_id;
    // Map: "doubao" -> default emotion name
    std::unordered_map<std::string, std::string> char_default_emotion;

    VoiceManagerImpl(gpt_sovits_engine_t eng) : engine(eng) {}
};

gpt_sovits_voice_manager_t gpt_sovits_voice_manager_init(gpt_sovits_engine_t engine) {
    if (!engine) return nullptr;
    return new VoiceManagerImpl(engine);
}

void gpt_sovits_voice_manager_free(gpt_sovits_voice_manager_t manager) {
    if (manager) {
        delete (VoiceManagerImpl*)manager;
    }
}

bool gpt_sovits_voice_manager_register_character(
    gpt_sovits_voice_manager_t manager,
    const char* voices_root_dir,
    const char* character_id
) {
    if (!manager || !voices_root_dir || !character_id) return false;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;
    Impl* impl = (Impl*)mgr->engine;
    if (!impl) return false;

    std::string cid(character_id);
    std::filesystem::path root_dir(voices_root_dir);
    std::filesystem::path config_file = root_dir / "config.json";
    std::filesystem::path audios_dir = root_dir / "audios";
    std::filesystem::path features_dir = root_dir / "features";

    if (!std::filesystem::exists(config_file)) {
        std::cerr << "[VoiceManager] Config file not found: " << config_file.string() << std::endl;
        return false;
    }

    std::ifstream config_in(config_file.string());
    if (!config_in.is_open()) {
        std::cerr << "[VoiceManager] Failed to open config file: " << config_file.string() << std::endl;
        return false;
    }

    std::string config_content((std::istreambuf_iterator<char>(config_in)), std::istreambuf_iterator<char>());

    std::string char_name;
    std::string char_lang;
    std::unordered_map<std::string, EmotionEntry> emotions;
    if (!voice_manager_parse_emotions_config(config_content, char_name, char_lang, emotions)) {
        std::cerr << "[VoiceManager] Failed to parse config.json (expecting 'name', 'lang', 'emotions' keys)" << std::endl;
        return false;
    }

    std::cout << "[VoiceManager] Registering character '" << cid << "' (config name='" << char_name << "') with " << emotions.size() << " emotions" << std::endl;

    bool any_ok = false;
    std::string default_emotion;

    for (const auto& [emo_name, entry] : emotions) {
        std::string cache_id = "voice_" + char_name + "_" + emo_name;
        std::string emo_key = cid + "/" + emo_name;
        mgr->emo_to_cache_id[emo_key] = cache_id;

        if (default_emotion.empty()) {
            default_emotion = emo_name;
            mgr->char_default_emotion[cid] = emo_name;
        }

        // Check for precomputed features
        std::filesystem::path features_file = features_dir / std::filesystem::u8path(emo_name + ".features.bin");
        if (std::filesystem::exists(features_file)) {
            std::cout << "[VoiceManager] Loading cached features for emotion '" << emo_name << "'..." << std::endl;
            PromptCache cache;
            if (deserialize_features(features_file.string(), cache)) {
                impl->prompt_caches[cache_id] = cache;
                std::cout << "[VoiceManager]   OK (from cache)" << std::endl;
                any_ok = true;
                continue;
            } else {
                std::cerr << "[VoiceManager]   Cache corrupt, re-extracting..." << std::endl;
            }
        }

        // Load audio
        std::filesystem::path audio_path = audios_dir / std::filesystem::u8path(entry.audio);
        if (!std::filesystem::exists(audio_path)) {
            std::cerr << "[VoiceManager] Audio not found for emotion '" << emo_name << "': " << audio_path.string() << std::endl;
            continue;
        }

        int sample_rate = 0;
        std::vector<float> audio_data = voice_manager_load_wav_file(audio_path.string(), sample_rate);
        if (audio_data.empty()) {
            std::cerr << "[VoiceManager] Failed to load audio for emotion '" << emo_name << "'" << std::endl;
            continue;
        }

        std::cout << "[VoiceManager] Extracting features for emotion '" << emo_name << "'..." << std::endl;
        gpt_sovits_get_or_create_prompt_cache(
            mgr->engine,
            cache_id.c_str(),
            audio_data.data(),
            audio_data.size(),
            entry.text.c_str(),
            char_lang.c_str()
        );

        auto it = impl->prompt_caches.find(cache_id);
        if (it != impl->prompt_caches.end()) {
            std::filesystem::create_directories(features_dir);
            if (serialize_features(features_file.string(), it->second)) {
                std::cout << "[VoiceManager]   Cached -> " << features_file.string() << std::endl;
                any_ok = true;
            } else {
                std::cerr << "[VoiceManager]   Failed to save cache for '" << emo_name << "'" << std::endl;
            }
        } else {
            std::cerr << "[VoiceManager]   Feature extraction failed for '" << emo_name << "'" << std::endl;
        }
    }

    return any_ok;
}

const char* gpt_sovits_voice_manager_get_cache_id(
    gpt_sovits_voice_manager_t manager,
    const char* character_id
) {
    if (!manager || !character_id) return nullptr;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;

    std::string cid(character_id);

    // Direct lookup: "doubao/安慰" -> exact emotion match
    auto it = mgr->emo_to_cache_id.find(cid);
    if (it != mgr->emo_to_cache_id.end()) {
        return it->second.c_str();
    }

    // Fallback: "doubao" -> default emotion
    auto def_it = mgr->char_default_emotion.find(cid);
    if (def_it != mgr->char_default_emotion.end()) {
        std::string emo_key = cid + "/" + def_it->second;
        auto emo_it = mgr->emo_to_cache_id.find(emo_key);
        if (emo_it != mgr->emo_to_cache_id.end()) {
            return emo_it->second.c_str();
        }
    }

    return nullptr;
}

const float* gpt_sovits_voice_manager_synthesize(
    gpt_sovits_voice_manager_t manager,
    const char* character_id,
    const char* text,
    const char* language,
    float speed,
    int* out_num_samples
) {
    if (!manager || !character_id || !text || !language) return nullptr;
    VoiceManagerImpl* mgr = (VoiceManagerImpl*)manager;
    const char* cache_id = gpt_sovits_voice_manager_get_cache_id(manager, character_id);
    if (!cache_id) {
        std::cerr << "[VoiceManager] Character not registered: " << character_id << std::endl;
        return nullptr;
    }
    return gpt_sovits_synthesize_with_cache(mgr->engine, text, language, cache_id, speed, out_num_samples);
}

} // extern "C"


