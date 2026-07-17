#include "models/models.h"
#include "phonemizer.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>

static std::string utf16_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
}
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <string>
#include <limits>

namespace {

static std::unordered_map<std::string, int32_t> bert_vocab;

static void load_bert_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[BERT Align] Failed to open BERT vocabulary: " << path << "\n";
        return;
    }
    std::string line;
    int32_t idx = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        bert_vocab[line] = idx++;
    }
    std::cout << "[BERT Align] Loaded " << bert_vocab.size() << " BERT vocab items\n";
}

static std::vector<int32_t> bert_tokenize(const std::string& text) {
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

static void write_f32_binary(const std::string & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[BERT Align] Failed to open binary output: " << path << "\n";
        return;
    }
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize)(data.size() * sizeof(float)));
}

static void write_summary(
    const std::string & path,
    const std::vector<int32_t> & ids,
    const std::vector<float> & bert_content,
    int64_t channels,
    int64_t seq_len) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "[BERT Align] Failed to open summary output: " << path << "\n";
        return;
    }

    float bert_min = std::numeric_limits<float>::max();
    float bert_max = std::numeric_limits<float>::lowest();
    double bert_sum = 0.0;
    for (float v : bert_content) {
        bert_min = std::min(bert_min, v);
        bert_max = std::max(bert_max, v);
        bert_sum += v;
    }

    file << "input_ids_len=" << ids.size() << "\n";
    file << "input_ids=";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) file << ",";
        file << ids[i];
    }
    file << "\n";

    file << "bert_content_shape=" << channels << "x" << seq_len << "\n";
    file << "bert_content_min=" << bert_min << "\n";
    file << "bert_content_max=" << bert_max << "\n";
    file << "bert_content_sum=" << bert_sum << "\n";
    file << "bert_content_ch0_first10=";
    for (int64_t i = 0; i < std::min<int64_t>(10, seq_len); ++i) {
        if (i) file << ",";
        file << bert_content[i * channels];
    }
    file << "\n";
}

static ggml_backend_t pick_backend(bool use_gpu) {
    ggml_backend_load_all();
    if (!use_gpu) {
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    }

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        const std::string name = ggml_backend_dev_name(dev);
        if (name.rfind("CUDA", 0) == 0 || name.rfind("SYCL", 0) == 0) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                return backend;
            }
        }
    }
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> args;
#ifdef _WIN32
    int wargc;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (wargv) {
        for (int i = 0; i < wargc; ++i) {
            args.push_back(utf16_to_utf8(wargv[i]));
        }
        LocalFree(wargv);
    }
#else
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
#endif

    std::string bert_path = "models/gpt_sovits/weights/bert/bert_f16.gguf";
    std::string dict_dir = "models/gpt_sovits/dict";
    std::string text = "欢迎来到营火，无火的余灰。";
    std::string out_prefix = "scratch/bert_alignment_cpp";
    bool use_gpu = true;

    int args_size = (int)args.size();
    for (int i = 1; i < args_size; ++i) {
        const std::string arg = args[i];
        if (arg == "--bert" && i + 1 < args_size) {
            bert_path = args[++i];
        } else if (arg == "--dict" && i + 1 < args_size) {
            dict_dir = args[++i];
        } else if (arg == "--text" && i + 1 < args_size) {
            text = args[++i];
        } else if (arg == "--out-prefix" && i + 1 < args_size) {
            out_prefix = args[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        }
    }

    ggml_backend_t backend = pick_backend(use_gpu);
    if (!backend) {
        std::cerr << "[BERT Align] Failed to initialize backend\n";
        return 1;
    }

    gpt_sovits::BertModel model;
    if (!model.load(bert_path, backend)) {
        std::cerr << "[BERT Align] Failed to load model: " << bert_path << "\n";
        ggml_backend_free(backend);
        return 1;
    }

    load_bert_vocab(dict_dir + "/bert_vocab.txt");
    std::vector<int32_t> ids = bert_tokenize(text);
    if (ids.empty()) {
        std::cerr << "[BERT Align] Tokenization returned empty ids!\n";
        ggml_backend_free(backend);
        return 1;
    }

    ggml_init_params init_params = {
        /* .mem_size   = */ 64 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(init_params);
    ggml_tensor * bert_content = model.forward(ctx, ids, backend);
    if (!bert_content) {
        std::cerr << "[BERT Align] BERT forward failed\n";
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const int64_t channels = bert_content->ne[0];
    const int64_t seq_len = bert_content->ne[1];
    std::vector<float> bert_content_host((size_t)(channels * seq_len));
    std::memcpy(bert_content_host.data(), bert_content->data, bert_content_host.size() * sizeof(float));

    write_f32_binary(out_prefix + "_features.f32", bert_content_host);
    write_summary(out_prefix + "_summary.txt", ids, bert_content_host, channels, seq_len);

    std::cout << "[BERT Align] Wrote:\n"
              << "  " << out_prefix << "_features.f32\n"
              << "  " << out_prefix << "_summary.txt\n";

    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}
