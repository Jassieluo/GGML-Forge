#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace gpt_sovits {

// Compatibility contract between GPT-SoVITS component artifacts.

struct ModelProfile {
    std::string exact_version = "unknown";
    std::string profile_id;
    std::string vocoder_architecture;

    int vits_version = 0;
    int symbol_version = 0;
    int expected_t2s_family = 0;
    int semantic_frame_stride = 0;
    int default_inference_steps = 0;
    int max_prompt_frames = 0;
    float feature_rate_scale = 0.0f;

    bool is_classic = false;
    bool uses_cfm = false;
    bool requires_sv_emb = false;

    int ge_dim = 512;
    int sv_emb_dim = 0;
    int ref_enc_channels = 704;

    int output_sampling_rate = 0;
    int reference_sampling_rate = 0;
    int prompt_mel_sampling_rate = 0;
    int filter_length = 2048;
    int hop_length = 640;
    int win_length = 2048;
    int prompt_mel_channels = 0;

    int upsample_initial_channel = 0;
    std::vector<int> upsample_rates;
    std::vector<int> upsample_kernel_sizes;

    std::string summary() const {
        std::ostringstream oss;
        oss << "version=" << exact_version
            << ", coarse_vits=" << vits_version
            << ", symbols=v" << symbol_version
            << ", t2s_family=v" << expected_t2s_family
            << ", semantic_stride=" << semantic_frame_stride
            << ", backend=" << (uses_cfm ? "cfm" : "classic")
            << ", ge_dim=" << ge_dim
            << ", ref_enc_channels=" << ref_enc_channels
            << ", output_sr=" << output_sampling_rate;
        if (requires_sv_emb) {
            oss << ", sv_emb_dim=" << sv_emb_dim;
        }
        if (upsample_initial_channel > 0) {
            oss << ", upsample_initial_channel=" << upsample_initial_channel;
        }
        if (!upsample_kernel_sizes.empty()) {
            oss << ", upsample_kernels=[";
            for (size_t i = 0; i < upsample_kernel_sizes.size(); ++i) {
                if (i > 0) oss << ",";
                oss << upsample_kernel_sizes[i];
            }
            oss << "]";
        }
        return oss.str();
    }
};

inline std::string canonical_model_version(const std::string& raw_version) {
    if (raw_version == "v1" || raw_version == "v2" || raw_version == "v2Pro" ||
        raw_version == "v2ProPlus" || raw_version == "v3" || raw_version == "v4") {
        return raw_version;
    }
    return {};
}

inline int coarse_version_from_string(const std::string& raw_version) {
    const std::string version = canonical_model_version(raw_version);
    if (version == "v1") return 1;
    if (version == "v3") return 3;
    if (version == "v4") return 4;
    if (version == "v2" || version == "v2Pro" || version == "v2ProPlus") return 2;
    return 0;
}

} // namespace gpt_sovits
