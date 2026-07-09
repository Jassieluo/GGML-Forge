#include "vits.h"
#include "ops/ops.h"
#include <iostream>
#include <cstring>
#include <vector>

namespace gpt_sovits {

struct ggml_tensor* VITSModelClassic::forward_from_latent(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* latent,
    struct ggml_tensor* speaker_embedding,
    ggml_backend_t backend
) {
    clear_conv_1d_params_pool();
    struct ggml_tensor* ge = speaker_embedding;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
    }
    return build_vits_generator(ctx_graph, latent, ge, *this, backend);
}

struct ggml_tensor* VITSModelClassic::forward(
    struct ggml_context* ctx_graph,
    struct ggml_tensor* phone_ids,
    struct ggml_tensor* phone_lengths,
    struct ggml_tensor* word2ph,
    struct ggml_tensor* bert_features,
    struct ggml_tensor* prompt_semantics,
    struct ggml_tensor* refer_audio,
    float speed,
    ggml_backend_t backend
) {
    clear_conv_1d_params_pool();
    (void)phone_lengths;
    (void)word2ph;
    (void)bert_features;
    (void)refer_audio;

    int semantic_len = (int)prompt_semantics->ne[0];
    if (GPT_SOVITS_DEBUG_ENABLED()) {
        std::cout << "[VITS-Classic] Inference Graph - semantic_len: " << semantic_len
                  << ", speed: " << speed << std::endl;
    }

    // Step 1: VQ Decode - semantic token IDs -> continuous features [768, N]
    struct ggml_tensor* decoded = vq_decode(ctx_graph, prompt_semantics, *this);
    if (!decoded) {
        std::cerr << "[VITS-Classic] Error: VQ decode failed!" << std::endl;
        return nullptr;
    }
    debug_decoded = decoded;

    // Step 2: Interpolate from 25Hz to 50Hz (2x nearest-neighbor)
    struct ggml_tensor* interp = interp_nearest_2x(ctx_graph, decoded, *this);
    int T_y = (int)interp->ne[1];
    debug_interp = interp;

    // Step 3: SSL Projection - 768 -> 192 channels via enc_p.ssl_proj
    struct ggml_tensor* ssl_proj_w = get_tensor("enc_p.ssl_proj.weight");
    struct ggml_tensor* ssl_proj_b = get_tensor("enc_p.ssl_proj.bias");
    struct ggml_tensor* y = interp;
    if (ssl_proj_w && ssl_proj_b) {
        y = ggml_conv_1d_with_bias(ctx_graph, interp, ssl_proj_w, ssl_proj_b, 1, 1, 0, backend);
        debug_ssl_proj = y;
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] After ssl_proj shape: [" << y->ne[0] << ", " << y->ne[1] << "]" << std::endl;
    } else {
        std::cerr << "[VITS-Classic] Warning: ssl_proj weights missing, feeding raw VQ features to generator." << std::endl;
    }

    // Step 4: Load speaker embedding (ge)
    struct ggml_tensor* ge = refer_audio;
    if (ge) {
        int64_t ge_size = ggml_nelements(ge);
        ge = ggml_reshape_2d(ctx_graph, ge, ge_size, 1);
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Using passed-in ge tensor: ne0=" << ge->ne[0] << " ne1=" << ge->ne[1] << std::endl;
    } else {
        int64_t ge_dim = 512;
        struct ggml_tensor* prelu_w = get_tensor("prelu.weight");
        if (prelu_w) {
            ge_dim = prelu_w->ne[0];
        }
        ge = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, ge_dim, 1);
        ge = ggml_fill(ctx_graph, ge, 0.0f);
    }

    struct ggml_tensor* ge_512 = ge;
    struct ggml_tensor* ge_to512_w = get_tensor("ge_to512.weight");
    struct ggml_tensor* ge_to512_b = get_tensor("ge_to512.bias");
    if (ge_to512_w && ge_to512_b && ge) {
        ge_512 = ggml_linear(ctx_graph, ge, ge_to512_w, ge_to512_b);
    }
    int n_head = 2;
    int d_k = 96;  // 192 / 2

    // Step 5: encoder_ssl (3 layers) on ssl features
    struct ggml_tensor* y_enc = build_encoder(ctx_graph, y, nullptr, *this, "enc_p.encoder_ssl", 3, n_head, d_k, T_y, backend);
    debug_enc_ssl_out = y_enc;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] encoder_ssl done." << std::endl;

    // Step 6: encoder_text (6 layers) on phone embeddings
    struct ggml_tensor* text_emb_w = get_tensor("enc_p.text_embedding.weight");
    int text_len = (int)phone_ids->ne[0];
    if (text_emb_w) {
        if (GPT_SOVITS_DEBUG_ENABLED()) {
            std::cout << "[VITS-Classic Debug] text_emb_w row count: " << text_emb_w->ne[1] << ", col count: " << text_emb_w->ne[0] << std::endl;
        }
        std::vector<int32_t> temp_ids(text_len);
        ggml_backend_tensor_get(phone_ids, temp_ids.data(), 0, text_len * sizeof(int32_t));
        for (int i = 0; i < text_len; ++i) {
            if (temp_ids[i] < 0 || temp_ids[i] >= text_emb_w->ne[1]) {
                std::cerr << "[VITS-Classic Error] phone_id " << temp_ids[i] << " is OUT OF BOUNDS for text_emb_w (0 to " << text_emb_w->ne[1] - 1 << ")!" << std::endl;
            }
        }
    }
    struct ggml_tensor* text_emb = ggml_get_rows(ctx_graph, text_emb_w, phone_ids);  // [192, text_len]
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] text_emb: [" << text_emb->ne[0] << ", " << text_emb->ne[1] << "]" << std::endl;

    struct ggml_tensor* text_enc = build_encoder(ctx_graph, text_emb, nullptr, *this, "enc_p.encoder_text", 6, n_head, d_k, text_len, backend);
    debug_enc_text_out = text_enc;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] encoder_text done." << std::endl;

    // Step 7: MRTE - cross-attention between y_enc and text_enc with speaker conditioning
    struct ggml_tensor* mrte_out = build_mrte(ctx_graph, y_enc, nullptr, text_enc, nullptr, ge_512, *this, backend);
    debug_enc_mrte_out = mrte_out;

    // Step 8: encoder2 (3 layers)
    struct ggml_tensor* y2 = build_encoder(ctx_graph, mrte_out, nullptr, *this, "enc_p.encoder2", 3, n_head, d_k, T_y, backend);
    debug_enc_enc2_out = y2;

    // Step 9: Speed scaling
    if (speed != 1.0f && speed > 0.0f) {
        int target_frames = (int)std::round((float)T_y / speed);
        if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Speed scaling: " << T_y << " -> " << target_frames << " frames" << std::endl;
        struct ggml_tensor* target = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 192, target_frames);
        y2 = ggml_repeat(ctx_graph, y2, target);
    }

    // Step 10: proj - Conv1d(192, 384, 1) -> split to m_p (192) and logs (192)
    struct ggml_tensor* proj_w = get_tensor("enc_p.proj.weight");
    struct ggml_tensor* proj_b = get_tensor("enc_p.proj.bias");
    struct ggml_tensor* stats = ggml_conv_1d_with_bias(ctx_graph, y2, proj_w, proj_b, 1, 1, 0, backend);  // [384, T_y]
    struct ggml_tensor* m_p = ggml_view_2d(ctx_graph, stats, 192, stats->ne[1], stats->nb[1], 0);
    m_p = ggml_cont(ctx_graph, m_p);
    debug_enc_m_p = m_p;

    // Step 11: Flow reverse (ResidualCouplingBlock)
    if (!VITSModel::flip_data_ready) {
        std::memset(VITSModel::flip_data, 0, sizeof(VITSModel::flip_data));
        for (int r = 0; r < 192; ++r) {
            VITSModel::flip_data[r * 192 + (191 - r)] = 1.0f;
        }
        VITSModel::flip_data_ready = true;
    }

    auto flip_ch = [&](struct ggml_tensor* t) -> struct ggml_tensor* {
        struct ggml_tensor* P = ggml_new_tensor_2d(ctx_graph, GGML_TYPE_F32, 192, 192);
        upload_entries.push_back({P, std::vector<uint8_t>((uint8_t*)flip_data, (uint8_t*)(flip_data + 192*192))});
        return ggml_cont(ctx_graph, ggml_mul_mat(ctx_graph, P, t));
    };

    struct ggml_tensor* z = m_p;
    for (int fi : {6, 4, 2, 0}) {
        z = flip_ch(z);
        std::string flow_p = "flow.flows." + std::to_string(fi) + ".";
        z = build_coupling_layer(ctx_graph, z, nullptr, ge, *this, flow_p,
            192, 192, 5, 2, 4, true, backend);
        if (fi == 6) debug_ref_enc_spectral_0 = z;
        if (fi == 4) debug_ref_enc_spectral_3 = z;
        if (fi == 2) debug_ref_enc_temporal_1 = z;
    }
    z = ggml_cont(ctx_graph, z);
    debug_enc_z = z;
    if (GPT_SOVITS_DEBUG_ENABLED()) std::cout << "[VITS-Classic] Flow done. z: [" << z->ne[0] << ", " << z->ne[1] << "]" << std::endl;

    return build_vits_generator(ctx_graph, z, ge, *this, backend);
}

} // namespace gpt_sovits
