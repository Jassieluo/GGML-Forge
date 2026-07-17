#include "ops/ops.h"
#include "ggml-ops-ext-bridge.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

extern "C" {
    __declspec(dllimport) void ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    __declspec(dllimport) void ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    __declspec(dllimport) void ggml_ops_ext_sycl_init();
#endif
}

static bool is_backend(const std::string& name, const char* prefix) {
    return name.rfind(prefix, 0) == 0;
}

int main() {
    ggml_ops_ext_cpu_init();
#ifdef GGML_USE_CUDA
    ggml_ops_ext_cuda_init();
#endif
#ifdef GGML_USE_SYCL
    ggml_ops_ext_sycl_init();
#endif
    ggml_backend_load_all();

    ggml_ops_ext::acquire_ops_hook();
    ggml_ops_ext::acquire_ops_hook();
    if (!g_ggml_bridge_hook || !g_ggml_bridge_supports_hook) {
        std::cerr << "Ops hooks were not installed." << std::endl;
        return 1;
    }
    ggml_ops_ext::release_ops_hook();
    if (!g_ggml_bridge_hook || !g_ggml_bridge_supports_hook) {
        std::cerr << "One runtime released hooks still used by another runtime." << std::endl;
        return 1;
    }

    ggml_init_params init_params = {};
    init_params.mem_size = 1024 * 1024;
    ggml_context* ctx = ggml_init(init_params);
    ggml_tensor* weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, 32, 3, 32);
    ggml_tensor* unsupported_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_Q5_0, 32, 3, 32);
    ggml_tensor* input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 32, 1);
    ggml_tensor* srcs[] = { weight, input };
    int32_t params[] = { 1, 0, 1, 1 };
    ggml_tensor* valid_conv_output = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 32, 1);
    ggml_tensor* invalid_conv_output = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 6, 32, 1);
    ggml_tensor* conv_t_weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 32);
    ggml_tensor* conv_t_srcs[] = { conv_t_weight, input };
    ggml_tensor* conv_t_input_f16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 8, 32, 1);
    ggml_tensor* conv_t_f16_srcs[] = { conv_t_weight, conv_t_input_f16 };
    ggml_tensor* glu_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 4);
    ggml_tensor* glu_q4 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 4);
    ggml_tensor* glu_f16_srcs[] = { glu_f16 };
    ggml_tensor* glu_q4_srcs[] = { glu_q4 };
    ggml_tensor* norm_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, 4);
    ggml_tensor* norm_gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* norm_beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 4);
    ggml_tensor* norm_srcs[] = { norm_x, norm_gamma, norm_beta };
    float norm_eps = 1e-5f;
    int32_t bad_gated_params[] = { 10 };
    ggml_tensor* bad_gated_srcs[] = { glu_f16 };
    ggml_tensor* attn_q = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 16, 8, 2);
    ggml_tensor* attn_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 16, 8, 2);
    ggml_tensor* attn_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 16, 8, 2);
    ggml_tensor* attn_srcs[] = { attn_q, attn_k, attn_v, nullptr, nullptr };
    int32_t attn_params[] = { 0, -1 };

    bool saw_cpu = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (!device) continue;
        const std::string name = ggml_backend_dev_name(device);
        if (!is_backend(name, "CPU") && !is_backend(name, "CUDA") && !is_backend(name, "SYCL")) continue;

        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        const bool supported = ggml_ops_backend_supports_op(
            backend,
            ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D,
            srcs,
            2,
            params,
            sizeof(params)
        );
        if (!supported) {
            std::cerr << name << " rejected supported Q4 Conv." << std::endl;
            return 1;
        }
        if (!ggml_ops_ext::probe_ops_kernel({
                device, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, srcs, 2,
                params, sizeof(params), valid_conv_output
            }).supported ||
            ggml_ops_ext::probe_ops_kernel({
                device, ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_1D, srcs, 2,
                params, sizeof(params), invalid_conv_output
            }).supported) {
            std::cerr << name << " did not enforce Conv output contract." << std::endl;
            return 1;
        }
        if (ggml_ops_conv_1d(ctx, unsupported_weight, input, 1, 0, 1, 1, backend) != nullptr) {
            std::cerr << name << " silently fell back for unsupported quantized Conv." << std::endl;
            return 1;
        }

        if (!ggml_ops_backend_supports_op(
                backend, ggml_ops_ext::GGML_OP_OPS_VIRT_GLU,
                glu_f16_srcs, 1, nullptr, 0)) {
            std::cerr << name << " rejected supported F16 GLU." << std::endl;
            return 1;
        }
        if (ggml_ops_backend_supports_op(
                backend, ggml_ops_ext::GGML_OP_OPS_VIRT_GLU,
                glu_q4_srcs, 1, nullptr, 0)) {
            std::cerr << name << " accepted quantized GLU activation." << std::endl;
            return 1;
        }
        if (!ggml_ops_backend_supports_op(
                backend, ggml_ops_ext::GGML_OP_OPS_VIRT_INSTANCE_NORM,
                norm_srcs, 3, &norm_eps, sizeof(norm_eps))) {
            std::cerr << name << " rejected supported mixed-precision InstanceNorm." << std::endl;
            return 1;
        }
        if (ggml_ops_backend_supports_op(
                backend, ggml_ops_ext::GGML_OP_OPS_VIRT_GATED_TANH_SIGMOID,
                bad_gated_srcs, 1, bad_gated_params, sizeof(bad_gated_params))) {
            std::cerr << name << " accepted invalid gated activation shape." << std::endl;
            return 1;
        }
        if (!ggml_ops_backend_supports_op(
                backend, ggml_ops_ext::GGML_OP_OPS_VIRT_FUSED_ATTN,
                attn_srcs, 5, attn_params, sizeof(attn_params))) {
            std::cerr << name << " rejected supported F16 streaming attention." << std::endl;
            return 1;
        }

        if (is_backend(name, "CPU")) {
            saw_cpu = true;
            if (!ggml_ops_backend_supports_op(
                    backend,
                    ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
                    conv_t_srcs,
                    2,
                    params,
                    sizeof(params))) {
                std::cerr << "CPU rejected custom F32 ConvTranspose." << std::endl;
                return 1;
            }
            if (!ggml_ops_backend_supports_op(
                    backend,
                    ggml_ops_ext::GGML_OP_OPS_VIRT_CONV_TRANSPOSE_1D,
                    conv_t_f16_srcs,
                    2,
                    params,
                    sizeof(params))) {
                std::cerr << "CPU rejected custom F16 ConvTranspose." << std::endl;
                return 1;
            }
        }
        ggml_backend_free(backend);
    }

    ggml_free(ctx);
    ggml_ops_ext::release_ops_hook();
    if (g_ggml_bridge_hook || g_ggml_bridge_supports_hook) {
        std::cerr << "Final runtime release did not uninstall ops hooks." << std::endl;
        return 1;
    }
    if (!saw_cpu) {
        std::cerr << "CPU backend was not available." << std::endl;
        return 1;
    }

    std::cout << "Ops architecture lifecycle and capability checks passed." << std::endl;
    return 0;
}
