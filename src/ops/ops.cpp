#include "ops/ops.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

static struct ggml_cgraph local_graph_view(struct ggml_cgraph * cgraph0, int i0, int i1) {
    struct ggml_cgraph cgraph = {};
    cgraph.size             = 0;
    cgraph.n_nodes          = i1 - i0;
    cgraph.n_leafs          = 0;
    cgraph.nodes            = cgraph0->nodes + i0;
    cgraph.grads            = nullptr;
    cgraph.grad_accs        = nullptr;
    cgraph.leafs            = nullptr;
    cgraph.use_counts       = cgraph0->use_counts;
    cgraph.visited_hash_set = cgraph0->visited_hash_set;
    cgraph.order            = cgraph0->order;
    cgraph.uid              = 0;
    return cgraph;
}
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>

namespace tts {
namespace ops {

typedef enum ggml_status (*ggml_backend_graph_compute_t)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

static std::unordered_map<ggml_backend_t, ggml_backend_graph_compute_t> g_original_computes;
static std::vector<ops_backend_interface> g_registered_backends;
static std::recursive_mutex g_hooks_mutex;

} // namespace ops
} // namespace tts

namespace gpt_sovits {
    extern thread_local ggml_backend_t current_vits_backend;
}

namespace tts {
namespace ops {

void register_ops_backend(const ops_backend_interface& iface) {
    std::lock_guard<std::recursive_mutex> lock(g_hooks_mutex);
    g_registered_backends.push_back(iface);
}

static const ops_backend_interface* find_ops_backend(ggml_backend_t backend) {
    const char* bname = ggml_backend_name(backend);
    if (!bname) return nullptr;

    for (const auto& iface : g_registered_backends) {
        size_t prefix_len = std::strlen(iface.backend_name_prefix);
        if (std::strncmp(bname, iface.backend_name_prefix, prefix_len) == 0) {
            return &iface;
        }
    }
    return nullptr;
}

static enum ggml_status ops_graph_compute_hook(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    const ops_backend_interface* ops_backend = find_ops_backend(backend);

    ggml_backend_graph_compute_t orig_compute = nullptr;
    {
        std::lock_guard<std::recursive_mutex> lock(g_hooks_mutex);
        auto it = g_original_computes.find(backend);
        if (it != g_original_computes.end()) {
            orig_compute = it->second;
        }
    }
    if (!orig_compute) {
        return GGML_STATUS_FAILED;
    }

    if (!ops_backend) {
        return orig_compute(backend, cgraph);
    }

    int start_idx = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor* node = cgraph->nodes[i];
        if (!node) continue;

        bool is_custom_conv = false;
        bool is_custom_conv_t = false;

        // Pattern A: Fused 1D Convolution (im2col + mul_mat)
        struct ggml_tensor* w_conv = nullptr;
        struct ggml_tensor* x_conv = nullptr;
        struct ggml_tensor* im2col_node = nullptr;
        int stride = 0, padding = 0, dilation = 0;

        if (node->op == GGML_OP_MUL_MAT) {
            struct ggml_tensor* src0 = node->src[0];
            struct ggml_tensor* src1 = node->src[1];

            if (src1) {
                if (src1->op == GGML_OP_IM2COL) {
                    im2col_node = src1;
                } else if (src1->op == GGML_OP_RESHAPE && src1->src[0] && src1->src[0]->op == GGML_OP_IM2COL) {
                    im2col_node = src1->src[0];
                }
            }

            if (im2col_node && ops_backend->compute_conv_1d) {
                w_conv = src0;
                if (w_conv->op == GGML_OP_RESHAPE) {
                    w_conv = w_conv->src[0];
                }
                x_conv = im2col_node->src[1];

                int32_t* params = (int32_t*)im2col_node->op_params;
                stride = params[0];
                padding = params[2];
                dilation = params[4];
                is_custom_conv = true;
            }
        }

        // Pattern B: 1D Transposed Convolution (GGML_OP_CONV_TRANSPOSE_1D)
        if (node->op == GGML_OP_CONV_TRANSPOSE_1D && ops_backend->compute_conv_transpose_1d) {
            is_custom_conv_t = true;
        }

        if (is_custom_conv || is_custom_conv_t) {
            // 1. Execute standard nodes preceding this custom node
            if (i > start_idx) {
                struct ggml_cgraph sub_graph = local_graph_view(cgraph, start_idx, i);
                enum ggml_status status = orig_compute(backend, &sub_graph);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }

            // 2. Execute this custom node
            if (is_custom_conv) {
                bool success = ops_backend->compute_conv_1d(backend, w_conv, x_conv, node, stride, padding, dilation);
                if (!success) {
                    return GGML_STATUS_FAILED;
                }
                im2col_node->op = GGML_OP_NONE;
                struct ggml_tensor* src1 = node->src[1];
                if (src1 && src1->op == GGML_OP_RESHAPE) {
                    src1->op = GGML_OP_NONE;
                }
                node->op = GGML_OP_NONE;
            } else if (is_custom_conv_t) {
                struct ggml_tensor* w = node->src[0];
                struct ggml_tensor* x = node->src[1];
                int32_t* params = (int32_t*)node->op_params;
                int stride = params[0];
                int padding = params[1];
                int dilation = params[2];

                bool success = ops_backend->compute_conv_transpose_1d(backend, w, x, node, stride, padding, dilation);
                if (!success) {
                    return GGML_STATUS_FAILED;
                }
                node->op = GGML_OP_NONE;
            }

            start_idx = i + 1;
        }
    }

    // 3. Execute any remaining standard nodes
    if (start_idx < cgraph->n_nodes) {
        struct ggml_cgraph sub_graph = local_graph_view(cgraph, start_idx, cgraph->n_nodes);
        enum ggml_status status = orig_compute(backend, &sub_graph);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }

    return GGML_STATUS_SUCCESS;
}

#ifdef GGML_USE_CUDA
namespace cuda {
    void register_backend();
}
#endif
#ifdef GGML_USE_SYCL
namespace sycl {
    void register_backend();
}
#endif

void install_ops_hook(ggml_backend_t backend) {
    if (!backend) return;
    std::lock_guard<std::recursive_mutex> lock(g_hooks_mutex);

    static bool backends_registered = false;
    if (!backends_registered) {
#ifdef GGML_USE_CUDA
        cuda::register_backend();
#endif
#ifdef GGML_USE_SYCL
        sycl::register_backend();
#endif
        backends_registered = true;
    }

    if (g_original_computes.find(backend) == g_original_computes.end()) {
        g_original_computes[backend] = backend->iface.graph_compute;
        backend->iface.graph_compute = ops_graph_compute_hook;
    }
}

void uninstall_ops_hook(ggml_backend_t backend) {
    if (!backend) return;
    std::lock_guard<std::recursive_mutex> lock(g_hooks_mutex);
    auto it = g_original_computes.find(backend);
    if (it != g_original_computes.end()) {
        backend->iface.graph_compute = it->second;
        g_original_computes.erase(it);
    }
}

struct ggml_tensor* ops_conv_transpose_1d(
    struct ggml_context* ctx,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    int stride,
    int padding,
    int dilation
) {
    bool is_gpu = false;
    if (gpt_sovits::current_vits_backend) {
        const char * bname = ggml_backend_name(gpt_sovits::current_vits_backend);
        if (bname && (strncmp(bname, "CUDA", 4) == 0 || strncmp(bname, "SYCL", 4) == 0)) {
            is_gpu = true;
        }
    }

    if (is_gpu) {
        // Native GPU path: build standard GGML_OP_CONV_TRANSPOSE_1D node with the actual padded shape
        int64_t out_w = (x->ne[0] - 1) * stride - 2 * padding + dilation * (w->ne[0] - 1) + 1;
        const int64_t ne[4] = { out_w, w->ne[1], x->ne[2], 1 };
        
        struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        result->op = GGML_OP_CONV_TRANSPOSE_1D;
        
        int32_t params[] = { stride, padding, dilation };
        ggml_set_op_params(result, params, sizeof(params));
        
        result->src[0] = w;
        result->src[1] = x;
        return result;
    } else {
        // CPU Fallback path: build padding=0 graph and crop the output to bypass standard GGML's padding restrictions
        struct ggml_tensor* conv_t_raw = ggml_conv_transpose_1d(ctx, w, x, stride, 0, dilation);
        
        if (padding > 0) {
            int64_t cropped_seq_len = conv_t_raw->ne[0] - 2 * padding;
            size_t offset_bytes = padding * conv_t_raw->nb[0];
            
            struct ggml_tensor* cropped_view = ggml_view_2d(
                ctx,
                conv_t_raw,
                cropped_seq_len,
                conv_t_raw->ne[1],
                conv_t_raw->nb[1],
                offset_bytes
            );
            
            return ggml_cont(ctx, cropped_view);
        }
        return conv_t_raw;
    }
}

} // namespace ops
} // namespace tts
