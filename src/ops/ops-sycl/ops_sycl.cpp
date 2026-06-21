#include "ops/ops.h"
#include "ops_sycl.h"
#include "ggml.h"
#include "ggml-backend.h"

namespace tts {
namespace ops {
namespace sycl {

bool compute_conv_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    // Placeholder: Future Intel oneDNN optimizations can be implemented here.
    // For now, return false to let the backend run the standard GGML nodes.
    (void)backend; (void)w; (void)x; (void)dst; (void)stride; (void)padding; (void)dilation;
    return false;
}

bool compute_conv_transpose_1d(
    ggml_backend_t backend,
    struct ggml_tensor* w,
    struct ggml_tensor* x,
    struct ggml_tensor* dst,
    int stride,
    int padding,
    int dilation
) {
    (void)backend; (void)w; (void)x; (void)dst; (void)stride; (void)padding; (void)dilation;
    return false;
}

void register_backend() {
    ops_backend_interface iface;
    iface.backend_name_prefix = "SYCL";
    iface.compute_conv_1d = compute_conv_1d;
    iface.compute_conv_transpose_1d = compute_conv_transpose_1d;
    register_ops_backend(iface);
}

} // namespace sycl
} // namespace ops
} // namespace tts
