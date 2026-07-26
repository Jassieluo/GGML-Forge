#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ops/ops.h"
#include "ops_sycl.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace ggml_ops_ext {
namespace sycl {

template <typename T> class PosEncodingSYCLKernel;

// Additive sinusoidal position encoding. dims (0 = feature, 1 = time),
// broadcast over dims 2/3. log(base) is precomputed on the host as a float so
// the kernel only uses float transcendentals (no fp64 on the target iGPU).
// The optional dynamic position scalar lives in DEVICE memory and must be read
// inside the kernel.
template <typename T>
static void launch_pos_encoding(::sycl::queue* queue, const T* x, T* dst,
                                const int32_t* position, int64_t width, int64_t tokens,
                                int64_t nelements, int32_t offset, float log_base) {
    constexpr size_t local_size = 256;
    const size_t global_size =
        (static_cast<size_t>(nelements) + local_size - 1) / local_size * local_size;

    queue->submit([&](::sycl::handler& handler) {
        handler.parallel_for<PosEncodingSYCLKernel<T>>(
            ::sycl::nd_range<1>(global_size, local_size), [=](::sycl::nd_item<1> item) {
                const int64_t index = static_cast<int64_t>(item.get_global_linear_id());
                if (index >= nelements) {
                    return;
                }
                const int64_t i = index % width;
                const int64_t t = (index / width) % tokens;
                const int64_t start =
                    static_cast<int64_t>(offset) +
                    (position ? static_cast<int64_t>(*position) : int64_t(0));
                const float pos = static_cast<float>(start + t);
                const int64_t pair = i - (i % 2);
                const float angle =
                    pos * ::sycl::exp(-log_base * static_cast<float>(pair) /
                                      static_cast<float>(width));
                const float pe =
                    ((i % 2) == 0) ? ::sycl::sin(angle) : ::sycl::cos(angle);
                dst[index] = static_cast<T>(static_cast<float>(x[index]) + pe);
            });
    });
}

bool ggml_sycl_op_pos_encoding_entry(ggml_backend_t backend, struct ggml_tensor* node) {
    try {
        if (!node || !node->src[0]) {
            return false;
        }
        ggml_tensor* x = node->src[0];
        ggml_tensor* position = node->src[1]; // optional I32 scalar, may be null

        ops_pos_encoding_params params;
        std::memcpy(&params, node->op_params, sizeof(params));

        ::sycl::queue* queue =
            static_cast<::sycl::queue*>(ggml_ops_ext_bridge_sycl_get_queue(backend));
        if (!queue) {
            return false;
        }

        const int64_t width = x->ne[0];
        const int64_t tokens = x->ne[1];
        const int64_t nelements = ggml_nelements(x);
        const float log_base = std::log(params.base);
        const int32_t* position_data =
            position ? static_cast<const int32_t*>(position->data) : nullptr;

        if (x->type == GGML_TYPE_F32) {
            launch_pos_encoding(queue, static_cast<const float*>(x->data),
                                static_cast<float*>(node->data), position_data, width, tokens,
                                nelements, params.offset, log_base);
            return true;
        }
        if (x->type == GGML_TYPE_F16) {
            launch_pos_encoding(queue, static_cast<const ::sycl::half*>(x->data),
                                static_cast<::sycl::half*>(node->data), position_data, width,
                                tokens, nelements, params.offset, log_base);
            return true;
        }
        return false;
    } catch (const ::sycl::exception& e) {
        std::fprintf(stderr, "SYCL PosEncoding Exception: %s\n", e.what());
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SYCL PosEncoding Exception: %s\n", e.what());
        return false;
    }
}

} // namespace sycl
} // namespace ggml_ops_ext
