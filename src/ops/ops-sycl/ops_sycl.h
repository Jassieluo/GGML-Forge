#pragma once

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

// ────────────────────────────────────────────────────────────
// Bridge function pointer — resolved at runtime from
// ggml-sycl.dll (MODULE_LIBRARY, no import library).
// ────────────────────────────────────────────────────────────

typedef void* (*pfn_bridge_sycl_get_queue_t)(ggml_backend_t);
typedef bool (*pfn_bridge_sycl_dequantize_t)(ggml_backend_t, const struct ggml_tensor *, void *, enum ggml_type);
typedef void* (*pfn_bridge_sycl_pool_alloc_t)(ggml_backend_t, size_t, size_t*);
typedef void (*pfn_bridge_sycl_pool_free_t)(ggml_backend_t, void*, size_t);

extern pfn_bridge_sycl_get_queue_t g_bridge_sycl_get_queue;
extern pfn_bridge_sycl_dequantize_t g_bridge_sycl_dequantize;
extern pfn_bridge_sycl_pool_alloc_t g_bridge_sycl_pool_alloc;
extern pfn_bridge_sycl_pool_free_t g_bridge_sycl_pool_free;

inline void * ggml_ops_ext_bridge_sycl_get_queue(ggml_backend_t backend) {
    return g_bridge_sycl_get_queue ? g_bridge_sycl_get_queue(backend) : nullptr;
}

inline bool ggml_ops_ext_bridge_sycl_dequantize(
    ggml_backend_t backend, const struct ggml_tensor * src, void * dst, enum ggml_type dst_type
) {
    return g_bridge_sycl_dequantize && g_bridge_sycl_dequantize(backend, src, dst, dst_type);
}

template <typename T>
class ops_sycl_pool_alloc {
public:
    explicit ops_sycl_pool_alloc(ggml_backend_t backend) : backend_(backend) {}
    ~ops_sycl_pool_alloc() { reset(); }

    ops_sycl_pool_alloc(const ops_sycl_pool_alloc&) = delete;
    ops_sycl_pool_alloc& operator=(const ops_sycl_pool_alloc&) = delete;

    bool alloc(size_t count) {
        reset();
        ptr_ = g_bridge_sycl_pool_alloc
            ? static_cast<T*>(g_bridge_sycl_pool_alloc(backend_, count * sizeof(T), &actual_size_))
            : nullptr;
        return ptr_ != nullptr;
    }

    void reset() {
        if (ptr_ && g_bridge_sycl_pool_free) {
            g_bridge_sycl_pool_free(backend_, ptr_, actual_size_);
        }
        ptr_ = nullptr;
        actual_size_ = 0;
    }

    T* get() const { return ptr_; }

private:
    ggml_backend_t backend_ = nullptr;
    T* ptr_ = nullptr;
    size_t actual_size_ = 0;
};

namespace ggml_ops_ext {
namespace sycl {

void register_backend();

} // namespace sycl
} // namespace ggml_ops_ext
