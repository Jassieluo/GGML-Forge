#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "nn/core/context.h"

#include <cstddef>

namespace nn {

class Executor {
public:
    explicit Executor(ggml_backend_t backend);
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    Executor(Executor&& other) noexcept;
    Executor& operator=(Executor&& other) noexcept;

    ggml_backend_t backend() const noexcept { return backend_; }
    const char* name() const noexcept;

    void prepare(Context& context, ggml_cgraph* graph);
    void compute(Context& context, ggml_cgraph* graph);
    void synchronize();
    void reset() noexcept;

    bool is_prepared() const noexcept { return prepared_graph_ != nullptr; }
    size_t buffer_size() const noexcept;

private:
    ggml_backend_t backend_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_context* prepared_context_ = nullptr;
    ggml_cgraph* prepared_graph_ = nullptr;

    void check_prepared(const Context& context, const ggml_cgraph* graph) const;
    static void check_status(ggml_status status, const char* operation);
};

} // namespace nn
