#include "nn/core/executor.h"
#include "ops/ops.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace nn {

Executor::Executor(ggml_backend_t backend)
    : backend_(backend) {
    if (backend_ == nullptr) {
        throw std::invalid_argument("nn::Executor backend cannot be null");
    }
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (allocator_ == nullptr) {
        throw std::runtime_error("failed to create graph allocator for nn::Executor");
    }
}

Executor::~Executor() {
    if (allocator_ != nullptr) {
        ggml_gallocr_free(allocator_);
    }
}

Executor::Executor(Executor&& other) noexcept
    : backend_(std::exchange(other.backend_, nullptr)),
      allocator_(std::exchange(other.allocator_, nullptr)),
      prepared_context_(std::exchange(other.prepared_context_, nullptr)),
      prepared_graph_(std::exchange(other.prepared_graph_, nullptr)) {}

Executor& Executor::operator=(Executor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (allocator_ != nullptr) {
        ggml_gallocr_free(allocator_);
    }
    backend_ = std::exchange(other.backend_, nullptr);
    allocator_ = std::exchange(other.allocator_, nullptr);
    prepared_context_ = std::exchange(other.prepared_context_, nullptr);
    prepared_graph_ = std::exchange(other.prepared_graph_, nullptr);
    return *this;
}

const char* Executor::name() const noexcept {
    return backend_ != nullptr ? ggml_backend_name(backend_) : "<moved-from>";
}

void Executor::prepare(Context& context, ggml_cgraph* graph) {
    if (backend_ == nullptr || allocator_ == nullptr) {
        throw std::logic_error("cannot use a moved-from nn::Executor");
    }
    if (graph == nullptr) {
        throw std::invalid_argument("graph cannot be null");
    }
    if (!ggml_gallocr_alloc_graph(allocator_, graph)) {
        throw std::runtime_error(std::string("failed to allocate graph on backend ") + name());
    }
    prepared_context_ = context.native_handle();
    prepared_graph_ = graph;
}

void Executor::check_prepared(const Context& context, const ggml_cgraph* graph) const {
    if (backend_ == nullptr || allocator_ == nullptr) {
        throw std::logic_error("cannot use a moved-from nn::Executor");
    }
    if (graph == nullptr) {
        throw std::invalid_argument("graph cannot be null");
    }
    if (prepared_context_ != context.native_handle() || prepared_graph_ != graph) {
        throw std::logic_error("graph must be prepared by this nn::Executor before compute");
    }
}

void Executor::check_status(ggml_status status, const char* operation) {
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with ggml status " + std::to_string(static_cast<int>(status)));
    }
}

void Executor::compute(Context& context, ggml_cgraph* graph) {
    check_prepared(context, graph);
    context.materialize();
    check_status(ggml_ops_ext::ops_backend_graph_compute(backend_, graph), "backend graph compute");
}

void Executor::synchronize() {
    if (backend_ == nullptr) {
        throw std::logic_error("cannot use a moved-from nn::Executor");
    }
    ggml_backend_synchronize(backend_);
}

void Executor::reset() noexcept {
    prepared_context_ = nullptr;
    prepared_graph_ = nullptr;
}

size_t Executor::buffer_size() const noexcept {
    return allocator_ != nullptr ? ggml_gallocr_get_buffer_size(allocator_, 0) : 0;
}

} // namespace nn
