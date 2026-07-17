#pragma once

#include "nn/io/source.h"

#include <memory>
#include <string>

struct gguf_context;

namespace nn::io {

class GGUFSource final : public Source {
public:
    explicit GGUFSource(const std::string& path);
    ~GGUFSource() override;

    GGUFSource(const GGUFSource&) = delete;
    GGUFSource& operator=(const GGUFSource&) = delete;
    GGUFSource(GGUFSource&&) noexcept;
    GGUFSource& operator=(GGUFSource&&) noexcept;

    size_t size() const override;
    const TensorInfo& info(size_t index) const override;
    std::optional<size_t> find(std::string_view name) const override;
    bool read(size_t index, size_t offset, void* destination, size_t bytes) override;

    const gguf_context* metadata_context() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nn::io
