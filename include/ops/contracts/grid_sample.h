#pragma once

#include "ops/types.h"

#include <cstdint>
#include <cstring>

namespace ggml_ops_ext {

enum class ops_grid_sample_mode : uint8_t { nearest = 0, bilinear = 1 };
enum class ops_grid_padding_mode : uint8_t { zeros = 0, border = 1, reflection = 2 };

struct ops_grid_sample_2d_config {
    ops_grid_sample_mode mode = ops_grid_sample_mode::bilinear;
    ops_grid_padding_mode padding = ops_grid_padding_mode::zeros;
    bool align_corners = false;
};
struct ops_grid_sample_2d_params {
    uint8_t mode = static_cast<uint8_t>(ops_grid_sample_mode::bilinear);
    uint8_t padding = static_cast<uint8_t>(ops_grid_padding_mode::zeros);
    uint8_t align_corners = 0;
    uint8_t reserved = 0;
};
static_assert(sizeof(ops_grid_sample_2d_params) <= 64);

struct ops_grid_sample_2d_desc {
    ops_grid_sample_mode mode = ops_grid_sample_mode::bilinear;
    ops_grid_padding_mode padding = ops_grid_padding_mode::zeros;
    bool align_corners = false;
    int64_t input_width = 0, input_height = 0, channels = 0, batch = 0;
    int64_t output_width = 0, output_height = 0;
};

inline ops_status ops_validate_grid_sample_2d(const ops_request& request,
                                              ops_grid_sample_2d_desc* output_desc = nullptr) {
    if (!request.srcs || request.n_srcs < 2 || !request.srcs[0] || !request.srcs[1] ||
        !request.params || request.params_size < sizeof(ops_grid_sample_2d_params)) {
        return ops_status::error(ops_status_code::invalid_request, "GridSample2D request is incomplete");
    }
    ops_grid_sample_2d_params params{}; std::memcpy(&params, request.params, sizeof(params));
    if (params.mode > static_cast<uint8_t>(ops_grid_sample_mode::bilinear) ||
        params.padding > static_cast<uint8_t>(ops_grid_padding_mode::reflection) || params.align_corners > 1) {
        return ops_status::error(ops_status_code::invalid_request, "GridSample2D parameters are invalid");
    }
    const ggml_tensor* input = request.srcs[0]; const ggml_tensor* grid = request.srcs[1];
    if (grid->ne[0] != 2 || grid->ne[3] != input->ne[3] || input->ne[0] <= 0 || input->ne[1] <= 0 ||
        input->ne[2] <= 0 || input->ne[3] <= 0 || grid->ne[1] <= 0 || grid->ne[2] <= 0) {
        return ops_status::error(ops_status_code::invalid_request, "GridSample2D shapes are invalid");
    }
    ops_grid_sample_2d_desc desc;
    desc.mode = static_cast<ops_grid_sample_mode>(params.mode);
    desc.padding = static_cast<ops_grid_padding_mode>(params.padding);
    desc.align_corners = params.align_corners != 0;
    desc.input_width = input->ne[0]; desc.input_height = input->ne[1]; desc.channels = input->ne[2];
    desc.batch = input->ne[3]; desc.output_width = grid->ne[1]; desc.output_height = grid->ne[2];
    if (request.output && (request.output->type != input->type || request.output->ne[0] != desc.output_width ||
        request.output->ne[1] != desc.output_height || request.output->ne[2] != desc.channels ||
        request.output->ne[3] != desc.batch)) {
        return ops_status::error(ops_status_code::invalid_request, "GridSample2D output mismatch");
    }
    if (output_desc) *output_desc = desc; return ops_status::ok();
}

} // namespace ggml_ops_ext
