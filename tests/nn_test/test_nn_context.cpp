#include "nn/context.h"
#include "nn/executor.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_equal(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "vector size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
            throw std::runtime_error("vector value mismatch");
        }
    }
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<nn::Context>);
    static_assert(std::is_move_constructible_v<nn::Context>);

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    require(backend != nullptr, "failed to create CPU backend");

    try {
        nn::Context context(4 * 1024 * 1024);

        std::vector<float> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        std::array<int32_t, 3> ids{7, 8, 9};

        ggml_tensor* input = context.input<float>("input", {2, 3}, nn::data::borrow(input_data));
        ggml_tensor* constant = context.constant<int32_t>("ids", {3}, nn::data::copy(ids));
        ggml_tensor* zeros = context.zeros<float>("zeros", {2, 3});
        ggml_tensor* filled = context.full<float>("filled", {2, 3}, 2.5f);
        ggml_tensor* slice = context.view(input, {2, 2}, 2, "slice");
        ggml_tensor* sum = ggml_add(context.native_handle(), input, zeros);
        ggml_set_name(sum, "sum");

        // Raw ggml nodes remain valid escape-hatch outputs.
        ggml_cgraph* graph = context.build(sum);
        context.expand(graph, slice);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context.native_handle(), backend);
        require(buffer != nullptr, "failed to allocate context tensors");

        context.materialize();

        std::vector<float> actual_input(6);
        std::vector<float> actual_zeros(6);
        std::vector<float> actual_filled(6);
        std::array<int32_t, 3> actual_ids{};
        context.read(input, actual_input.data(), actual_input.size());
        context.read(zeros, actual_zeros.data(), actual_zeros.size());
        context.read(filled, actual_filled.data(), actual_filled.size());
        context.read(constant, actual_ids.data(), actual_ids.size());

        require_equal(actual_input, input_data);
        require_equal(actual_zeros, std::vector<float>(6, 0.0f));
        require_equal(actual_filled, std::vector<float>(6, 2.5f));
        require(actual_ids == ids, "constant copy mismatch");

        std::vector<float> rebound{6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        context.bind(input, nn::data::borrow(rebound));
        context.materialize();
        context.read(input, actual_input.data(), actual_input.size());
        require_equal(actual_input, rebound);

        std::vector<float> replacement{10.0f, 11.0f};
        context.write(input, replacement.data(), replacement.size(), 2);
        context.read(input, actual_input.data(), actual_input.size());
        require_equal(actual_input, {6.0f, 5.0f, 10.0f, 11.0f, 2.0f, 1.0f});

        bool rejected_bad_count = false;
        try {
            context.bind(input, nn::data::borrow(rebound.data(), rebound.size() - 1));
        } catch (const std::invalid_argument&) {
            rejected_bad_count = true;
        }
        require(rejected_bad_count, "mismatched input size was not rejected");

        ggml_backend_buffer_free(buffer);

        context.reset();
        ggml_tensor* reset_input = context.input<float>(
            "reset_input", {2}, nn::data::borrow(replacement));
        require(reset_input != nullptr && ggml_nelements(reset_input) == 2,
                "context reset did not allow a fresh graph declaration");

        ggml_context* native = ggml_init({1024 * 1024, nullptr, false});
        require(native != nullptr, "failed to create native context");
        {
            nn::Context borrowed = nn::Context::borrow(native);
            require(borrowed.empty<float>("borrowed", {2}) != nullptr,
                    "borrowed context could not create a tensor");
        }
        require(ggml_new_tensor_1d(native, GGML_TYPE_F32, 2) != nullptr,
                "borrowed context incorrectly released its native context");
        ggml_free(native);

        nn::Context execution_context(4 * 1024 * 1024);
        std::vector<float> lhs_data{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> rhs_data{4.0f, 3.0f, 2.0f, 1.0f};
        ggml_tensor* lhs = execution_context.input<float>("lhs", {4}, nn::data::borrow(lhs_data));
        ggml_tensor* rhs = execution_context.input<float>("rhs", {4}, nn::data::borrow(rhs_data));
        ggml_tensor* output = ggml_add(execution_context.native_handle(), lhs, rhs);
        ggml_cgraph* execution_graph = execution_context.build(output);

        nn::Executor executor(backend);
        executor.prepare(execution_context, execution_graph);
        require(executor.is_prepared(), "executor was not prepared");
        require(executor.buffer_size() > 0, "executor did not allocate a graph buffer");
        executor.compute(execution_context, execution_graph);

        std::vector<float> output_data(4);
        execution_context.read(output, output_data.data(), output_data.size());
        require_equal(output_data, {5.0f, 5.0f, 5.0f, 5.0f});
    } catch (...) {
        ggml_backend_free(backend);
        throw;
    }

    ggml_backend_free(backend);
    std::cout << "nn::Context tests passed\n";
    return 0;
}
