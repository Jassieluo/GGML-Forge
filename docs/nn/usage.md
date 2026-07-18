# Using NN

Include `nn/nn.h` for the complete public NN surface, or include individual
headers when a smaller dependency boundary is preferred.

## Build and execute a graph

`nn::Context` owns graph metadata and input bindings. `nn::Executor` owns the
backend graph allocator and executes one prepared graph.

```cpp
#include "nn/nn.h"

#include <array>

ggml_backend_t backend = /* application-owned backend */;
std::array<float, 4> input_values = {1, 2, 3, 4};
std::array<float, 4> output_values = {};

nn::Context context(1024 * 1024);
ggml_tensor* input = context.input<float>(
    "input", {4}, nn::data::borrow(input_values));
ggml_tensor* output = ggml_sqr(context.native_handle(), input);
ggml_cgraph* graph = context.build(output);

nn::Executor executor(backend);
executor.prepare(context, graph);
executor.compute(context, graph);
context.read(output, output_values.data(), output_values.size());
```

The backend must outlive the executor. Borrowed input memory must remain valid
through `compute`; use `nn::data::copy` when the context should own a copy.
Every required input must be bound before execution.

Call `prepare` again after rebuilding or resetting a graph. For asynchronous
execution, call `compute_async` followed by `synchronize` before reading output
or reusing mutable input storage.

## Define modules

Derive from `nn::Module<Derived>`, register parameters with `parameter`, and
register owned child modules with `submodule`. Parameter names form the
canonical state-dictionary path, so they must match the converter and model
schema.

```cpp
class Block : public nn::Module<Block> {
public:
    nn::Linear& projection = submodule<nn::Linear>("projection");

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input) {
        return projection(ctx, input);
    }
};
```

Use `nn::io::load_into` with an `nn::io::Source` to validate and attach model
state to a module. Model-category and provider decisions do not belong in NN
modules.
