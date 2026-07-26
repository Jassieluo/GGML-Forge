# Extending NN

## Choose the correct layer

Before adding code, classify the requirement:

| Requirement | Location |
| --- | --- |
| A stateless composition of existing NN/Ops/GGML calls | `include/nn/functional/` and `src/nn/functional/` |
| A reusable object with parameters or child modules | `include/nn/layers/` and `src/nn/layers/` |
| Mutable reusable execution state | `include/nn/runtime/` and `src/nn/runtime/` |
| Artifact parsing or model weight binding | `include/nn/io/`, `src/nn/io/`, and schema code |
| New fused semantics or a missing efficient backend implementation | Add/extend Ops first, then expose it through NN |
| Model-version, provider, tokenizer, or pipeline policy | The provider, not NN |

Do not create a Forge virtual operator merely to rename an existing GGML
primitive. A small NN wrapper is enough when GGML already has correct and
optimized backend support.

## Add a functional operation

1. Declare a stateless graph builder in the relevant header under
   `include/nn/functional/`.
2. Implement it under `src/nn/functional/`.
3. Prefer a public `ggml_ops_*` wrapper when Forge owns semantics or routing;
   otherwise compose native GGML nodes.
4. Return `nullptr` unchanged when a required lower-level call is unsupported.
5. Add the source file to `src/nn/CMakeLists.txt` when it is new.
6. Export the header from `include/nn/nn.h` when it is general-purpose.

```cpp
// include/nn/functional/example.h
ggml_tensor* example(
    ggml_context* ctx, ggml_tensor* input,
    ggml_backend_t backend = nullptr);

// src/nn/functional/example.cpp
ggml_tensor* nn::functional::example(
    ggml_context* ctx, ggml_tensor* input, ggml_backend_t backend) {
    if (!ctx || !input) return nullptr;
    return ggml_ops_example(ctx, input, backend);
}
```

Functional code owns public shape conventions and argument defaults. Backend
kernel details, streams, synchronization, and device-name checks do not belong
here.

## Add a stateful layer

Derive from `Module<Derived>`, register parameters and child modules as member
references, and implement `forward` through functional calls.

```cpp
class AffineActivation : public nn::Module<AffineActivation> {
public:
    nn::Parameter& weight = parameter(
        "weight", nn::Parameter::required(
            std::nullopt, nn::Parameter::Usage::linear_weight));
    nn::Parameter& bias = parameter(
        "bias", nn::Parameter::optional(
            std::nullopt, nn::Parameter::Usage::bias));

    ggml_tensor* forward(
        nn::Context& context, ggml_tensor* input,
        ggml_backend_t backend = nullptr) {
        ggml_context* ctx = context.native_handle();
        ggml_tensor* value = nn::F::linear(
            ctx, input, weight.tensor(), bias.local_tensor(), backend);
        return value ? ggml_relu(ctx, value) : nullptr;
    }
};
```

Rules:

- Module entry points (`forward`, `prefill`, `decode`) take `nn::Context&`;
  only the stateless `nn::functional` tier takes a raw `ggml_context*`.
  Unwrap with `context.native_handle()` when calling functional or native
  GGML builders.

- Parameter names must match converter/GGUF tensor paths after parent module
  prefixes are applied.
- Use `local_tensor()` for optional parameters; `tensor()` is appropriate when
  a required or tied parameter must resolve.
- Declare the most specific `Parameter::Usage`; never mark convolution or
  embedding weights as generic merely to bypass loader validation.
- Do not manually own child modules with raw pointers. `submodule` provides
  ownership and registration.
- Modules must remain non-copyable/non-movable. Use module containers for
  runtime-sized collections.
- `forward` builds a graph only. Persistent mutable buffers belong in runtime
  objects, not temporary contexts or process-global variables.

## Dynamic module structures

Use `ModuleList<T>` for numbered repeated blocks, `ModuleDict<T>` for named
variants, and `ParameterList` for repeated parameters. Their generated names
become state paths (`layers.0.*`, `experts.router.*`, and so on).

```cpp
nn::ModuleList<Block>& layers =
    submodule<nn::ModuleList<Block>>("layers");

for (int i = 0; i < count; ++i) {
    layers.emplace_back(/* constructor args */);
}
```

Construct the final module tree before loading weights. Changing the tree after
`load_into` invalidates the expected schema and leaves new parameters unbound.

## Add parameter storage support

Parameter storage policy is defined by `ops_parameter_usage` and
`ops_storage_capability` in `include/ops/contracts/quantization.h`.

When a new parameter family needs a distinct layout or type policy:

1. Add a semantic usage value; do not key policy by tensor name.
2. Define accepted direct storage types and quantized layout.
3. Assign the usage in the NN layer.
4. Make the converter emit `storage_shape`, `logical_shape`, and `Layout`
   metadata that satisfy the policy.
5. Add strict loader tests for accepted and rejected artifacts.

The loader intentionally does not dequantize unsupported weights or guess
layouts. If a backend cannot consume the stored type directly, conversion must
choose another storage type or the Ops backend must gain support.

## Add a custom artifact source

Implement `nn::io::Source` when weights do not come from GGUF. The implementation
must provide stable `TensorInfo` records, unique lookup names, exact byte sizes,
and bounded `read` behavior. Keep format/version policy in the provider or
converter; `load_into` should continue to see a generic tensor source.

## Tests

At minimum, cover:

- canonical parameter paths and `ModelSchema` output;
- required, optional, tied, missing, extra, and duplicate parameters;
- logical/storage shape and layout validation;
- graph output against a host reference or established GGML composition;
- all intended CPU/CUDA/SYCL routes when the NN feature exposes a Forge op;
- lifetime behavior for borrowed/copied input and asynchronous execution.

NN tests live under `tests/nn_test/`. Ops-level numerical and backend tests
belong under `tests/ops_test/`; avoid testing the same kernel implementation
only through a large provider model.

## Build checklist

For NN-only changes, build the NN library and relevant test target. When the
change reaches Ops, use the focused Ops preset first:

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --target <test-target> -j 32
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --output-on-failure
```

Run an end-to-end provider test after changing a layer used by a real model.

## Review checklist

- Is the code in NN rather than a provider or backend for a concrete reason?
- Are parameter paths stable and converter-compatible?
- Are logical shape and storage layout kept separate?
- Does unsupported construction return `nullptr` or a clear error?
- Are backend/device-name branches absent from model and functional code?
- Are ownership and mutable state explicit?
- Are English and Chinese API/extension documents updated?
