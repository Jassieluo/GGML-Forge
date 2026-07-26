# NN public API reference

## Scope and include paths

The NN layer sits between provider models and Ops:

```text
provider model -> nn::layers / nn::functional -> Forge Ops or native GGML
```

Include `nn/nn.h` for the complete API. Individual headers under
`include/nn/` may be used to keep dependencies narrow. `nn::F` is an alias for
`nn::functional`.

NN objects build GGML graphs; they do not execute eagerly. Tensor dimensions
follow GGML order, with dimension 0 contiguous. A backend passed to an NN
function is used for capability routing while the graph is built. The backend,
weight storage, graph context, and borrowed input data must remain alive for
their documented lifetimes.

## Context and tensor bindings

`nn::Context` owns or borrows a `ggml_context`, tracks tensors created through
it, and uploads bound input data before execution.

| API | Purpose |
| --- | --- |
| `Context(bytes, no_alloc)` | Own a GGML metadata context. `no_alloc=true` is the normal backend-execution mode. |
| `Context::borrow(ctx)` | Wrap an existing `ggml_context` without taking ownership. |
| `empty(name, shape, type)` | Create an uninitialized tensor. |
| `input<T>(...)` | Create an input that must be bound before execution. |
| `constant<T>(..., data::copy(...))` | Copy host data into context-owned staging storage. |
| `tensor<T>(..., data::borrow(...))` | Bind host storage without copying it. |
| `zeros<T>` / `full<T>` | Create staged constant data. |
| `bind`, `write`, `read` | Bind or transfer typed tensor data. |
| `view` / `view_bytes` | Create validated tensor views. Use byte offsets for quantized storage. |
| `create_graph`, `build`, `expand` | Create and populate a forward graph. |
| `materialize()` | Upload dirty bound inputs. `Executor::compute` calls it automatically. |
| `reset()` | Invalidate tensors and bindings, then reset the GGML context. |

`nn::data::borrow` does not extend the source lifetime. The source must remain
valid through `materialize()` or `compute()`. `nn::data::copy` is safer for
temporary containers. Rvalue vectors cannot be borrowed.

```cpp
std::vector<float> values(256);
nn::Context context(2 * 1024 * 1024);
ggml_tensor* input = context.input<float>(
    "input", {64, 4}, nn::data::borrow(values));
ggml_tensor* output = ggml_sqr(context.native_handle(), input);
ggml_cgraph* graph = context.build(output);
```

## Executor

`nn::Executor` owns a GGML graph allocator for one application-owned backend.
The backend must outlive the executor. It is a single-backend convenience: code
that needs scheduler-based multi-backend graph splitting (`ggml_backend_sched_t`)
should manage its own allocation instead.

```cpp
nn::Executor executor(backend);
executor.prepare(context, graph);
executor.compute(context, graph);
context.read(output, result.data(), result.size());
```

- `prepare` allocates the graph and binds the exact `Context`/graph pair.
- `compute` materializes inputs and uses `ops_backend_graph_compute`, so Forge
  virtual operators are dispatched correctly.
- `synchronize` waits for the backend to finish outstanding work.
- Rebuilds and `Context::reset()` require another `prepare`.
- `buffer_size()` reports the graph allocator buffer size.

## Modules and parameter paths

Stateful layers derive from `nn::Module<Derived>` and implement `forward`.
`Module::operator()` forwards directly to `forward`.

Parameters and owned children must be registered through the protected
`parameter` and `submodule` helpers. Registration creates canonical dot paths
used by schemas and loaders.

```cpp
class ProjectionBlock : public nn::Module<ProjectionBlock> {
public:
    nn::Linear& projection = submodule<nn::Linear>("projection");
    nn::Parameter& scale = parameter(
        "scale", nn::Parameter::optional(
            std::nullopt, nn::Parameter::Usage::scalar));

    ggml_tensor* forward(ggml_context* ctx, ggml_tensor* input) {
        ggml_tensor* value = projection(ctx, input);
        return scale.is_bound() ? ggml_mul(ctx, value, scale.tensor()) : value;
    }
};
```

Child and parameter names must be one path segment and must not contain `.`.
Modules are non-copyable and non-movable because registered members store stable
addresses. Use `ModuleList<T>`, `ModuleDict<T>`, and `ParameterList` for dynamic
collections.

`ModuleBase` provides:

- `for_each_parameter`, `find_parameter`, and `parameter_count`;
- `schema()` for a loader-visible `ModelSchema`;
- `to(backend)` to propagate a backend pointer to the module tree;
- `attach_state_dict` and `state_dict` for immutable loaded weight ownership.

## Parameter

`nn::Parameter` describes presence, logical shape, and storage usage separately
from the bound tensor.

```cpp
nn::Parameter::required(
    nn::Shape{768, 768}, nn::Parameter::Usage::linear_weight);
nn::Parameter::optional(std::nullopt, nn::Parameter::Usage::bias);
```

Important methods:

- `bind(tensor)` or `bind(tensor, logical_shape, layout)`;
- `tensor()` to resolve and return a bound or tied parameter;
- `local_tensor()` to inspect only local binding;
- `tie(other)` and `resolve_tie()` for shared weights;
- `supports_direct_storage(type)` and `storage_capability()` for loader checks;
- `logical_shape()`, `layout()`, and `storage_type()` for artifact/runtime
  separation.

`Parameter::Usage` comes from the Ops quantization contract. Use the most
specific value (`linear_weight`, `embedding_weight`, `conv1d_weight`,
`conv2d_weight`, `norm_affine`, `bias`, and so on), because it controls accepted
storage types and quantized layouts.

## State dictionaries, schemas, and loading

`ModelSchema::from(module)` or `module.schema()` produces a list of canonical
parameter paths, required/optional presence, logical shapes, storage
capabilities, and quantized layout requirements. `to_json()` is suitable for
conversion tooling and diagnostics.

`nn::io::Source` is the artifact abstraction. It exposes tensor metadata,
lookup, and ranged reads. `GGUFSource` implements it for GGUF files.

```cpp
nn::io::GGUFSource source(path);
nn::io::LoadResult loaded = nn::io::load_into(model, source, backend);
if (!loaded) {
    throw std::runtime_error(loaded.error);
}
```

Loading is strict: required tensors must exist, unexpected tensors are rejected,
names must be unique, logical shapes must match, quantized rows must be block
aligned, and storage/layout must be directly executable. A successful
`StateDict` owns the backend buffer and remains attached to the module.

## Functional API

Functional calls are stateless graph builders. They may create a Forge virtual
node, compose native GGML nodes, or do both through capability routing.

| Header | Public functions |
| --- | --- |
| `functional/linear.h` | `linear` |
| `functional/activation.h` | `mish`, `gated_tanh_sigmoid`, `alias_free_activation1d`, `swiglu`, `geglu`, `reglu` |
| `functional/attention.h` | `attention`, `relative_position_keys`, `relative_position_values` |
| `functional/convolution.h` | Conv/ConvTranspose 1D, 2D, and packed 3D |
| `functional/normalization.h` | `layer_norm`, `rms_norm`, `group_norm`, `l2_normalize` |
| `functional/pooling.h` | Max/Avg and adaptive Max/Avg pool 1D/2D/3D |
| `functional/padding.h` | Pad 1D/2D/3D |
| `functional/interpolation.h` | legacy interpolation plus native Resize 1D/2D/3D |
| `functional/positional.h` | sinusoidal position embedding |
| `functional/elementwise.h` | scalar elementwise helpers |

Passing `backend=nullptr` is valid only when the called function has a native
GGML composition fallback. Direct-only Forge operators return `nullptr` when
the concrete backend, dtype, layout, or shape is unsupported.

## Layer API

Reusable stateful layers include:

- `Linear`, `Embedding`;
- `Conv1d`, `ConvTranspose1d`, `Conv2d`, `ConvTranspose2d`, `Conv3d`, and
  `ConvTranspose3d`;
- `LayerNorm`, `InstanceNorm`, `AdaLN`, and `AdaLayerNormZero`;
- `Snake`, `PReLU`, and `GLU`;
- `FeedForward`, `MultiHeadAttention`, and `KVHeadAttention`;
- encoder, decoder, and DiT transformer blocks.

Layer fields intentionally expose architecture configuration. Providers remain
responsible for assigning head counts, dimensions, strides, padding, and model
version policy before graph construction.

## KV cache

`nn::KVCache` owns backend-resident K/V tensors. `allocate` accepts F32, F16,
Q8_0, or Q4_0 key/value storage when `head_dim` satisfies the format block size.
`prefill_attention` and `decode_attention` build cache update and attention
nodes for a selected layer. The cache and its backend must outlive every graph
that references it.

## Failure and lifetime rules

- Exceptions report API misuse, allocation failure, missing bindings, and
  loader validation errors.
- A `nullptr` graph tensor from a functional/Ops wrapper means unsupported or
  invalid graph construction; never add it to a graph.
- `ggml_context` metadata must outlive its tensors and graphs.
- Model `StateDict`, mutable KV cache, backend, and graph buffers have separate
  ownership and must not be conflated.
- NN does not silently move an unsupported operation to CPU.
