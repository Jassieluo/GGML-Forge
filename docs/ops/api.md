# Ops public API reference

## Architecture boundary

Forge Ops has four layers:

```text
public ggml_ops_* wrapper
        -> semantic contract and encoded parameters
        -> virtual GGML node and capability probe
        -> CPU / CUDA / SYCL registered kernel
```

Use `include/ops/ops.h` for the public API and dispatcher types. Contracts live
under `include/ops/contracts/`. Project kernels live under
`src/ops/ops-cpu`, `ops-cuda`, and `ops-sycl`.

Native GGML remains the preferred implementation for MatMul/Linear, RoPE,
ordinary Softmax, elementwise arithmetic, and other mature primitives. Forge
Ops is used for missing semantics, fused operations, quantized direct kernels,
or cases where a native GGML composition has unacceptable intermediate memory.

## Runtime lifecycle

Backend libraries must be loaded and their Forge registration entry points
must run before the first hook acquisition. Category runtimes handle this. A
standalone executable normally performs backend loading/initialization, then:

```cpp
ggml_ops_ext::acquire_ops_hook();

ggml_tensor* output = ggml_ops_mish(ctx, input, backend);
if (!output) {
    // Invalid or unsupported backend/dtype/layout/shape.
}

ggml_cgraph* graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, output);
ggml_status status =
    ggml_ops_ext::ops_backend_graph_compute(backend, graph);

ggml_ops_ext::release_ops_hook();
```

Every acquire must have a matching release. The first acquire freezes the
kernel registry; registration after that point fails. The hook itself is
process-wide and reference-counted.

`ops_backend_graph_compute` serializes whole-graph execution only for callers
sharing the same backend instance. Separate backend instances and different
devices retain independent execution lanes. There is no automatic CPU fallback.

## Capability and dispatch API

| API | Meaning |
| --- | --- |
| `ggml_ops_backend_supports_op` | Graph-build query using source tensors and optional encoded parameters. |
| `probe_ops_kernel(device, ...)` | Query a concrete device without a backend instance. |
| `probe_ops_kernel(ops_request)` | Full probe; may validate the output tensor and report workspace bytes/reason. |
| `register_ops_backend` | Cold-path registration of a backend prefix and kernel table. |
| `execute_ops_kernel` | Execute one Forge virtual node on a backend. Normally called by the bridge hook. |
| `ops_new_virtual_node` | Create a virtual node after semantic and capability checks. |
| `ops_backend_graph_compute` | Execute a graph with Forge lane serialization. |

`ops_request` carries `device`, `op_id`, sources, encoded parameters, and an
optional output tensor. A probe with no output validates only what is known at
construction time; backend execution probes must also validate output shape and
dtype.

`ops_probe_result` contains `supported`, optional `workspace_bytes`, and a
diagnostic `reason`. Dispatch chooses the highest-priority compatible kernel for
the concrete backend prefix.

## Public operator inventory

Forge currently exposes 37 `ggml_ops_*` graph builders.

### Indexing

| Function | Semantics |
| --- | --- |
| `ggml_ops_embedding` | Shared 2D embedding table with up to three I32 index dimensions. Returns `[embedding_width, ...index_shape]`. Uses native GGML GET_ROWS kernels and supports floating or compatible quantized rows. |

### Attention and cache

| Function | Semantics |
| --- | --- |
| `ggml_ops_attention` | Unified MHA/GQA attention with optional bias, weight output, sliding window, runtime valid length, and dependency. |
| `ggml_ops_kv_cache_update` | In-place logical K/V cache update for F32/F16/Q4_0/Q8_0 cache storage. |
| `ggml_ops_relative_pe_keys` | Relative-position key score contribution. |
| `ggml_ops_relative_pe_values` | Relative-position value contribution. |

Attention routing is capability-based:

```text
native GGML FlashAttention
  -> Forge streaming/fused Attention
  -> native GGML MatMul/Softmax composition
```

FlashAttention is used only when its semantics match the request. Generic
per-head bias, requested attention weights, runtime valid length, and cache
dependency may require the Forge path. No GPU model name is part of routing.

### Convolution

| Functions | Semantics |
| --- | --- |
| `ggml_ops_conv_1d`, `ggml_ops_conv_transpose_1d` | 1D grouped/depthwise convolution and transpose. |
| `ggml_ops_conv_2d`, `ggml_ops_conv_transpose_2d` | 2D direct convolution with per-axis geometry and output padding for transpose. |
| `ggml_ops_conv_3d`, `ggml_ops_conv_transpose_3d` | Packed 3D convolution; logical W/H/D is carried by the config. |

ConvND supports stride, asymmetric padding, dilation, groups, bias, channel-row
and flattened-row weight layouts, and backend-supported quantized storage.
Conv2D uses `[W,H,C,N]`. Packed Conv3D uses `[W*H*D,C,N]` because GGML has four
physical dimensions.

### Pooling, padding, and resize

| Functions | Semantics |
| --- | --- |
| `ggml_ops_pool_1d/2d/3d` | Max or average pool; kernel, stride, dilation, asymmetric padding, ceil mode, and average padding policy. |
| `ggml_ops_adaptive_pool_1d/2d/3d` | Adaptive max or average pool to explicit output sizes. |
| `ggml_ops_pad_1d/2d/3d` | Constant, reflect, replicate, or circular padding. |
| `ggml_ops_resize_1d/2d/3d` | Nearest or N-linear resize, including `align_corners`. |

These kernels map output coordinates directly and do not materialize im2col or
index/weight workspaces. 3D variants use the same packed spatial convention as
Conv3D.

### Normalization

| Function | Implementation |
| --- | --- |
| `ggml_ops_layer_norm` | Forge fused CPU/CUDA/SYCL kernel. |
| `ggml_ops_instance_norm` | Forge fused CPU/CUDA/SYCL kernel. |
| `ggml_ops_ada_ln` | Forge kernel with native GGML composition fallback. |
| `ggml_ops_rms_norm` | Native GGML RMSNorm plus optional affine multiplication. |
| `ggml_ops_group_norm` | Native GGML GroupNorm plus optional affine terms. |
| `ggml_ops_l2_normalize` | Axis-aware native GGML composition. |

### Activation and fused blocks

| Function | Semantics |
| --- | --- |
| `ggml_ops_mish` | Mish activation. |
| `ggml_ops_double_swish` | DoubleSwish activation. |
| `ggml_ops_glu` | Sigmoid GLU with composition fallback. |
| `ggml_ops_gated_activation` | SwiGLU, GeGLU, ReGLU, or identity gate on a selected axis. |
| `ggml_ops_gated_tanh_sigmoid` | Fused tanh/sigmoid gating. |
| `ggml_ops_snake`, `ggml_ops_snake_beta` | Periodic audio activations. |
| `ggml_ops_alias_free_activation` | Fused depthwise 2x up-filter, SnakeBeta, and 2x down-filter. CPU/CUDA/SYCL avoid the 2x intermediate tensor. |

## Configuration contracts

Public geometry structs are defined in contract headers:

- `ops_conv_nd_config`;
- `ops_pool_nd_config` and `ops_adaptive_pool_nd_config`;
- `ops_pad_nd_config`;
- `ops_resize_nd_config`.

Public configs use readable 32-bit fields. Wrappers validate and encode them
into compact parameter structs stored in `ggml_tensor::op_params`. Encoded
structs must fit the 64-byte GGML limit. Backend kernels decode the same struct;
they must not reinterpret public configs or independently infer geometry.

## Types and quantization

`ops_quantization_desc` separates storage, compute, accumulation type, block
size, quantization scheme, axis, and weight layout. Use
`ops_describe_quantization` instead of switching on tensor names.

Exact support is backend- and operator-specific, so capability probes are the
authority. The main policy is:

- activations are F32/F16, with BF16 on selected CPU/CUDA paths;
- normalization and accumulation-sensitive reductions use F32 accumulation;
- Conv/ConvTranspose can consume supported low-bit weights directly without a
  complete F16/F32 expansion;
- quantized rows must satisfy their GGML block size;
- an unsupported storage type is an explicit unsupported result, not a hidden
  conversion or CPU transfer.

See [runtime.md](runtime.md) and [NN quantization](../nn/quantization.md) for
the detailed ConvND storage matrix and conversion policy.

## Return values and errors

Public graph builders return `nullptr` when their inputs/configuration are
invalid or the selected backend has no compatible direct kernel and no semantic
fallback. Callers must check before graph expansion.

Kernel execution uses `ops_status`:

- `success`;
- `not_handled`;
- `invalid_request`;
- `unsupported`;
- `execution_failed`.

Backend kernels submit work to the backend stream/queue and should not perform
unnecessary device-wide waits. Synchronization belongs at graph, transfer, or
application boundaries.
