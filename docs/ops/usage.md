# Using Ops

Custom operations are exposed through `include/ops/ops.h`. Most model code
should use the corresponding `nn::functional` or `nn::layers` wrapper; direct
Ops use is intended for lower-level graph construction and backend work.

## Runtime lifecycle

Backend libraries register their kernels when loaded. A standalone Ops runtime
must keep the Forge bridge installed while custom graphs may execute:

```cpp
ggml_ops_ext::acquire_ops_hook();

ggml_tensor* output = ggml_ops_mish(context, input, backend);
if (!output) {
    // The concrete backend, dtype, or shape is unsupported.
}

ggml_cgraph* graph = ggml_new_graph_custom(context, GGML_DEFAULT_GRAPH_SIZE, false);
ggml_build_forward_expand(graph, output);
ggml_status status = ggml_ops_ext::ops_backend_graph_compute(backend, graph);

ggml_ops_ext::release_ops_hook();
```

Every successful acquire must have a matching release. Higher-level category
runtimes manage this lifetime themselves.

## Capability checks

Use `ggml_ops_backend_supports_op` before creating a node when only source
tensors and encoded parameters are known. Use `probe_ops_kernel` with an
`ops_request` when the output tensor is available and its dtype and shape must
also be validated. Unsupported quantization, layout, dtype, or shape is an
explicit failure; callers must not assume silent CPU fallback.

CPU thread count is backend-instance state and can be configured with
`ggml_ops_ext_cpu_set_n_threads`. Distinct backend instances have independent
execution lanes; sessions sharing one backend instance serialize whole-graph
execution.

## Inference convolution family

Forge provides `Conv2D`, `ConvTranspose2D`, `Conv3D`, and
`ConvTranspose3D` in addition to the existing 1D family. All variants support
stride, per-side padding, dilation, groups, depthwise grouping, optional bias,
and transposed-convolution output padding.

New ConvND weights use channel-row storage:

```text
ConvND:          [input_channels/groups, kernel_volume, output_channels]
ConvTransposeND: [output_channels/groups, kernel_volume, input_channels]
```

Conv2D activations use `[width, height, channels, batch]`. Batched Conv3D has
five logical dimensions, while GGML tensors have four physical dimensions, so
its public physical representation is `[width*height*depth, channels, batch]`.
`ops_conv_nd_config::input_size` preserves the logical spatial geometry.

```cpp
ggml_ops_ext::ops_conv_nd_config config;
config.spatial_dims = 3;
config.input_size[0] = width;
config.input_size[1] = height;
config.input_size[2] = depth;
config.kernel_size[0] = config.kernel_size[1] = config.kernel_size[2] = 3;

ggml_tensor* output = ggml_ops_conv_3d(
    context, weight, packed_input, config, backend, bias);
```

`nn::Conv2d`, `nn::ConvTranspose2d`, `nn::Conv3d`, and
`nn::ConvTranspose3d` accept the corresponding parameter layouts. `nn::Conv2d`
detects channel-row and flattened-row storage and sends both through direct
ConvND kernels; existing flattened GGUF files therefore require no migration.

## Pooling family

Forge provides native MaxPool and AvgPool for one, two, and three spatial
dimensions. They support per-axis kernel size, stride, dilation, asymmetric
padding, ceil mode, and the AvgPool `count_include_pad` policy. Pool2D uses
`[width,height,channels,batch]`; Pool3D uses the same packed
`[width*height*depth,channels,batch]` convention as Conv3D.

```cpp
ggml_ops_ext::ops_pool_nd_config config;
config.spatial_dims = 2;
config.input_size[0] = width;
config.input_size[1] = height;
config.kernel_size[0] = config.kernel_size[1] = 3;
config.stride[0] = config.stride[1] = 2;

ggml_tensor* output = nn::functional::max_pool2d(
    context, input, config, backend);
```

CPU accepts F32, F16, and BF16. CUDA accepts F32, F16, and BF16; SYCL accepts
F32 and F16. Every output is reduced directly from the input, so pooling needs
no im2col tensor or other activation-sized workspace.

Adaptive MaxPool and AvgPool are available for the same 1D, 2D, and packed 3D
layouts. Each output interval starts at `floor(out * input / output)` and ends
at `ceil((out + 1) * input / output)`, matching the uneven and overlapping
window semantics used by PyTorch. They also require no temporary workspace.

Resize1D/2D/3D supports nearest-neighbor and N-linear interpolation (linear,
bilinear, and trilinear). Linear interpolation supports both half-pixel
coordinates and `align_corners=true`; nearest uses floor-based asymmetric
coordinates. The native API uses the standard Forge spatial layouts and emits
one virtual node instead of materializing index and weight tensors.

## Padding family

Pad1D, Pad2D, and packed Pad3D support constant, reflect, replicate, and
circular modes with independent padding on both sides of every spatial axis.
Reflect padding follows the PyTorch constraint that each side must be smaller
than the corresponding input dimension. Padding maps each output coordinate
directly to an input coordinate and allocates no intermediate workspace.

```cpp
ggml_ops_ext::ops_pad_nd_config config;
config.spatial_dims = 2;
config.input_size[0] = width;
config.input_size[1] = height;
config.padding_before[0] = 1;
config.padding_after[0] = 2;
config.mode = ggml_ops_ext::ops_pad_mode::reflect;

ggml_tensor* output = nn::functional::pad2d(
    context, input, config, backend);
```

## Foundational operators

Common inference building blocks use native GGML nodes where GGML already has
the correct primitive, rather than duplicating backend kernels:

- affine RMSNorm;
- affine GroupNorm;
- axis-aware L2 normalization;
- SwiGLU, GeGLU, and ReGLU through the generic gated activation.

Model code should normally use the corresponding `nn::functional` functions.
