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
