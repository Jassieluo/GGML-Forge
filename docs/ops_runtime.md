# Ops Runtime Architecture

The custom operator runtime separates operator semantics from backend kernels:

```text
nn/model -> op contract -> virtual GGML node -> kernel dispatcher -> backend kernel
```

## Contracts

An op contract owns parameter encoding, validation, weight-layout interpretation,
and output-shape inference. Backend capability checks do not redefine these rules.
`Conv1D` and `ConvTranspose1D` use the shared contract in
`include/ops/contracts/conv1d.h`.

Contracts are grouped by operator semantics under `include/ops/contracts/`:
convolution, activation, normalization, and attention/position encoding. The
dispatcher passes an `ops_request` containing inputs, parameters, and the optional
output tensor. Graph-time probes therefore validate output dtype and shape, while
pre-build capability queries may omit the output.

## Kernel registry

Each backend registers named kernels with a priority, probe function, and execute
function. Registration is frozen when the first ops hook is acquired. Dispatch is
therefore lock-free during graph execution. Multiple implementations of one op can
be selected by priority without adding backend branches to model code.

Kernel execution receives an `ops_execution_context`. Mutable resources such as
streams and scratch memory belong to the backend instance or execution lane, never
to a process-global kernel cache.

## Quantization

Quantization distinguishes storage, compute, and accumulation types. Current
convolution storage support is Q4_0, Q4_K, and Q8_0. Storage types are decoded into
`ops_quantization_desc` before backend capability decisions. Activations remain
F32/F16 and accumulation-sensitive paths use F32.

Packed convolution rows use the channel dimension as the quantization row. Export
must obey the format block size:

- Q4_0 and Q8_0 require rows divisible by 32.
- Q4_K requires rows divisible by 256.
- Incompatible Q4_K rows fall back to Q8_0, then F16.

Full model loading keeps packed convolution tensors in their GGUF storage type and
layout. CPU, CUDA, and SYCL Conv/ConvTranspose kernels read Q4_0, Q4_K, Q8_0, and
F16 weights directly with F32 accumulation. No backend retains or constructs a
complete F16/F32 copy of a compressed convolution tensor.

CPU Q4_0/Q4_K/Q8_0 kernels decode packed values directly with AVX2 and use FMA
block dot or AXPY loops. ConvTranspose reuses only a transient 32-value SIMD
micro-tile on the worker stack; it does not retain or expand layer weights.
CUDA and SYCL decode blocks inside device kernels;
one 32-lane warp/work-group cooperatively accumulates each quantized convolution
output in F32.
Execution workspaces may contain reordered activations or partial accumulators,
but never a full expanded model weight.

Quantized ConvTranspose selects its GPU execution shape from the reduction size.
Groups with at most eight input channels use one scalar work-item per output;
larger reductions use the cooperative 32-lane kernel. Quantized Conv rows are
block-aligned by format and therefore keep the cooperative path.

## Concurrency

The kernel registry is immutable after initialization. Model-specific immutable
transforms belong to the model instance; the ops runtime has no process-global
prepared-weight cache.
`ops_backend_graph_compute` assigns an execution lane to each backend instance:
sessions sharing one backend are serialized for the whole graph, while distinct
backend instances and heterogeneous devices remain concurrent. Custom kernel
dispatch reuses the same recursive lane.

CPU thread counts are also backend-instance state. The runtime configures both
GGML CPU execution and custom CPU ops with the same `n_threads` value through
`ggml_backend_cpu_set_n_threads` and `ggml_ops_ext_cpu_set_n_threads`; no
process-global OpenMP setting is required. Every custom CPU parallel region,
including MKL-backed ConvTranspose GEMM, consumes this backend-local value.
Thread-count reads use immutable copy-on-write snapshots, so independent CPU
sessions do not serialize on a global configuration mutex during inference.

CUDA uses stream-ordered allocations. SYCL operators submit asynchronously and
use backend synchronization at graph or data-transfer boundaries instead of
calling `queue.wait()` inside each operator. Session state and backend instances
remain runtime-owned.

## Performance validation

`test_ops_quantized_conv --benchmark [CPU|CUDA0|SYCL0]` reports large-channel
Conv/ConvTranspose and small-group ConvTranspose timings. Correctness coverage
includes Q4_0/Q4_K/Q8_0 with F32/F16 activations on CPU, CUDA, and SYCL, plus
small-group quantized ConvTranspose and F32 grouped/depthwise ConvTranspose.
Absolute thresholds are intentionally not hard-coded because the same binary
targets heterogeneous devices; CI should compare these named cases against
per-device baselines.
