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

Quantization distinguishes storage, compute, and accumulation types. ConvND
storage support on CPU, CUDA, and SYCL includes Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, and MXFP4. Storage types are decoded into
`ops_quantization_desc` before backend capability decisions. Activations remain
F32/F16, with BF16 available for CPU and CUDA ConvND, and accumulation-sensitive
paths use F32.

ConvND accepts both channel-row and flattened-row storage. Channel rows keep one
input-channel row per kernel position; flattened rows keep the complete
`kernel_volume * channels` receptive field in one row. CPU, CUDA, and SYCL consume
both layouts directly without constructing an im2col activation matrix. Export
must obey the format block size:

- Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 require rows divisible by 32.
- Q2_K/Q3_K/Q4_K/Q5_K/Q6_K require rows divisible by 256.
- IQ4_NL and MXFP4 use 32-value rows; IQ4_XS uses 256-value rows.
- An incompatible preferred format falls back through compatible Q formats and
  finally F16.

Full model loading keeps packed convolution tensors in their GGUF storage type and
layout. CPU, CUDA, and SYCL Conv/ConvTranspose kernels read the supported formats
directly with F32 accumulation. No backend retains or constructs a
complete F16/F32 copy of a compressed convolution tensor.

CPU Q4_0/Q4_K/Q8_0 kernels decode packed values directly with AVX2 and use FMA
block dot or AXPY loops. ConvTranspose reuses only a transient 32-value SIMD
micro-tile on the worker stack; it does not retain or expand layer weights.
CUDA and SYCL decode blocks inside device kernels. ConvND direct kernels consume
channel rows without materializing a complete floating weight tensor.
Execution workspaces may contain reordered activations or partial accumulators,
but never a full expanded model weight.

ConvND CPU execution tiles the contiguous output-width dimension, reuses each
decoded output-channel row across every batch item, and dispatches activation
loads by type outside the hot loop so F32/F16/BF16 conversion is resolved at
compile time. Large CUDA and SYCL forward convolutions use a 16x16 implicit-GEMM
tile: one block/work-group computes 16 spatial positions by 16 output channels,
cooperatively caching both the generated input tile and decoded weight tile in
shared/local memory. The receptive field is generated on demand and never stored
as an im2col tensor. Smaller shapes use the one-output-channel spatial tile;
depthwise and other very small groups bypass synchronization and use the scalar
direct path.

On CUDA devices with compute capability 7.0 or newer, large F16-weight/F16-input
forward convolutions reuse the same 16x16 implicit tiles but accumulate them with
WMMA Tensor Core instructions in F32. Pointwise tiles use a 64-thread block so the
single WMMA warp does not leave six extra warps idle; larger receptive fields retain
256-thread cooperative loading to sustain reduction-tile bandwidth. The
capability result is cached per device; unsupported devices retain the tiled FMA
kernel. Quantized weights do not pass through F16 expansion and therefore keep
their quantized direct kernel.

The CUDA WMMA loader specializes pointwise convolution (`1x1`, unit stride, zero
padding): activation tiles map directly to `[spatial, channel]` and skip kernel
decomposition and convolution coordinate reconstruction. General kernels retain
the on-demand receptive-field generator.

The common 2D `3x3`, unit-stride, unit-dilation path is also tile-specialized
when the per-group input channel count is a multiple of 16. A reduction tile then
belongs to exactly one kernel position, so CUDA and SYCL compute the kernel index and
channel base once per tile instead of dividing and taking remainders for every
work-item. This applies to F32, quantized tiled FMA, and F16 WMMA execution.

CPU F32 pointwise convolution uses the physical Forge layout directly as
`[Cout,Cin] x [Cin,spatial] -> [Cout,spatial]` and dispatches to the existing
thread-local BLAS/MKL utility. It requires no im2col tensor, activation reorder,
or output transpose. Small pointwise shapes remain on the direct scalar/tiled
path to avoid BLAS launch overhead.

CPU F32 pointwise ConvTranspose uses the stored `[Cin,Cout]` rows directly as
`W^T x X`, again writing `[Cout,spatial]` without col2im, atomics, or temporary
activation matrices. SYCL pointwise ConvTranspose uses the same 16x16
space-by-output-channel implicit tile as forward convolution, with transposed
weight addressing selected at compile time.

CPU quantized pointwise convolution keeps decoded storage bounded to a 32-row
weight tile. Forward execution decodes output-channel tiles and multiplies each
tile directly by the channel-by-spatial activation matrix. ConvTranspose decodes
input-channel tiles and accumulates their transposed products into the output.
Both reuse a decoded tile across batches; neither expands the complete layer nor
creates an im2col/col2im workspace.

For ConvTranspose with unit dilation and stride greater than one, CPU, CUDA, and
SYCL derive the valid kernel phase from each output coordinate. They iterate only
kernel positions congruent with that phase instead of testing every position with
division and modulo. General dilation retains the fully general direct path. No
path creates a persistent decoded-weight copy.

CUDA quantized ConvTranspose2D additionally groups output positions by stride
phase and feeds each compact phase into a 16x16 space-by-output-channel tile. A
tile reduces only over kernel taps valid for that phase, so stride-2 execution does
not spend four times the arithmetic on inserted zeros. Packed weights are decoded
inside the tile; no col2im tensor or complete floating weight copy is created.

Windows SYCL presets compile both upstream GGML and Forge-owned kernels ahead of
time for the Raptor Lake-P `rplp` device through the `spir64_gen` target. A missing
`GGML_SYCL_DEVICE_ARCH` is a configuration error for Forge SYCL operators; generic
SPIR-V fallback and its first-run Intel driver compilation are intentionally not
accepted by these presets. The Windows preset also sets
`FORGE_SYCL_AOT_LINK_JOBS=20`, matching the current machine's logical processors,
so ocloc can use every logical processor during the build. It sets
`FORGE_SYCL_DEVICE_CODE_SPLIT=off` at device link time: each of `ggml-sycl.dll`
and `ggml_ops_ext_sycl.dll` contains one Raptor Lake-P device image. Editing a
Forge-owned SYCL kernel therefore relinks only the smaller Forge ops image; the
upstream GGML image is unaffected.

Batched Conv3D keeps logical five-dimensional geometry without changing
`GGML_MAX_DIMS`: the activation volume is flattened into one contiguous spatial
dimension and the contract carries width, height, depth, channels, groups, and
batch. This avoids creating a GGML fork that would obstruct provider updates.

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
includes Q4_0/Q4_K/Q8_0 with F32/F16 activations on legacy Conv1D paths, plus
small-group quantized ConvTranspose and F32 grouped/depthwise ConvTranspose.
`test_conv_nd` covers 2D/3D forward and transpose execution, asymmetric padding,
output padding, groups, batches, and all common Q/K formats across the three
project backends. `test_conv_nd --benchmark [CPU|CUDA0|SYCL0]` explicitly
synchronizes the backend and reports F32/Q4_0 flattened-row Conv2D and stride-2
ConvTranspose2D latency and effective GFLOP/s for a fixed 3x3 workload. It also
reports the F16-weight/F16-activation path used for CUDA Tensor Core validation.
`test_ops_foundation` validates affine RMSNorm, axis-aware L2
normalization, and gated activation composition.
`test_pool_nd` validates MaxPool/AvgPool 1D, 2D, and packed 3D execution,
including dilation, asymmetric padding, ceil mode, and both average-padding
policies across every available Forge backend.
`test_pad_nd` validates constant, reflect, replicate, and circular Pad1D/2D/3D
against a host reference on every available Forge backend.
`test_adaptive_pool_nd` validates uneven MaxPool/AvgPool output partitions in
one, two, and packed three dimensions on every available Forge backend.
`test_resize_nd` validates nearest, half-pixel N-linear, and aligned-corner
linear/bilinear/trilinear interpolation across every available Forge backend.
Absolute thresholds are intentionally not hard-coded because the same binary
targets heterogeneous devices; CI should compare these named cases against
per-device baselines.

## Development build

The Windows CUDA/SYCL/CPU preset has a focused Ops development companion:

```powershell
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev
```

The build preset targets `forge-ops-dev`, which contains the Forge Ops backend
libraries and Ops test executables. It does not build provider applications,
UI, tools, or examples. The regular full build preset remains the validation
path after changing GGML, the compiler/AOT target, shared ABI configuration, or
project-wide CMake structure.
