# Extending Forge Ops

## Decide whether a new Forge op is justified

Add a Forge virtual operator only when at least one condition holds:

- GGML has no equivalent semantics;
- several nodes can be fused to remove material intermediate tensors or kernel
  launches;
- inference needs a direct quantized path that GGML cannot express;
- a common operation needs consistent CPU/CUDA/SYCL behavior and shape rules;
- model code would otherwise contain backend-specific branches.

Do not add a virtual op for a mature native GGML primitive solely to make the
name look like PyTorch. Expose it through `nn::functional` instead. Embedding is
an example of a Forge public semantic wrapper that intentionally keeps native
GET_ROWS kernels.

## Files involved

A complete new operator normally touches:

```text
include/ops/contracts/<op>.h          semantic contract
include/ops/ops.h                     virtual op id, validation routing, public API
src/ops/core_ops/ops_<op>.cpp         public graph builder and fallback
src/ops/ops-cpu/<op>.cpp              CPU kernel
src/ops/ops-cuda/<op>.cu              CUDA kernel
src/ops/ops-sycl/<op>.cpp             SYCL kernel
src/ops/ops-*/ops_*.{cpp,cu}          declarations, probes, registry entries
src/ops/ops-*/CMakeLists.txt           backend source lists
src/ops/CMakeLists.txt                 core wrapper source
tests/ops_test/test_<op>.cpp           reference and backend tests
tests/ops_test/CMakeLists.txt           test registration
include/nn/functional/...              optional NN surface
docs/ops/api*.md                       English and Chinese API updates
```

An operator may intentionally use native GGML or support fewer backends, but
that decision and fallback behavior must be explicit and tested.

## 1. Define semantics and parameter encoding

The contract is the single source of truth for:

- input order and optional inputs;
- logical tensor layout;
- shape inference;
- dtype and quantized-storage rules shared by backends;
- public configuration validation;
- compact `op_params` encoding/decoding;
- output validation.

```cpp
// include/ops/contracts/example.h
namespace ggml_ops_ext {

struct ops_example_config {
    int32_t axis = 0;
    float scale = 1.0f;
};

struct ops_example_encoded_params {
    int32_t axis = 0;
    float scale = 1.0f;
};
static_assert(sizeof(ops_example_encoded_params) <= 64);

inline bool ops_encode_example(
    const ops_example_config& config,
    ops_example_encoded_params& encoded) {
    if (config.axis < 0 || config.axis >= GGML_MAX_DIMS) return false;
    encoded = {config.axis, config.scale};
    return true;
}

inline ops_status ops_validate_example(
    const ops_request& request,
    /* optional decoded description */) {
    if (!request.srcs || request.n_srcs < 1 || !request.srcs[0]) {
        return ops_status::error(
            ops_status_code::invalid_request, "Example input is missing");
    }
    // Decode params, validate source shape/type, then validate request.output
    // when it is non-null.
    return ops_status::ok();
}

} // namespace ggml_ops_ext
```

GGML reserves 64 bytes for `op_params`; never store pointers, host-only objects,
`size_t`, or ABI-dependent structures there. Use fixed-width fields and
`memcpy` for float/integer bit transport. Validate overflow before narrowing.

Backends must call the shared contract. A CUDA or SYCL kernel must not silently
accept a shape that CPU rejects, nor reinterpret axis or padding order.

## 2. Reserve a virtual operator ID

Add a value before `GGML_OP_OPS_VIRT_COUNT` in `ops_virt_op_type`. Virtual IDs
start at 2000 to avoid GGML core op values. Because the value crosses shared
library boundaries, rebuild every Forge Ops backend after changing the enum.

If the operator belongs to `ops_validate_request_contract`, add its validation
case there. Geometry families may use a dedicated probe just like ConvND,
PoolND, PadND, and ResizeND.

## 3. Implement the public graph builder

The core wrapper validates public inputs/config, probes the selected backend,
creates the output shape, stores encoded parameters, and defines fallback
policy.

```cpp
ggml_tensor* ggml_ops_example(
    ggml_context* ctx, ggml_tensor* input,
    const ggml_ops_ext::ops_example_config& config,
    ggml_backend_t backend) {
    if (!ctx || !input) return nullptr;

    ggml_ops_ext::ops_example_encoded_params params;
    if (!ggml_ops_ext::ops_encode_example(config, params)) return nullptr;

    ggml_tensor* sources[] = {input};
    if (ggml_ops_backend_supports_op(
            backend, ggml_ops_ext::GGML_OP_OPS_VIRT_EXAMPLE,
            sources, 1, &params, sizeof(params))) {
        int64_t shape[GGML_MAX_DIMS] = {
            input->ne[0], input->ne[1], input->ne[2], input->ne[3]};
        ggml_tensor* output = ggml_ops_ext::ops_new_virtual_node(
            ctx, ggml_ops_ext::GGML_OP_OPS_VIRT_EXAMPLE,
            input->type, ggml_n_dims(input), shape, 1, sources);
        ggml_set_op_params(output, &params, sizeof(params));
        return output;
    }

    // Return a semantically equivalent GGML composition when one exists.
    // Otherwise return nullptr. Never move the op to CPU implicitly.
    return nullptr;
}
```

Fallback order belongs here, not inside model code. A fallback must preserve
shape, dtype, masking, mutation, and dependency semantics. Avoid a fallback
that creates a known prohibitive workspace merely to avoid returning
`nullptr`.

## 4. Implement backend kernels

The convenience form is a boolean entry point adapted by `make_ops_kernel`:

```cpp
bool ops_cpu_op_example(ggml_backend_t backend, ggml_tensor* node) {
    if (!node) return false;
    ops_request request{
        ggml_backend_get_device(backend),
        GGML_OP_OPS_VIRT_EXAMPLE,
        node->src, 1, node->op_params, sizeof(node->op_params), node};
    if (!ops_validate_example(request)) return false;
    // Execute.
    return true;
}
```

Implement an `ops_kernel_execute_t` returning `ops_status` directly when the
kernel needs richer errors or `ops_execution_context` resources.

### CPU rules

- Read thread count with `ggml_ops_ext::cpu::backend_thread_count(backend)`;
  do not use process-global thread configuration.
- Parallelize outer independent work and keep temporary storage bounded per
  worker/tile.
- Dispatch dtype and quantization outside hot loops.
- Use existing AVX2/F16C and quantized-block helpers when applicable.
- Do not expand a complete quantized weight tensor as a shortcut.

### CUDA rules

- Obtain device and stream through the functions in `ops_cuda_common.cuh`.
- Set the device before launch and submit to the backend stream.
- Use stream-ordered temporary allocation helpers when workspace is required.
- Report launch errors, but do not call `cudaDeviceSynchronize` inside the op.
- Route by capability/shape/type, not a product name or PCI ID.

### SYCL rules

- Obtain the queue through `ops_sycl.h` bridge functions.
- Submit asynchronously; do not call `queue.wait()` inside every operator.
- Bound local memory and work-group size to values guaranteed by the targeted
  architecture class.
- Avoid fixed subgroup assumptions unless the kernel explicitly requests and
  probes them.
- Keep Forge kernels in `ggml_ops_ext_sycl`; do not patch upstream GGML for a
  Forge-only operation.

All backends must support nontrivial lengths that cross tile boundaries and
must implement exact padding/domain semantics at the first and last tile.

## 5. Add probes and registry entries

Each backend file `ops_cpu.cpp`, `ops_cuda.cu`, or `ops_sycl.cpp` needs:

1. an entry-point declaration;
2. a probe function or shared contract probe;
3. a named kernel table entry;
4. the source file in that backend's CMake list.

```cpp
static ops_probe_result supports_example(const ops_request& request) {
    const ops_status valid = ops_validate_example(request);
    if (!valid) return {false, 0, valid.message};
    const ggml_type type = request.srcs[0]->type;
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16;
}

static const ops_kernel_entry CPU_KERNELS[] = {
    make_ops_kernel<ops_cpu_op_example>(
        GGML_OP_OPS_VIRT_EXAMPLE,
        "cpu.example", supports_example, 100),
};
```

Kernel names should be stable diagnostic identifiers. Priority selects between
multiple compatible implementations; support checks must not benchmark or
inspect a concrete product name.

Registration matches `backend_name_prefix` such as `CPU`, `CUDA`, or `SYCL`.
Do not register the same prefix twice.

## 6. Quantized storage

If the op accepts weights:

1. define the semantic weight layout;
2. call `ops_describe_quantization` once outside hot loops;
3. enforce row block alignment in the contract/probe;
4. decode only tiles/blocks needed by current work;
5. accumulate reductions in F32 unless a documented policy says otherwise;
6. add converter and NN `Parameter::Usage` support if this is a new parameter
   family.

Quantization support means direct execution from packed storage. Expanding all
weights at model load or per invocation is not considered support.

## 7. Expose through NN

General model code should not include backend kernel headers. Add a
`nn::functional` wrapper and, when state is involved, an `nn::Module` layer.
NN owns convenient argument defaults and model-facing layout; Ops remains the
semantic/backend boundary.

## 8. Test correctly

A new operator test must:

- call `ggml_backend_load_all` in dynamic-backend builds;
- initialize linked Forge backend libraries and acquire the Ops hook;
- require at least one device to run, so an empty enumeration cannot pass;
- enumerate CPU, CUDA, and SYCL without matching concrete GPU model names;
- test every promised dtype and quantized type;
- test tiny, odd, boundary, multi-tile, batched, grouped, and non-default
  parameter cases as applicable;
- compare against an independent host reference or a trusted semantic
  composition;
- synchronize before timing;
- separate correctness tolerances by dtype/quantization, not by device model;
- compare fused performance with the previous composition and reject clear
  regressions.

Register the test in `tests/ops_test/CMakeLists.txt` and in
`REGISTERED_OPS_TEST_TARGETS`.

## 9. Build and validation

Use the project preset and focused development target:

```powershell
cmake --preset x64-windows-cuda-sycl-cpu-dl-release-f16
cmake --build --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --target test_<op> -j 32
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev -R ops.test_<op> --output-on-failure
ctest --preset x64-windows-cuda-sycl-cpu-dl-release-f16-ops-dev --output-on-failure
git diff --check
```

Changing a public Ops header rebuilds many dependents. Changing only one CUDA
or SYCL source should rebuild that smaller Forge backend library, not upstream
GGML. Run an end-to-end provider model after unit tests when the operator is on
a real inference path.

## Completion checklist

- Semantics and layout are written before kernel code.
- Encoded parameters are fixed-width and no larger than 64 bytes.
- Every backend uses the same contract and validates output.
- Unsupported cases are explicit; there is no silent CPU transfer.
- Fallback semantics and workspace cost are acceptable.
- Quantized weights remain packed.
- Kernels submit asynchronously and use the backend-owned stream/queue.
- Routing contains no concrete device model checks.
- CPU/CUDA/SYCL correctness and performance are measured.
- NN wrappers and English/Chinese documents are updated.
